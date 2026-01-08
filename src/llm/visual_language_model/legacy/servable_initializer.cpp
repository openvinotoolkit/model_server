//*****************************************************************************
// Copyright 2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/genai/visual_language/vision_encoder.hpp>
#include <openvino/genai/visual_language/embeddings_model.hpp>
#include <openvino/genai/visual_language/inputs_embedder.hpp>
#include <openvino/openvino.hpp>

#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/tanh.hpp>
#include <openvino/op/transpose.hpp>

#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../../json_parser.hpp"
#include "../../../logging.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../../status.hpp"
#include "servable.hpp"
#include "servable_initializer.hpp"

namespace ovms {

std::pair<size_t, size_t> get_kv_axes_pos(std::shared_ptr<const ov::Model> model) {
    // sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    // therefore usually seq_length_axis = 2 and batch = 0
    std::pair<size_t, size_t> kv_pos { 0u, 2u };

    // "ReadValue" node is KV cache representation in stateful model
    std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

    for (const auto& op : model->get_ops()) {
        // check input size, as in LoRA adapters case it could be 0
        if (op->get_type_name() != kv_node_type_name || op->get_input_size() < 1) {
            continue;
        }

        // Shape example: [-1,4,0,64]
        auto shape = op->get_input_partial_shape(0);

        for (size_t i = 0; i < shape.rank().get_length(); i++) {
            // Find axis = 0. This would be sequence length axis.
            if (shape[i] == 0) {
                kv_pos.second = i;
            } else if (shape[i].is_dynamic()) {
                // Dynamic axis is a batch
                kv_pos.first = i;
            }
        }
        break;
    }

    return kv_pos;
}

Status VisualLanguageModelLegacyServableInitializer::initialize(std::shared_ptr<GenAiServable>& servable, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) {
    std::string parsedModelsPath;
    auto status = parseModelsPath(parsedModelsPath, nodeOptions.models_path(), graphPath);
    if (!status.ok()) {
        return status;
    }

    servable = std::make_shared<VisualLanguageModelLegacyServable>();
    auto properties = std::static_pointer_cast<VisualLanguageModelLegacyServableProperties>(servable->getProperties());

    properties->modelsPath = parsedModelsPath;
    std::filesystem::path modelGenerationConfigPath = std::filesystem::path(parsedModelsPath) / "generation_config.json";
    if (std::filesystem::exists(modelGenerationConfigPath)) {
        properties->baseGenerationConfig = ov::genai::GenerationConfig(modelGenerationConfigPath.string());
    }
    properties->schedulerConfig.max_num_batched_tokens = nodeOptions.max_num_batched_tokens();
    properties->schedulerConfig.cache_size = nodeOptions.cache_size();
    properties->schedulerConfig.dynamic_split_fuse = nodeOptions.dynamic_split_fuse();
    properties->schedulerConfig.max_num_seqs = nodeOptions.max_num_seqs();
    properties->schedulerConfig.enable_prefix_caching = nodeOptions.enable_prefix_caching();

    properties->device = nodeOptions.device();

    if (nodeOptions.has_draft_max_num_batched_tokens() || nodeOptions.has_draft_cache_size() || nodeOptions.has_draft_dynamic_split_fuse() || nodeOptions.has_draft_max_num_seqs() || nodeOptions.has_draft_block_size() || nodeOptions.has_draft_device()) {
        // Consider moving draft parameters to separate structure in node options, so it's validated on the proto level
        SPDLOG_ERROR("Draft model path is not provided, but draft scheduler options are set.");
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), properties->pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        return status;
    }

//#define NEW_CONSTRUCTORS
//#define NEW_CONSTRUCTORS_V2
#define NEW_CONSTRUCTORS_V3

    try {
#ifdef NEW_CONSTRUCTORS
        // TODO: Make ov::Core a singleton shared among single models and  all calculators (+ possibly genai?)
        ov::genai::CompiledModelsMap compiledModelsMap;
        
        ov::Core core;
        auto language_model_path = std::filesystem::path(parsedModelsPath) / "openvino_language_model.xml";
        std::cout << "read_model for llm model" << std::endl;
        auto language_model = core.read_model(language_model_path, {}, properties->pluginConfig);
        auto kv_pos = get_kv_axes_pos(language_model);
        std::cout << "compile for llm model" << std::endl;
        compiledModelsMap["language"] = core.compile_model(language_model, properties->device, properties->pluginConfig);

        auto vision_model_path = std::filesystem::path(parsedModelsPath) / "openvino_vision_embeddings_model.xml";
        std::cout << "read_model for vision embeddings model" << std::endl;
        auto vision_model = core.read_model(vision_model_path, {}, properties->pluginConfig);
        const std::string vision_embeddings_device = nodeOptions.vision_embeddings_device().size() > 0 ? nodeOptions.vision_embeddings_device() : properties->device;
        std::cout << "compile for vision embeddings model" << std::endl;
        compiledModelsMap["vision_embeddings"] = core.compile_model(vision_model, 
            vision_embeddings_device, properties->pluginConfig);

        auto text_embeddings_model_path = std::filesystem::path(parsedModelsPath) / "openvino_text_embeddings_model.xml";
        std::cout << "read_model for text embeddings model" << std::endl;
        auto text_embeddings_model = core.read_model(text_embeddings_model_path, {}, properties->pluginConfig);
        
        //?
        const float scale_emb = 1.0f; // to be read from config.json -> scale_emb
        ov::preprocess::PrePostProcessor ppp(text_embeddings_model);
        ppp.output().postprocess().custom([scale_emb](const ov::Output<ov::Node>& node) {
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, scale_emb);
            return std::make_shared<ov::op::v1::Multiply>(node, constant);
        });
        ppp.build();
        
        const std::string text_embeddings_device = nodeOptions.text_embeddings_device().size() > 0 ? nodeOptions.text_embeddings_device() : properties->device;
        std::cout << "compile for text embeddings model" << std::endl;
        compiledModelsMap["text_embeddings"] = core.compile_model(text_embeddings_model, text_embeddings_device, properties->pluginConfig);

        SPDLOG_ERROR("Selecting Devices: Language: {}, Vision Embeddings: {}, Text Embeddings: {}", properties->device, vision_embeddings_device, text_embeddings_device);
        properties->pipeline = std::make_shared<ov::genai::VLMPipeline>(parsedModelsPath, compiledModelsMap, kv_pos.first, kv_pos.second, properties->pluginConfig);
#elif defined(NEW_CONSTRUCTORS_V2)
        ov::genai::DeviceMapping deviceMapping{
            {"language", properties->device},
            {"text_embeddings", nodeOptions.text_embeddings_device().empty() ? properties->device : nodeOptions.text_embeddings_device()},
            {"vision_embeddings", nodeOptions.vision_embeddings_device().empty() ? properties->device : nodeOptions.vision_embeddings_device()}
        };
        properties->pipeline = std::make_shared<ov::genai::VLMPipeline>(parsedModelsPath, deviceMapping, properties->pluginConfig);
#elif defined(NEW_CONSTRUCTORS_V3)
        // Construct Vision Encoder
        auto visionEncoder = std::make_shared<ov::genai::VisionEncoder>(parsedModelsPath, nodeOptions.vision_embeddings_device().empty() ? properties->device : nodeOptions.vision_embeddings_device(), properties->pluginConfig);
        auto textEmbeddingsModel = std::make_shared<ov::genai::EmbeddingsModel>(parsedModelsPath, nodeOptions.text_embeddings_device().empty() ? properties->device : nodeOptions.text_embeddings_device(), properties->pluginConfig);
        ov::genai::Tokenizer tokenizer(parsedModelsPath);

        ov::genai::InputsEmbedder inputsEmbedder(tokenizer, visionEncoder, textEmbeddingsModel, parsedModelsPath);

        ov::genai::DeviceMapping deviceMapping{
            {"language", properties->device},
            {"text_embeddings", nodeOptions.text_embeddings_device().empty() ? properties->device : nodeOptions.text_embeddings_device()},
            {"vision_embeddings", nodeOptions.vision_embeddings_device().empty() ? properties->device : nodeOptions.vision_embeddings_device()}
        };
        properties->pipeline = std::make_shared<ov::genai::VLMPipeline>(parsedModelsPath, deviceMapping, properties->pluginConfig);
#else
        properties->pipeline = std::make_shared<ov::genai::VLMPipeline>(parsedModelsPath, properties->device, properties->pluginConfig);
#endif
        properties->tokenizer = properties->pipeline->get_tokenizer();
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {} exception: {}", parsedModelsPath, e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {}", parsedModelsPath);
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    properties->legacyExecutor = std::make_shared<VisualLanguageModelLegacyExecutorWrapper>(properties->pipeline);
    if (nodeOptions.has_max_tokens_limit()) {
        properties->maxTokensLimit = nodeOptions.max_tokens_limit();
    }
    properties->bestOfLimit = nodeOptions.best_of_limit();
    properties->maxModelLength = parseMaxModelLength(parsedModelsPath);
    return StatusCode::OK;
}

}  // namespace ovms
