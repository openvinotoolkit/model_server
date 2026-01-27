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

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "sidepacket_servable.hpp"
#include "logging.hpp"
#include <spdlog/spdlog.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>

#include <vector>
#include <string>
#include <memory>

#include "json_parser.hpp"
#include "status.hpp"
#include "config.hpp"

#include "filesystem.hpp"

using namespace ov::genai;
using namespace ov;

namespace ovms {

#define SET_TOKEN_ID(token, token_id_name)                                                                                            \
    if (modelConfig.HasMember(token_id_name) && modelConfig[token_id_name].IsInt64() && modelConfig[token_id_name].GetInt64() != 0) { \
        token = modelConfig[token_id_name].GetInt64();                                                                                \
    }

#define SET_TOKEN(token)                                                                                                       \
    if (!token.has_value()) {                                                                                                  \
        if (tokenizerConfig.HasMember(#token) && tokenizerConfig[#token].IsString()) {                                         \
            auto tokenizedInputs = tokenizer->encode(tokenizerConfig[#token].GetString());                                     \
            if (tokenizedInputs.input_ids.get_size() == 1 && tokenizedInputs.input_ids.get_element_type() == ov::element::i64) \
                token = reinterpret_cast<int64_t*>(tokenizedInputs.input_ids.data())[0];                                       \
            else                                                                                                               \
                SPDLOG_DEBUG("Parsing {} token from tokenizer_config.json failed", #token);                                    \
        }                                                                                                                      \
    }

SidepacketServable::SidepacketServable(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) {
    return;
}

struct KVAxesPosition {
    size_t batch;
    size_t seq_len;
};

struct KVDesc {
    uint32_t max_prompt_len;
    uint32_t min_response_len;
};

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        std::optional<ov::Any> found = std::make_optional(it->second);
        config.erase(it);
        return found;
    }
    return std::nullopt;
}

std::optional<uint32_t> pop_int_and_cast(ov::AnyMap& config, const std::string& key) {
    auto anyopt = pop_option(config, key);
    if (anyopt.has_value()) {
        const auto any = anyopt.value();
        int64_t value;
        // NB: Integer value coming from python has int64_t datatype
        if (any.is<int64_t>()) {
            value = any.as<int64_t>();
        } else if (any.is<int>()) {
            value = any.as<int>();
        } else {
            OPENVINO_THROW("Failed to extract " + key + ". Type mismatch: expected types: int or int64_t");
        }
        if (value < 0) {
            OPENVINO_THROW(key + " cannot be negative!");
        }
        return std::make_optional(static_cast<uint32_t>(value));
    }
    return std::nullopt;
}

KVAxesPosition get_kv_axes_pos(std::shared_ptr<const ov::Model> model) {
    // sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    // therefore usually seq_length_axis = 2 and batch = 0
    KVAxesPosition kv_pos { 0u, 2u };

    // "ReadValue" node is KV cache representation in stateful model
    std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

    for (const auto &op : model->get_ops()) {
        // check input size, as in LoRA adapters case it could be 0
        if (op->get_type_name() != kv_node_type_name || op->get_input_size() < 1) {
            continue;
        }

        // Shape example: [-1,4,0,64]
        auto shape = op->get_input_partial_shape(0);

        for (size_t i = 0; i < shape.rank().get_length(); i++) {
            // Find axis = 0. This would be sequence length axis.
            if (shape[i] == 0) {
                kv_pos.seq_len = i;
            } else if (shape[i].is_dynamic()) {
                // Dynamic axis is a batch
                kv_pos.batch = i;
            }
        }
        break;
    }

    return kv_pos;
}

void update_config(ov::AnyMap& config, const std::pair<std::string, ov::Any>& pair) {
    if (config.count(pair.first) == 0) {
        config.insert(pair);
    }
}

void update_npu_config_text_embedding(ov::AnyMap& config,
                                      const KVAxesPosition& kv_pos,
                                      const KVDesc& kv_desc) {
    update_config(config, {"NPU_USE_NPUW", "YES"});
    update_config(config, {"NPUW_LLM", "YES"});
    update_config(config, {"NPUW_LLM_BATCH_DIM", kv_pos.batch});
    update_config(config, {"NPUW_LLM_SEQ_LEN_DIM", kv_pos.seq_len});

    update_config(config, {"NPUW_LLM_MAX_PROMPT_LEN", kv_desc.max_prompt_len});
    update_config(config, {"NPUW_LLM_MIN_RESPONSE_LEN", kv_desc.min_response_len});
    update_config(config, {"NPUW_LLM_SHARED_HEAD", "NO"});

    update_config(config, {"NPUW_TEXT_EMBED", "YES"});
}

void get_npu_text_embedding_config(ov::AnyMap& properties,
                                   const KVAxesPosition& kv_pos,
                                   KVDesc& kv_desc,
                                   const TextEmbeddingPipeline::Config& text_embed_config) {
    if (text_embed_config.max_length.has_value()) {
        kv_desc.max_prompt_len = text_embed_config.max_length.value();
    } else {
        kv_desc.max_prompt_len = pop_int_and_cast(properties, "MAX_PROMPT_LEN").value_or(1024u);
    }
    kv_desc.min_response_len = kv_desc.max_prompt_len;
    update_npu_config_text_embedding(properties, kv_pos, kv_desc);
}


void set_node_name(const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->set_friendly_name(name);
    node->get_output_tensor(0).set_names({name});
}

/**
 * CLS pooling slices first element from seq_length dimension
 * [batch_size, seq_length, hidden_size] -> [batch_size, seq_length[0], hidden_size]
 * [10, 5, 768] -> [10, 768]
 */
std::shared_ptr<op::Op> get_cls_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node) {
    auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto stop = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

    auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

    auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
}

std::shared_ptr<op::Op> get_mean_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node,
                                            const ov::Output<ov::Node>& attention_mask) {
    auto shape_of = std::make_shared<op::v3::ShapeOf>(last_hidden_state_node);

    auto unsqueze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto unsqueze = std::make_shared<op::v0::Unsqueeze>(attention_mask, unsqueze_axis);

    auto input_mask_expanded = std::make_shared<op::v3::Broadcast>(unsqueze, shape_of);

    auto input_mask_expanded_convert =
        std::make_shared<op::v0::Convert>(input_mask_expanded, last_hidden_state_node.get_element_type());

    auto last_hidden_node_with_applied_attention_mask =
        std::make_shared<op::v1::Multiply>(last_hidden_state_node, input_mask_expanded_convert->outputs()[0]);

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto sum_hidden_state = std::make_shared<op::v1::ReduceSum>(last_hidden_node_with_applied_attention_mask, axis_1);

    // f32 overflow possible
    // ReduceMean might help with overflow but its precision diverges from LlamaIndex
    auto sum_expanded_mask = std::make_shared<op::v1::ReduceSum>(input_mask_expanded_convert, axis_1);

    auto nearest_to_zero =
        std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1e-12});
    auto max_expanded_mask = std::make_shared<op::v1::Maximum>(sum_expanded_mask, nearest_to_zero);

    // shape: [batch_size, hidden_state_size]
    return std::make_shared<op::v1::Divide>(sum_hidden_state, max_expanded_mask);
}

std::shared_ptr<op::Op> get_last_token_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node,
                                                  const ov::Output<ov::Node>& attention_mask,
                                                  const TextEmbeddingPipeline::Config& config) {
    const auto left_padding = config.padding_side.has_value() && config.padding_side.value() == "left";

    // shortcut for left padding. We can slice last token directly
    if (left_padding) {
        auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto stop = std::make_shared<op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{1},
                                                       std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
        auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

        auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

        auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
    }

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(attention_mask, axis_1);
    auto subtract_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto subtract = std::make_shared<op::v1::Subtract>(reduce_sum, subtract_1);

    return std::make_shared<op::v8::Gather>(last_hidden_state_node, subtract, axis_1, 1);
}

std::shared_ptr<op::Op> create_post_ops(const ov::Output<ov::Node>& input,
                                        const ov::Output<ov::Node>& attention_mask,
                                        const TextEmbeddingPipeline::Config& config) {
    if (config.pooling_type == TextEmbeddingPipeline::PoolingType::CLS) {
        return get_cls_pooling_op(input);
    } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::MEAN) {
        return get_mean_pooling_op(input, attention_mask);
    } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::LAST_TOKEN) {
        return get_last_token_pooling_op(input, attention_mask, config);
    }

    OPENVINO_THROW("Pooling type is not supported");
}

std::shared_ptr<op::Op> create_normalize_ops(const ov::Output<ov::Node>& input,
                                             const TextEmbeddingPipeline::Config& config) {
    if (config.normalize) {
        auto axis_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
        return std::make_shared<op::v0::NormalizeL2>(input, axis_const, 1e-12, op::EpsMode::MAX);
    }
    return std::dynamic_pointer_cast<op::Op>(input.get_node_shared_ptr());
}

std::shared_ptr<ov::Model> create_post_model(std::shared_ptr<ov::Model> model,
                                             const TextEmbeddingPipeline::Config& config) {
    auto output_node = model->outputs()[0];
    auto output_shape = output_node.get_partial_shape();
    auto input_param = std::make_shared<ov::op::v0::Parameter>(output_node.get_element_type(),
                                                               ov::PartialShape{1, -1, output_shape[2]});
    set_node_name(input_param, "embedding_hidden_state");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    set_node_name(attention_mask, "attention_mask");

    auto post_output = create_post_ops(input_param, attention_mask, config);
    auto post_normalize_output = create_normalize_ops(post_output, config);
    OPENVINO_ASSERT(post_normalize_output != nullptr);

    auto result_node = std::make_shared<ov::op::v0::Result>(post_normalize_output);
    set_node_name(result_node, "last_hidden_state");
    auto post_model =
        std::make_shared<ov::Model>(ov::OutputVector{result_node}, ov::ParameterVector{input_param, attention_mask});
    post_model->set_friendly_name(model->get_friendly_name() + "_post_process");
    post_model->validate_nodes_and_infer_types();
    return post_model;
}

void SidepacketServable::initialize(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) {
    auto fsModelsPath = std::filesystem::path(modelDir);
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
    } else {
        parsedModelsPath = fsModelsPath.string();
    }
    std::filesystem::path configPath = (parsedModelsPath / "config.json");
    if (std::filesystem::exists(configPath)) {
        std::ifstream ifs(configPath.string());
        if (ifs.is_open()) {
            rapidjson::Document modelConfig;
            rapidjson::IStreamWrapper isw(ifs);
            rapidjson::ParseResult parseResult = modelConfig.ParseStream(isw);
            if (parseResult.Code()) {
                SPDLOG_ERROR("Parsing config.json failed: {}", rapidjson::GetParseError_En(parseResult.Code()));
            } else {
                std::vector<std::string> maxLengthFields = {"max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"};
                for (auto field : maxLengthFields) {
                    if (modelConfig.HasMember(field.c_str()) && modelConfig[field.c_str()].IsUint()) {
                        maxModelLength = modelConfig[field.c_str()].GetUint();
                        break;
                    }
                }
                SET_TOKEN_ID(pad_token, "pad_token_id");
                SET_TOKEN_ID(eos_token, "eos_token_id");
                SET_TOKEN_ID(bos_token, "bos_token_id");
                if (modelConfig.HasMember("sep_token_id") && modelConfig["sep_token_id"].IsInt64()) {
                    sep_token = modelConfig["sep_token_id"].GetInt64();
                } else {
                    sep_token = eos_token;
                }
            }
        }
    }

    ov::AnyMap properties;
    auto status = JsonParser::parsePluginConfig(pluginConfig, properties);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during embeddings node plugin_config option parsing to JSON: {}", pluginConfig);
    }
    ov::AnyMap tokenizerProperties = {{"add_special_tokens", false}};
    tokenizer = std::make_shared<ov::genai::Tokenizer>(parsedModelsPath, tokenizerProperties);
    std::filesystem::path tokenizerConfigPath = (std::filesystem::path(graphPath) / fsModelsPath / "tokenizer_config.json");
    if (std::filesystem::exists(tokenizerConfigPath)) {
        std::ifstream ifs(tokenizerConfigPath.string());
        if (ifs.is_open()) {
            rapidjson::Document tokenizerConfig;
            rapidjson::IStreamWrapper isw(ifs);
            rapidjson::ParseResult parseResult = tokenizerConfig.ParseStream(isw);
            if (parseResult.Code()) {
                SPDLOG_ERROR("Parsing tokenizer_config.json failed: {}", rapidjson::GetParseError_En(parseResult.Code()));
            } else {
                SET_TOKEN(pad_token);
                SET_TOKEN(eos_token);
                SET_TOKEN(bos_token);
                if (!sep_token.has_value()) {
                    if (tokenizerConfig.HasMember("sep_token") && tokenizerConfig["sep_token"].IsString()) {
                        auto tokenizedInputs = tokenizer->encode(tokenizerConfig["sep_token"].GetString());
                        if (tokenizedInputs.input_ids.get_size() == 1 && tokenizedInputs.input_ids.get_element_type() == ov::element::i64)
                            sep_token = reinterpret_cast<int64_t*>(tokenizedInputs.input_ids.data())[0];
                        else
                            SPDLOG_DEBUG("Parsing sep token from tokenizer_config.json failed");
                    } else if (eos_token.has_value()) {
                        sep_token = eos_token;
                    }
                }
            }
        }
    }

    auto& ovmsConfig = ovms::Config::instance();
    uint32_t numberOfParallelInferRequests = 1;
    if (ovmsConfig.nireq() > 0) {
        // nireq is set globally for all models in ovms startup parameters
        numberOfParallelInferRequests = ovmsConfig.nireq();
    }

    ov::Core core;
    std::shared_ptr<ov::Model> m_model = core.read_model(parsedModelsPath / std::filesystem::path("openvino_model.xml"), {}, properties);
    if (targetDevice == "NPU") {
        // TODO: if (config.batch_size.has_value() && is_seq_len_fixed) {
            // utils::reshape_model(model, config, max_position_embeddings);
        // }
        if (m_model->is_dynamic()) {
            // TODO: Setup proper config based on calculator options
            TextEmbeddingPipeline::Config config;
            config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::LAST_TOKEN;
            config.normalize = true;
            // Compile additional CPU model for NPU dynamic model case
            auto post_model = create_post_model(m_model, config);
            postProcCompiledModel = core.compile_model(post_model, "CPU", properties);
            try {
                numberOfParallelInferRequests = postProcCompiledModel.get_property(ov::optimal_number_of_infer_requests);
            } catch (const ov::Exception& ex) {
                SPDLOG_WARN("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS with error {}. Using 1 nireq.", ex.what());
                numberOfParallelInferRequests = 1u;
            }
            postProcInferRequestsQueue = std::make_unique<OVInferRequestsQueue>(postProcCompiledModel, numberOfParallelInferRequests);
            npuPostprocessingRequired = true;

            // Set additional properties for NPU model
            auto kv_pos = get_kv_axes_pos(m_model);
            KVDesc kv_desc;
            get_npu_text_embedding_config(properties, kv_pos, kv_desc, config); 
        }
    } else {
        m_model = this->applyPrePostProcessing(m_model);
    }

    compiledModel = core.compile_model(m_model, targetDevice, properties);

    try {
        numberOfParallelInferRequests = compiledModel.get_property(ov::optimal_number_of_infer_requests);
    } catch (const ov::Exception& ex) {
        SPDLOG_WARN("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS with error {}. Using 1 nireq.", ex.what());
        numberOfParallelInferRequests = 1u;
    }
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(compiledModel, numberOfParallelInferRequests);
}

}  // namespace ovms
