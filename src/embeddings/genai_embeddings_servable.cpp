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
#include "genai_embeddings_servable.hpp"

#include <vector>

#include "../logging.hpp"

#include "openvino/core/except.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"

#include "../json_parser.hpp"

using namespace ov::genai;
using namespace ov;

namespace ovms {
void GenaiEmbeddingsServable::initialize(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) {
    auto fsModelsPath = std::filesystem::path(modelDir);
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
    } else {
        parsedModelsPath = fsModelsPath.string();
    }

    ov::AnyMap properties;
    auto status = JsonParser::parsePluginConfig(pluginConfig, properties);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during embeddings node plugin_config option parsing to JSON: {}", pluginConfig);
    }

    TextEmbeddingPipeline::Config config(properties);
    switch (pooling) {
        case mediapipe::EmbeddingsCalculatorOVOptions_Pooling_CLS:
            config.pooling_type = TextEmbeddingPipeline::PoolingType::CLS;
            break;
        case mediapipe::EmbeddingsCalculatorOVOptions_Pooling_LAST:
            config.pooling_type = TextEmbeddingPipeline::PoolingType::LAST_TOKEN;
            break;
        case mediapipe::EmbeddingsCalculatorOVOptions_Pooling_MEAN:
            config.pooling_type = TextEmbeddingPipeline::PoolingType::MEAN;
            break;
        default:
            config.pooling_type = TextEmbeddingPipeline::PoolingType::CLS;
            break;
        }

    m_pipeline = std::make_unique<TextEmbeddingPipeline>(parsedModelsPath, targetDevice, config);
}

}  // namespace ovms
