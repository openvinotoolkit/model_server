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
#pragma once

#include "../filesystem.hpp"
#include "../sidepacket_servable.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "src/embeddings/embeddings_calculator_ov.pb.h"
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace ovms {

struct GenaiEmbeddingsServable : public SidepacketServable {
public:
    GenaiEmbeddingsServable(
        const std::string& modelDir,
        const std::string& targetDevice,
        const std::string& pluginConfig,
        const std::string& graphPath,
        mediapipe::EmbeddingsCalculatorOVOptions_Pooling pooling,
        bool normalizeEmbeddings) :
        SidepacketServable(modelDir, targetDevice, pluginConfig, graphPath),
        pooling(pooling),
        normalizeEmbeddings(normalizeEmbeddings) {}

    void initialize(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath);
    int getTargetOutputIndex() const {
        return targetOutputIndex;
    }

    std::unique_ptr<ov::genai::TextEmbeddingPipeline> m_pipeline;

protected:
    // std::shared_ptr<ov::Model> applyPrePostProcessing(std::shared_ptr<ov::Model> model) override;

private:
    mediapipe::EmbeddingsCalculatorOVOptions_Pooling pooling;
    bool normalizeEmbeddings;

    int targetOutputIndex = -1;
};

using GenaiEmbeddingsServableMap = std::unordered_map<std::string, std::shared_ptr<GenaiEmbeddingsServable>>;
}  // namespace ovms
