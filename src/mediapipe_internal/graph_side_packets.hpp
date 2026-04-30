//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ovms {
class PythonNodeResources;
class GenAiServable;
struct ImageGenerationPipelines;
struct EmbeddingsServable;
struct RerankServable;
struct SttServable;
class TtsServable;

using PythonNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>;
using GenAiServableMap = std::unordered_map<std::string, std::shared_ptr<GenAiServable>>;
using RerankServableMap = std::unordered_map<std::string, std::shared_ptr<RerankServable>>;
using SttServableMap = std::unordered_map<std::string, std::shared_ptr<SttServable>>;
using TtsServableMap = std::unordered_map<std::string, std::shared_ptr<TtsServable>>;
using EmbeddingsServableMap = std::unordered_map<std::string, std::shared_ptr<EmbeddingsServable>>;
using ImageGenerationPipelinesMap = std::unordered_map<std::string, std::shared_ptr<ImageGenerationPipelines>>;

struct GraphSidePackets {
    PythonNodeResourcesMap pythonNodeResourcesMap;
    GenAiServableMap genAiServableMap;
    ImageGenerationPipelinesMap imageGenPipelinesMap;
    EmbeddingsServableMap embeddingsServableMap;
    RerankServableMap rerankServableMap;
    SttServableMap sttServableMap;
    TtsServableMap ttsServableMap;
    std::vector<std::string> loraAliases;
    void clear() {
        pythonNodeResourcesMap.clear();
        genAiServableMap.clear();
        imageGenPipelinesMap.clear();
        embeddingsServableMap.clear();
        rerankServableMap.clear();
        sttServableMap.clear();
        ttsServableMap.clear();
        loraAliases.clear();
    }
    bool empty() {
        return (pythonNodeResourcesMap.empty() &&
                genAiServableMap.empty() &&
                imageGenPipelinesMap.empty() &&
                embeddingsServableMap.empty() &&
                rerankServableMap.empty() &&
                sttServableMap.empty() &&
                ttsServableMap.empty());
    }
};
}  // namespace ovms
