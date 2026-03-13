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

#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>

namespace ovms {

// Forward declarations - only shared_ptrs are stored so full definitions are not needed
class PythonNodeResources;
class GenAiServable;
struct GenAiServableExecutionContext;
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

struct GenAiExecutionContextHolder {
    std::mutex mutex;
    std::shared_ptr<GenAiServableExecutionContext> executionContext;
};
using GenAiExecutionContextMap = std::unordered_map<std::string, std::shared_ptr<GenAiExecutionContextHolder>>;

struct GraphSidePackets {
    PythonNodeResourcesMap pythonNodeResourcesMap;
    GenAiServableMap genAiServableMap;
    GenAiExecutionContextMap genAiExecutionContextMap;
    ImageGenerationPipelinesMap imageGenPipelinesMap;
    EmbeddingsServableMap embeddingsServableMap;
    RerankServableMap rerankServableMap;
    SttServableMap sttServableMap;
    TtsServableMap ttsServableMap;
    void clear() {
        pythonNodeResourcesMap.clear();
        genAiServableMap.clear();
        genAiExecutionContextMap.clear();
        imageGenPipelinesMap.clear();
        embeddingsServableMap.clear();
        rerankServableMap.clear();
        sttServableMap.clear();
        ttsServableMap.clear();
    }
    bool empty() {
        return (pythonNodeResourcesMap.empty() &&
                genAiServableMap.empty() &&
                genAiExecutionContextMap.empty() &&
                imageGenPipelinesMap.empty() &&
                embeddingsServableMap.empty() &&
                rerankServableMap.empty() &&
                sttServableMap.empty() &&
                ttsServableMap.empty());
    }
};

}  // namespace ovms
