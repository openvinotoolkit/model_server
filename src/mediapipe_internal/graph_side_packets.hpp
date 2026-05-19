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
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace ovms {

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

// Holds a per-graph LLM execution context that is swapped between requests.
// The mutex synchronizes the handoff between the executor thread (which creates
// a fresh context before each request via set()) and the MediaPipe scheduler
// thread (which reads it in the calculator's Process() via get()).
// In the queue path these run concurrently because the graph stays running.
class GenAiExecutionContextHolder {
public:
    std::shared_ptr<GenAiServableExecutionContext> get() {
        std::lock_guard<std::mutex> lock(executionContextMtx);
        return executionContext;
    }
    void set(std::shared_ptr<GenAiServableExecutionContext> ctx) {
        std::lock_guard<std::mutex> lock(executionContextMtx);
        executionContext = std::move(ctx);
    }
    void reset() {
        std::lock_guard<std::mutex> lock(executionContextMtx);
        executionContext.reset();
    }

private:
    std::mutex executionContextMtx;
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
