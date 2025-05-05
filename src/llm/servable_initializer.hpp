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

#include <memory>
#include <string>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#include "src/llm/llm_calculator.pb.h"

namespace ovms {

// Defines what servable type should be initialized based on the pipeline type
enum class PipelineType {
    LM,      // Single modality (text only), text generation based on LLMPipeline
    VLM,     // Multimodal (text and image), text generation based on LLMPipeline
    LM_CB,   // Single modality (text only), text generation based on ContinuousBatchingPipeline
    VLM_CB,  // Multimodal (text and image), text generation based on ContinuousBatchingPipeline

    // Note that *_CB pipelines do not support execution on NPU
};

class Status;
class GenAiServable;
struct GenAiServableProperties;

class GenAiServableInitializer {
public:
    virtual ~GenAiServableInitializer() = default;
#if (PYTHON_DISABLE == 0)
    // Use Python Jinja module for template processing
    static void loadPyTemplateProcessor(std::shared_ptr<GenAiServableProperties> properties, const std::string& chatTemplateDirectory);
#else
    // In C++ only version we use GenAI for template processing, but to have the same behavior as in Python-enabled version
    // we use default template if model does not have its own, so that servable can also work on chat/completion endpoint.
    static void loadDefaultTemplateProcessorIfNeeded(std::shared_ptr<GenAiServableProperties> properties);
#endif
    /*
    initialize method implementation MUST fill servable with all required properties i.e. pipeline, tokenizer, configs etc. based on mediapipe node options.
    It is strictly connected with the servable, so implementation of this method in a derived class should be aware of the specific servable class structure
    and fill both common and servable specific properties required for the servable to implement its interface.
    */
    virtual Status initialize(std::shared_ptr<GenAiServable>& servable, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) = 0;
};
Status parseModelsPath(std::string& outPath, std::string modelsPath, std::string graphPath);
std::optional<uint32_t> parseMaxModelLength(std::string& modelsPath);
Status determinePipelineType(PipelineType& pipelineType, const mediapipe::LLMCalculatorOptions& nodeOptions, const std::string& graphPath);
Status initializeGenAiServable(std::shared_ptr<GenAiServable>& servable, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath);
}  // namespace ovms
