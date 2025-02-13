//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#include "src/llm/llm_calculator.pb.h"

namespace ovms {
class Status;
class GenAiServable;
class GenAiServableProperties;

class GenAiServableInitializer {
    std::string basePath;

protected:
    Status parseModelsPath(std::string modelsPath, std::string graphPath);

public:
    virtual ~GenAiServableInitializer() = default;
    std::string getBasePath() const {
        return basePath;
    }
    static void loadTextProcessor(std::shared_ptr<GenAiServableProperties> properties, const std::string& chatTemplateDirectory);
    virtual Status initialize(std::shared_ptr<GenAiServable>& servable, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) = 0;
};

Status initializeGenAiServable(std::shared_ptr<GenAiServable>& servable, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath);
}  // namespace ovms
