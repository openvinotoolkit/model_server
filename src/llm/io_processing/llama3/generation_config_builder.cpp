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

#include <string>
#include <utility>
#include <openvino/genai/generation_config.hpp>

#include "generation_config_builder.hpp"

namespace ovms {

void Llama3GenerationConfigBuilder::parseConfigFromRequest(const OpenAIChatCompletionsRequest& request) {
    // Call the base class method to fill in common configuration
    BaseGenerationConfigBuilder::parseConfigFromRequest(request);

    // For now the only specific part is related to tools, so if there are no tools provided in the request
    // we can exit early
    if (request.toolNameSchemaMap.empty()) {
        return;
    }

    // Set tool guided generation config specific to Llama-3 model
    ov::genai::StructuralTagsConfig structuralTagsConfig;
    static const std::string beginOfToolsString = "<|python_tag|>";
    structuralTagsConfig.triggers.push_back(beginOfToolsString);

    for (const auto& [toolName, toolSchemaPair] : request.toolNameSchemaMap) {
        const auto& [_, toolSchema] = toolSchemaPair;
        ov::genai::StructuralTagItem tagItem;
        std::string toolCallTrigger = "{\"name\": \"" + toolName + "\", \"parameters\": ";
        structuralTagsConfig.triggers.push_back(toolCallTrigger);
        tagItem.begin = toolCallTrigger;
        tagItem.schema = toolSchema;
        structuralTagsConfig.structural_tags.push_back(tagItem);
    }
    setStructuralTagsConfig(structuralTagsConfig);
}

}  // namespace ovms
