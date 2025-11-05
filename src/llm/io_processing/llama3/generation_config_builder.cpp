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

    if (enableToolGuidedGeneration || request.toolChoice == "required") {
        // Set tool guided generation config specific to Llama-3 model
        auto triggeredTags = std::make_shared<ov::genai::StructuredOutputConfig::TriggeredTags>();
        triggeredTags->triggers.push_back("{\"name\":");

        for (const auto& [toolName, toolSchemaWrapper] : request.toolNameSchemaMap) {
            const auto& toolSchema = toolSchemaWrapper.stringRepr;
            ov::genai::StructuredOutputConfig::Tag tagItem;
            tagItem.begin = "{\"name\": \"" + toolName + "\", \"parameters\": ";
            tagItem.end = "}";
            tagItem.content = ov::genai::StructuredOutputConfig::JSONSchema(toolSchema);
            triggeredTags->tags.push_back(tagItem);
        }
        if (request.toolChoice == "required") {
            triggeredTags->at_least_one = true;
        }
        ov::genai::StructuredOutputConfig::StructuralTag structuralTag = triggeredTags;
        setStructuralTagsConfig(structuralTag);
    }
}

}  // namespace ovms
