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

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <openvino/genai/generation_config.hpp>

#include "generation_config_builder.hpp"

namespace ovms {

void MistralGenerationConfigBuilder::parseConfigFromRequest(const OpenAIChatCompletionsRequest& request) {
    // Call the base class method to fill in common configuration
    BaseGenerationConfigBuilder::parseConfigFromRequest(request);

    // For now the only specific part is related to tools, so if there are no tools provided in the request
    // we can exit early
    if (request.toolNameSchemaMap.empty()) {
        return;
    }

    std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;

    if (enableToolGuidedGeneration || request.toolChoice == "required") {
        // Set tool guided generation config specific to Mistral model as described in template from:
        // https://github.com/vllm-project/vllm/blob/v0.10.2/examples/tool_chat_template_mistral_parallel.jinja

        static const std::string beginOfToolsString = "[TOOL_CALLS] [";
        auto triggeredTags = std::make_shared<ov::genai::StructuredOutputConfig::TriggeredTags>();
        triggeredTags->triggers.push_back(beginOfToolsString);
        ov::genai::StructuredOutputConfig::Tag tagItem;
        tagItem.begin = beginOfToolsString;

        // Add [TOOL_CALLS] as a stop string to prevent it from being generated multiple times
        addStopString("[TOOL_CALLS]");

        // Build schema for a single tool call object
        std::string toolCallSchema = R"({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [)";
        
        bool first = true;
        for (const auto& [toolName, _] : request.toolNameSchemaMap) {
            if (!first) {
                toolCallSchema += ",";
            }
            first = false;
            toolCallSchema += "\"" + toolName + "\"";
        }
        
        toolCallSchema += R"(]
                },
                "arguments": {
                    "type": "object",
                    "oneOf": [)";
        
        first = true;
        for (const auto& [toolName, toolSchemaWrapper] : request.toolNameSchemaMap) {
            const auto& toolSchema = toolSchemaWrapper.stringRepr;
            if (!first) {
                toolCallSchema += ",";
            }
            first = false;
            toolCallSchema += toolSchema;
        }
        
        toolCallSchema += R"(]
                }
            },
            "required": ["name", "arguments"],
            "additionalProperties": false
        })";

        // Schema for array of tool calls
        std::string schema = R"({
            "type": "array",
            "items": )" + toolCallSchema + R"(,
            "minItems": 1
        })";

        tagItem.content = ov::genai::StructuredOutputConfig::JSONSchema(schema);
        triggeredTags->tags.push_back(tagItem);
        if (request.toolChoice == "required") {
            triggeredTags->at_least_one = true;
        }
        ov::genai::StructuredOutputConfig::StructuralTag structuralTag = triggeredTags;
        setStructuralTagsConfig(structuralTag);
    }
}

}  // namespace ovms
