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

#include "generation_config_builder.hpp"

#include <memory>

#include "src/port/rapidjson_document.hpp"

namespace ovms {

void Qwen3CoderGenerationConfigBuilder::applyToolGuidance(const OpenAIRequest& request, const ModelGenerationContext& generationContext) {
    using SOC = ov::genai::StructuredOutputConfig;
    auto tags = std::make_shared<SOC::TriggeredTags>();
    tags->triggers.push_back("<tool_call>");
    rapidjson::Document parameterlessSchema;
    parameterlessSchema.Parse(R"({"type":"object","properties":{},"additionalProperties":false})");
    for (const auto& [name, schema] : request.toolNameSchemaMap) {
        rapidjson::Document parsedSchema;
        const rapidjson::Value* parameters = schema.rapidjsonRepr;
        if (parameters == nullptr) {
            parsedSchema.Parse(schema.stringRepr.c_str());
            parameters = &parsedSchema;
        }
        // The XML converter emits invalid EBNF for the closed empty-object schema.
        // Match only optional whitespace so an empty parameter block stays valid.
        const SOC::StructuralTag arguments = *parameters == parameterlessSchema
                                                 ? SOC::StructuralTag(SOC::Regex(R"([ \t\r\n]*)"))
                                                 : SOC::StructuralTag(SOC::QwenXMLParametersFormat(schema.stringRepr));
        tags->tags.emplace_back("<tool_call>\n<function=" + name + ">\n",
            arguments, "\n</function>\n</tool_call>");
    }
    setToolGrammar(request, tags, generationContext);
}

}  // namespace ovms
