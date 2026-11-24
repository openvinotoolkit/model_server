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
#include <vector>

namespace ovms {

void MistralGenerationConfigBuilder::applyToolGuidance(const OpenAIRequest& request, const ModelGenerationContext& generationContext) {
    using SOC = ov::genai::StructuredOutputConfig;
    auto calls = std::make_shared<SOC::TagsWithSeparator>();
    calls->separator = ", ";
    calls->at_least_one = true;
    calls->stop_after_first = requiresNamedToolCall(request);
    for (const auto& [name, schema] : request.toolNameSchemaMap) {
        // Allow optional header whitespace: requiring a space after "name": caused repeated <unk>
        // tokens in a Mistral probe; the relaxed header completed. The underlying cause is unconfirmed.
        auto content = std::make_shared<SOC::Concat>(std::vector<SOC::StructuralTag>{
            SOC::Regex(R"([ \t\r\n]*"name"[ \t\r\n]*:[ \t\r\n]*)"),
            SOC::ConstString(SOC::format_for_json(name)),
            SOC::Regex(R"([ \t\r\n]*,[ \t\r\n]*"arguments"[ \t\r\n]*:[ \t\r\n]*)"),
            buildToolArgumentGrammar(schema.stringRepr)});
        calls->tags.emplace_back("{", content, "}");
    }
    auto tags = std::make_shared<SOC::TriggeredTags>();
    tags->triggers.push_back("[TOOL_CALLS]");
    tags->tags.emplace_back("[TOOL_CALLS][", calls, "]");
    setToolGrammar(request, tags, generationContext);
}

}  // namespace ovms
