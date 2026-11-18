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

namespace ovms {

void MistralGenerationConfigBuilder::applyToolGuidance(const OpenAIRequest& request, const ModelGenerationContext&) {
    using SOC = ov::genai::StructuredOutputConfig;
    auto calls = std::make_shared<SOC::TagsWithSeparator>();
    calls->separator = ", ";
    calls->at_least_one = true;
    calls->stop_after_first = requiresNamedToolCall(request);
    for (const auto& [name, schema] : request.toolNameSchemaMap) {
        calls->tags.emplace_back("{\"name\": " + SOC::format_for_json(name) + ", \"arguments\": ",
            buildToolArgumentGrammar(schema.stringRepr), "}");
    }
    auto tags = std::make_shared<SOC::TriggeredTags>();
    tags->triggers.push_back("[TOOL_CALLS]");
    tags->tags.emplace_back("[TOOL_CALLS][", calls, "]");
    setToolGrammar(request, tags);
}

}  // namespace ovms
