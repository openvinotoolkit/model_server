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

#include <openvino/genai/chat_history.hpp>

#include "../chat_template/caps.hpp"
#include "../base_input_processor.hpp"

namespace ovms {

namespace chat_template_adapter {

// Converts tool_call arguments from string to object.
// Models like Gemma require arguments as a dict/object, not a stringified JSON.
void funcArgsToObjectHistory(ov::genai::ChatHistory& chatHistory);

// Ensures assistant messages with tool_calls have non-null content.
// Some templates require content="" (for example llama) rather than content=null.
void ensureNonNullContentHistory(ov::genai::ChatHistory& chatHistory);

// Apply all relevant adaptations to the ChatHistory based on detected capabilities.
void applyToHistory(const ChatTemplateCaps& caps, ov::genai::ChatHistory& chatHistory);

}  // namespace chat_template_adapter

// Input processor that applies chat template adaptations to ChatHistory.
// Runs before ChatTemplateProcessor so the template receives corrected input.
class ChatTemplateAdapter : public BaseInputProcessor {
public:
    explicit ChatTemplateAdapter(const ChatTemplateCaps& caps);

    absl::Status process(InputRequest& req) override;

private:
    const ChatTemplateCaps& caps;
};

}  // namespace ovms
