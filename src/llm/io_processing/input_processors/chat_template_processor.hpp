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

#include <string>

#include <openvino/genai/tokenizer.hpp>

#include "../base_input_processor.hpp"
#include "src/llm/runtime_chat_template.hpp"

namespace ovms {

// Applies the chat template to ChatHistory, producing req.promptText.
// Active when: input is ChatHistory variant (CHAT_COMPLETIONS and RESPONSES).
//
// The processor decides the path based on configuration/resources passed in constructor:
// - useMinja=true forces tokenizer.apply_chat_template().
// - useMinja=false uses the prepared runtime Jinja path.
class ChatTemplateProcessor : public BaseInputProcessor {
public:
    ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer,
        bool useMinja,
        const PreparedRuntimeChatTemplate* preparedRuntimeChatTemplate);

    absl::Status process(InputRequest& req) override;

    // Serialises chatHistory to {"messages":[...], "tools":[...], "chat_template_kwargs":{...}}
    // for the runtime Python/Jinja template engine.
    //
    // Public for unit tests: this is the OVMS-owned JSON shape passed to the
    // runtime chat-template path, so the tests can lock it down independently
    // of the Python runtime being loaded.
    static std::string serializeForJinja(const ov::genai::ChatHistory& chatHistory);

private:
    ov::genai::Tokenizer& tokenizer;  // non-owning; lifetime tied to InputProcessorContext
    bool useMinja = false;
    const PreparedRuntimeChatTemplate* preparedRuntimeChatTemplate = nullptr;

    // add_generation_prompt lives inside chat_template_kwargs; MINJA's apply_chat_template
    // takes it as a dedicated argument, so this extracts it out and drops it from the returned
    // kwargs so it isn't supplied twice.
    static absl::Status extractAddGenerationPrompt(const ov::genai::ChatHistory& chatHistory,
        ov::genai::JsonContainer& kwargs, bool& addGenerationPrompt);
};

}  // namespace ovms
