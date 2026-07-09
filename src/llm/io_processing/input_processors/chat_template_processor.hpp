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

#include <functional>
#include <optional>
#include <string>

#include <openvino/genai/tokenizer.hpp>

#include "../base_input_processor.hpp"
#include "../../runtime_chat_template.hpp"

#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif

namespace ovms {

// Applies the chat template to ChatHistory, producing req.promptText.
// Active when: input is ChatHistory variant (CHAT_COMPLETIONS and RESPONSES).
//
// Under PYTHON_DISABLE==0 two constructors select the path:
//   - Jinja constructor (takes prepared runtime template and optional in-process processor)
//     uses runtime Python Jinja first, then optional in-process fallback.
//   - Minja  constructor (tokenizer only): calls tokenizer.apply_chat_template().
// Under PYTHON_DISABLE==1 only the native tokenizer.apply_chat_template() path exists.
class ChatTemplateProcessor : public BaseInputProcessor {
public:
#if (PYTHON_DISABLE == 0)
    // Jinja path: preparedRuntimeChatTemplate may be null/unprepared; templateProcessor is optional fallback.
    ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer,
        const PreparedRuntimeChatTemplate* preparedRuntimeChatTemplate,
        PyJinjaTemplateProcessor* templateProcessor);
    // Minja / native-OV path: no PyJinja processor needed.
    explicit ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer);
#else
    ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer,
        const PreparedRuntimeChatTemplate* preparedRuntimeChatTemplate);
    explicit ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer);
#endif

    absl::Status process(InputRequest& req) override;

private:
    ov::genai::Tokenizer& tokenizer;  // non-owning; lifetime tied to InputProcessorContext
    const PreparedRuntimeChatTemplate* preparedRuntimeChatTemplate = nullptr;
#if (PYTHON_DISABLE == 0)
    // Present only on the PyJinja path; nullopt → use tokenizer.apply_chat_template().
    std::optional<std::reference_wrapper<PyJinjaTemplateProcessor>> templateProcessor;
#endif
    // Serialises chatHistory to {"messages":[...], "tools":[...], "chat_template_kwargs":{...}}
    // for Python Jinja template engines.
    static std::string serializeForJinja(const ov::genai::ChatHistory& chatHistory);
};

}  // namespace ovms
