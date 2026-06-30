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

#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif

namespace ovms {

// Applies the chat template to ChatHistory, producing req.promptText.
// Active when: input is ChatHistory variant (CHAT_COMPLETIONS and RESPONSES).
//
// Under PYTHON_DISABLE==0 two constructors select the path:
//   - PyJinja constructor (takes PyJinjaTemplateProcessor&): uses the Python Jinja engine.
//   - Minja  constructor (tokenizer only): calls tokenizer.apply_chat_template().
// Under PYTHON_DISABLE==1 only the native tokenizer.apply_chat_template() path exists.
class ChatTemplateProcessor : public BaseInputProcessor {
public:
#if (PYTHON_DISABLE == 0)
    // PyJinja path: templateProcessor must be valid (guaranteed by non-null reference param).
    ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer,
        PyJinjaTemplateProcessor& templateProcessor);
    // Minja / native-OV path: no PyJinja processor needed.
    explicit ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer);
#else
    explicit ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer);
#endif

    absl::Status process(InputRequest& req) override;

private:
    ov::genai::Tokenizer& tokenizer;  // non-owning; lifetime tied to InputProcessorContext
#if (PYTHON_DISABLE == 0)
    // Present only on the PyJinja path; nullopt → use tokenizer.apply_chat_template().
    std::optional<std::reference_wrapper<PyJinjaTemplateProcessor>> templateProcessor;
    // Serialises chatHistory to {"messages":[...], "tools":[...], "chat_template_kwargs":{...}}
    // for the Python Jinja template engine.
    static std::string serializeForPyJinja(const ov::genai::ChatHistory& chatHistory);
#endif
};

}  // namespace ovms
