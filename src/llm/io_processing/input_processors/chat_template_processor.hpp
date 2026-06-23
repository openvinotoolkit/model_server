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

#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif

namespace ovms {

// Applies the chat template to ChatHistory, producing req.promptText.
// Active when: input is ChatHistory variant (CHAT_COMPLETIONS and RESPONSES).
// Python path (PYTHON_DISABLE==0): serialises ChatHistory to JSON and calls PyJinjaTemplateProcessor.
// Native path (PYTHON_DISABLE==1): calls tokenizer.apply_chat_template().
class ChatTemplateProcessor : public BaseInputProcessor {
public:
#if (PYTHON_DISABLE == 0)
    ChatTemplateProcessor(ov::genai::Tokenizer tokenizer,
        PyJinjaTemplateProcessor& templateProcessor,
        const std::string& modelsPath,
        bool useMinja = false);
#else
    explicit ChatTemplateProcessor(ov::genai::Tokenizer tokenizer);
#endif

    absl::Status process(InputRequest& req) override;

private:
    ov::genai::Tokenizer tokenizer;
#if (PYTHON_DISABLE == 0)
    PyJinjaTemplateProcessor& templateProcessor;
    std::string modelsPath;
    bool useMinja;
    // Serialises chatHistory to {"messages":[...], "tools":[...], "chat_template_kwargs":{...}}
    // for the Python Jinja template engine.
    static std::string serializeForPyJinja(const ov::genai::ChatHistory& chatHistory);
#endif
};

}  // namespace ovms
