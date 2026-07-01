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
#pragma once

#include <string>

#include "chat_template_caps.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvino/genai/tokenizer.hpp>
#pragma GCC diagnostic pop

namespace ovms {

// Probes whether a chat template can render a basic user/assistant exchange.
// Returns false if minja silently fails (e.g. unsupported Jinja extensions like {%- generation -%}).
// Should be called before tool-specific probing — if this fails, the template is broken.
bool probeChatTemplateBasicRender(ov::genai::Tokenizer& tokenizer);

// Probes a chat template by dry-running it with synthetic tool_call inputs
// to empirically detect whether the template requires object arguments.
// Updates caps.requiresObjectArguments based on probe results.
// Only performs probing if caps.supportsToolCalls is true.
// Returns false if minja silently failed to render tool calls (template unsupported).
bool probeChatTemplateCapsMinja(ov::genai::Tokenizer& tokenizer, ChatTemplateCaps& caps);

#if (PYTHON_DISABLE == 0)
class PyJinjaTemplateProcessor;

// Probes a chat template via Python Jinja by dry-running it with synthetic tool_call inputs.
// Updates caps.requiresObjectArguments based on probe results.
// Only performs probing if caps.supportsToolCalls is true.
// Returns false if Jinja silently failed to render tool calls (template unsupported).
bool probeChatTemplateCapsJinja(PyJinjaTemplateProcessor& templateProcessor, const std::string& modelsPath, ChatTemplateCaps& caps);
#endif

}  // namespace ovms
