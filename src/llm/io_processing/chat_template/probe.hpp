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

#include "caps.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvino/genai/tokenizer.hpp>
#pragma GCC diagnostic pop

namespace ovms {

// Probes whether a chat template can render a basic user/assistant exchange.
// Returns false if minja silently fails.
// Should be called before tool-specific probing — if this fails,
// the template is completely broken even for non-agentic use cases.
bool probeChatTemplateBasicRenderMinja(ov::genai::Tokenizer& tokenizer);

// Probes a chat template by dry-running it with synthetic inputs
// to empirically detect whether the template requires special input workarounds.
// Updates caps parameter based on probe results.
// Returns false if minja silently fails to render (template unsupported).
bool probeChatTemplateCapsMinja(ov::genai::Tokenizer& tokenizer, ChatTemplateCaps& caps);

#if (PYTHON_DISABLE == 0)
class PyJinjaTemplateProcessor;

// The same, but for Jinja.
bool probeChatTemplateCapsJinja(PyJinjaTemplateProcessor& templateProcessor, ChatTemplateCaps& caps);
#endif

}  // namespace ovms
