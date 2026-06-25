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

#include "chat_template_caps.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvino/genai/tokenizer.hpp>
#pragma GCC diagnostic pop

namespace ovms {

// Probes a chat template by dry-running it with synthetic tool_call inputs
// to empirically detect whether the template requires object arguments.
// Updates caps.requiresObjectArguments based on probe results.
// Only performs probing if caps.supportsToolCalls is true.
void probeChatTemplateCaps(ov::genai::Tokenizer& tokenizer, ChatTemplateCaps& caps);

}  // namespace ovms
