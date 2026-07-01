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

#include <openvino/genai/chat_history.hpp>

#include "chat_template_caps.hpp"

namespace ovms {
namespace input_workarounds {

// --- Individual workaround functions (ChatHistory path) ---
// Operates on ov::genai::ChatHistory for both GenAI C++ tokenizer and PyJinja paths.

// Convert tool_call arguments from string to object in ChatHistory.
// Models like Gemma require arguments as a dict/object, not a stringified JSON.
void funcArgsToObjectHistory(ov::genai::ChatHistory& chatHistory);

// Ensure assistant messages with tool_calls have non-null content in ChatHistory.
// Some templates require content="" rather than content=null.
void ensureNonNullContentHistory(ov::genai::ChatHistory& chatHistory);

// --- Aggregate application ---

// Apply all relevant workarounds to the ChatHistory.
// Modifies the chat history in-place based on detected capabilities.
void applyToHistory(const ChatTemplateCaps& caps, ov::genai::ChatHistory& chatHistory);

}  // namespace input_workarounds
}  // namespace ovms
