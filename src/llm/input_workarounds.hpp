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

#include <openvino/genai/tokenizer.hpp>
#include <rapidjson/document.h>

#include "chat_template_caps.hpp"

namespace ovms {
namespace input_workarounds {

// --- Individual workaround functions (JSON path) ---
// Each operates on the full request document containing "messages" array.
// Exposed individually for unit testing and for selective use during refactoring.

// Convert tool_call arguments from JSON string to parsed JSON object.
// Models like Gemma require arguments as a dict/object, not a stringified JSON.
void funcArgsToObjectJson(rapidjson::Document& doc);

// Ensure assistant messages with tool_calls have non-null content field.
// Some templates require content="" rather than content=null.
void ensureNonNullContentJson(rapidjson::Document& doc);

// --- Individual workaround functions (ChatHistory path) ---
// Operates on ov::genai::ChatHistory for the GenAI C++ tokenizer path.

// Convert tool_call arguments from string to object in ChatHistory.
void funcArgsToObjectHistory(ov::genai::ChatHistory& chatHistory);

// Ensure assistant messages with tool_calls have non-null content in ChatHistory.
void ensureNonNullContentHistory(ov::genai::ChatHistory& chatHistory);

// --- Aggregate application ---

// Apply all relevant workarounds to the JSON document (Python Jinja path).
// Modifies the document in-place based on detected capabilities.
void applyToJson(const ChatTemplateCaps& caps, const std::string& modelFamily, rapidjson::Document& doc);

// Apply all relevant workarounds to the ChatHistory (GenAI C++ tokenizer path).
// Modifies the chat history in-place based on detected capabilities.
void applyToHistory(const ChatTemplateCaps& caps, const std::string& modelFamily, ov::genai::ChatHistory& chatHistory);

}  // namespace input_workarounds
}  // namespace ovms
