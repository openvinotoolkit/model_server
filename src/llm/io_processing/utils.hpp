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

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/writer.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#pragma warning(pop)

namespace ovms {
// Generates random alphanumeric string of length 9 for tool call ID
std::string generateRandomId();

size_t findInStringRespectingSpecialChars(const std::string& str, const std::string& target, size_t startPos);
void writeArgumentOfAnyType(const rapidjson::Value& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer);

// ---- Tool parser helpers shared between attribute/tag style parsers (e.g. qwen3coder, minicpm5) ----

// Trims a single leading and a single trailing '\n' from str (in place).
void trimNewline(std::string& str);

// Returns a human-readable name of the JSON value type (for tracing).
const char* jsonTypeOf(const rapidjson::Value& val);

// Re-serializes a JSON value and stores it back as a JSON string value.
void enforceStringValue(rapidjson::Value& v, rapidjson::Document::AllocatorType& alloc);

// Normalizes Python-style booleans ("True"/"TRUE" -> "true", "False"/"FALSE" -> "false") in place.
void normalizeBooleanString(std::string& value);

// Replaces single-quote string delimiters with double quotes for JSON compatibility.
// Handles nested quoting: apostrophes inside double-quoted strings are preserved.
std::string replaceSingleWithDoubleQuotes(const std::string& input);
}  // namespace ovms
