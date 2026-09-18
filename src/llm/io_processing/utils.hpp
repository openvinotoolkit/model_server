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
#include <optional>
#include <string>
#include <vector>

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

// Trims one wrapping pair of quotes (either " or ') from an XML-style tag attribute value
// (in place). Only a clean wrapping pair is treated as a delimiter: the value is left
// untouched unless the same quote character is both its first and its last character and
// does not occur anywhere in between.
void trimSurroundingQuotes(std::string& str);

// Returns a human-readable name of the JSON value type (for tracing).
const char* jsonTypeOf(const rapidjson::Value& val);

// Re-serializes a JSON value and stores it back as a JSON string value.
void enforceStringValue(rapidjson::Value& v, rapidjson::Document::AllocatorType& alloc);

// Normalizes Python-style booleans ("True"/"TRUE" -> "true", "False"/"FALSE" -> "false") in place.
void normalizeBooleanString(std::string& value);

// Replaces single-quote string delimiters with double quotes for JSON compatibility.
// Handles nested quoting: apostrophes inside double-quoted strings are preserved.
std::string replaceSingleWithDoubleQuotes(const std::string& input);

// Converts a raw string payload into a quoted, properly escaped JSON string value.
// Interprets rawValue as JSON string content when that is valid (respecting existing
// escapes), otherwise takes it literally (e.g. Windows paths are not validly escaped JSON).
std::string escapeAsJsonString(const std::string& rawValue);

// Returns true if value is wrapped on both ends by delimiter (and long enough to hold two).
bool isWrappedByDelimiter(const std::string& value, const std::string& delimiter);

// Masks '"', '\'', '{', '}', '[', ']' found inside delimiter...delimiter pairs (same length,
// the delimiters themselves left intact) so a string value's own payload (e.g. code containing
// commas/braces/quotes) can't be mistaken for structural tokens by findInStringRespectingSpecialChars.
// An unclosed trailing value (still streaming) is masked through the current buffer end too; a later
// call re-masks from scratch once its closing delimiter has arrived.
std::string maskDelimitedStringValues(const std::string& text, const std::string& delimiter);

// Splits content into top-level parts on separator, ignoring separators that are nested inside
// a string value, an object or an array. maskedContent, if provided, must be a same-length version
// of content with in-string special characters masked out (see maskDelimitedStringValues);
// when omitted, content itself is used to look up separators.
std::vector<std::string> splitRespectingSpecialChars(const std::string& content, const std::string& separator, const std::optional<std::string>& maskedContent = std::nullopt);
}  // namespace ovms
