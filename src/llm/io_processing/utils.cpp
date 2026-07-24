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
#include <random>
#include <string>
#include <cctype>

#include "utils.hpp"

namespace ovms {
std::string generateRandomId() {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    static constexpr int idLength = 9;
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_int_distribution<> dist(0, sizeof(alphanum) - 2);

    std::string id;
    id.reserve(idLength);
    for (int i = 0; i < idLength; ++i) {
        id += alphanum[dist(rng)];
    }
    return id;
}

void writeArgumentOfAnyType(const rapidjson::Value& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer) {
    if (arg.IsString()) {
        writer.String(arg.GetString());
    } else if (arg.IsInt64()) {
        writer.Int64(arg.GetInt64());
    } else if (arg.IsDouble()) {
        writer.Double(arg.GetDouble());
    } else if (arg.IsBool()) {
        writer.Bool(arg.GetBool());
    } else if (arg.IsArray()) {
        writer.StartArray();
        for (auto& elem : arg.GetArray()) {
            writeArgumentOfAnyType(elem, writer);
        }
        writer.EndArray();
    } else if (arg.IsObject()) {
        writer.StartObject();
        for (auto it = arg.MemberBegin(); it != arg.MemberEnd(); ++it) {
            writer.Key(it->name.GetString());
            writeArgumentOfAnyType(it->value, writer);
        }
        writer.EndObject();
    } else {
        writer.String("");
    }
}

size_t findInStringRespectingSpecialChars(const std::string& str, const std::string& target, size_t startPos) {
    int bracketDepth = 0;
    int braceDepth = 0;
    int quoteDepth = 0;
    int singleQuoteDepth = 0;

    auto isWordChar = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
    };

    for (size_t i = startPos; i < str.length(); ++i) {
        if (bracketDepth == 0 && braceDepth == 0 && quoteDepth == 0 && singleQuoteDepth == 0 &&
            str.compare(i, target.length(), target) == 0) {
            return i;
        }

        if (str[i] == '{') {
            braceDepth++;
        } else if (str[i] == '}') {
            braceDepth--;
        } else if (str[i] == '[') {
            bracketDepth++;
        } else if (str[i] == ']') {
            bracketDepth--;
        } else if (str[i] == '"' && (i == 0 || str[i - 1] != '\\')) {
            quoteDepth = 1 - quoteDepth;
        } else if (quoteDepth == 0 && str[i] == '\'' && (i == 0 || str[i - 1] != '\\')) {
            const bool prevIsWord = (i > 0) && isWordChar(str[i - 1]);
            const bool nextIsWord = (i + 1 < str.size()) && isWordChar(str[i + 1]);

            if (singleQuoteDepth == 0) {
                // Opening single quote: ignore apostrophes inside words.
                if (!(prevIsWord && nextIsWord)) {
                    singleQuoteDepth = 1;
                }
            } else {
                // Inside single-quoted text: treat apostrophes in words as plain
                // characters (it's, Johns'). Close only when the following
                // non-space token looks like an argument/list/object delimiter.
                if (prevIsWord && nextIsWord) {
                    continue;
                }

                size_t j = i + 1;
                while (j < str.size() && std::isspace(static_cast<unsigned char>(str[j])) != 0) {
                    ++j;
                }
                if (j == str.size() || str[j] == ',' || str[j] == ':' || str[j] == ']' || str[j] == '}' || str[j] == ')') {
                    singleQuoteDepth = 0;
                }
            }
        }
    }
    return std::string::npos;
}

void trimNewline(std::string& str) {
    if (str.empty()) {
        return;
    }
    if (str.back() == '\n') {
        str.pop_back();
    }
    if (str.empty()) {
        return;
    }
    if (str.front() == '\n') {
        str.erase(str.begin());
    }
}

const char* jsonTypeOf(const rapidjson::Value& val) {
    if (val.IsObject())
        return "object";
    if (val.IsArray())
        return "array";
    if (val.IsString())
        return "string";
    if (val.IsBool())
        return "bool";
    if (val.IsInt())
        return "int";
    if (val.IsUint())
        return "uint";
    if (val.IsInt64())
        return "int64";
    if (val.IsUint64())
        return "uint64";
    if (val.IsDouble())
        return "double";
    if (val.IsNumber())
        return "number";
    if (val.IsNull())
        return "null";
    return "unknown";
}

void enforceStringValue(rapidjson::Value& v, rapidjson::Document::AllocatorType& alloc) {
    if (v.IsString()) {
        return;
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    v.Accept(writer);
    v.SetString(buffer.GetString(), buffer.GetLength(), alloc);
}

void normalizeBooleanString(std::string& value) {
    if (value == "True" || value == "TRUE") {
        value = "true";
    } else if (value == "False" || value == "FALSE") {
        value = "false";
    }
}

std::string replaceSingleWithDoubleQuotes(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    bool insideDoubleQuote = false;
    bool insideSingleQuote = false;
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '\\' && i + 1 < input.size()) {
            result += c;
            result += input[++i];
            continue;
        }
        if (c == '"' && !insideSingleQuote) {
            insideDoubleQuote = !insideDoubleQuote;
            result += c;
        } else if (c == '\'' && !insideDoubleQuote) {
            insideSingleQuote = !insideSingleQuote;
            result += '"';
        } else {
            result += c;
        }
    }
    return result;
}

}  // namespace ovms
