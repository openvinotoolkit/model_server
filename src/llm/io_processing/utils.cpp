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

    for (size_t i = startPos; i < str.length(); ++i) {
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
        } else if (str[i] == '\'' && (i == 0 || str[i - 1] != '\\')) {
            singleQuoteDepth = 1 - singleQuoteDepth;
        } else if (bracketDepth == 0 && braceDepth == 0 && quoteDepth == 0 && singleQuoteDepth == 0 &&
                   str.compare(i, target.length(), target) == 0) {
            return i;
        }
    }
    return std::string::npos;
}

}  // namespace ovms
