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
#include "adjuster.hpp"

#include <cctype>
#include <string>

namespace ovms {
namespace {

bool isIdentifierChar(char value) {
    return std::isalnum(static_cast<unsigned char>(value)) || value == '_';
}

bool findFilter(const std::string& source, const std::string& function,
    size_t searchFrom, size_t& pipePosition, size_t& functionEnd) {
    for (size_t pipe = source.find('|', searchFrom); pipe != std::string::npos; pipe = source.find('|', pipe + 1)) {
        size_t functionStart = pipe + 1;
        while (functionStart < source.size() && std::isspace(static_cast<unsigned char>(source[functionStart]))) {
            ++functionStart;
        }
        if (source.compare(functionStart, function.size(), function) != 0) {
            continue;
        }
        functionEnd = functionStart + function.size();
        if (functionEnd < source.size() && isIdentifierChar(source[functionEnd])) {
            continue;
        }
        pipePosition = pipe;
        return true;
    }
    return false;
}

size_t findClosingParenthesis(const std::string& source, size_t openPosition) {
    size_t depth = 0;
    char quote = '\0';
    bool escaped = false;
    for (size_t position = openPosition; position < source.size(); ++position) {
        const char value = source[position];
        if (quote != '\0') {
            if (escaped) {
                escaped = false;
            } else if (value == '\\') {
                escaped = true;
            } else if (value == quote) {
                quote = '\0';
            }
            continue;
        }
        if (value == '\'' || value == '"') {
            quote = value;
        } else if (value == '(') {
            ++depth;
        } else if (value == ')' && --depth == 0) {
            return position;
        }
    }
    return std::string::npos;
}

std::string preprocess(const std::string& templateSource, const ChatTemplateAdjusterConfig& config) {
    std::string processed = templateSource;
    for (const auto& function : config.removeFunctionsParams) {
        size_t searchFrom = 0;
        size_t pipePosition = 0;
        size_t functionEnd = 0;
        while (findFilter(processed, function, searchFrom, pipePosition, functionEnd)) {
            size_t openPosition = functionEnd;
            while (openPosition < processed.size() && std::isspace(static_cast<unsigned char>(processed[openPosition]))) {
                ++openPosition;
            }
            if (openPosition >= processed.size() || processed[openPosition] != '(') {
                searchFrom = functionEnd;
                continue;
            }
            const size_t closePosition = findClosingParenthesis(processed, openPosition);
            if (closePosition == std::string::npos) {
                searchFrom = openPosition + 1;
                continue;
            }
            processed.erase(openPosition, closePosition - openPosition + 1);
            searchFrom = openPosition;
        }
    }

    for (const auto& function : config.removeFunctions) {
        size_t searchFrom = 0;
        size_t pipePosition = 0;
        size_t functionEnd = 0;
        while (findFilter(processed, function, searchFrom, pipePosition, functionEnd)) {
            processed.erase(pipePosition, functionEnd - pipePosition);
            searchFrom = pipePosition;
        }
    }
    return processed;
}

}  // namespace

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forMinja() {
    return {{"tojson"}, {"min", "from_json"}};
}

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forJinja() {
    return {};
}

void ChatTemplateAdjuster::adjust(std::string& templateSource, const ChatTemplateAdjusterConfig& config) {
    templateSource = preprocess(templateSource, config);
}

}  // namespace ovms