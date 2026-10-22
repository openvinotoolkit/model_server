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
#include <regex>
#include <string>
#include <utility>

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

}  // namespace

RemoveFunctionParamsAdjustment::RemoveFunctionParamsAdjustment(std::vector<std::string> functions) :
    functions(std::move(functions)) {}

void RemoveFunctionParamsAdjustment::apply(std::string& processed) const {
    for (const auto& function : functions) {
        size_t searchFrom = 0;
        size_t pipePosition = 0;
        size_t functionEnd = 0;
        while (findFilter(processed, function, searchFrom, pipePosition, functionEnd)) {
            size_t openPosition = functionEnd;
            while (openPosition < processed.size() && std::isspace(static_cast<unsigned char>(processed[openPosition]))) {
                ++openPosition;
            }
            searchFrom = functionEnd;
            if (openPosition >= processed.size() || processed[openPosition] != '(') {
                continue;
            }
            const size_t closePosition = findClosingParenthesis(processed, openPosition);
            if (closePosition != std::string::npos) {
                processed.erase(openPosition, closePosition - openPosition + 1);
            }
        }
    }
}

RemoveFiltersAdjustment::RemoveFiltersAdjustment(std::vector<std::string> functions) :
    functions(std::move(functions)) {}

void RemoveFiltersAdjustment::apply(std::string& processed) const {
    for (const auto& function : functions) {
        size_t searchFrom = 0;
        size_t pipePosition = 0;
        size_t functionEnd = 0;
        while (findFilter(processed, function, searchFrom, pipePosition, functionEnd)) {
            size_t removalStart = pipePosition;
            while (removalStart > 0 && (processed[removalStart - 1] == ' ' || processed[removalStart - 1] == '\t')) {
                --removalStart;
            }
            processed.erase(removalStart, functionEnd - removalStart);
            searchFrom = removalStart;
        }
    }
}

ConcatMultiLineStringAdjustment::ConcatMultiLineStringAdjustment() {}

void ConcatMultiLineStringAdjustment::apply(std::string& processed) const {
    static const std::regex stringLiteral(
        R"regex("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')regex");
    std::string concatenated;
    concatenated.reserve(processed.size());
    size_t previousEnd = 0;
    bool previousWasDoubleQuoted = false;
    for (std::sregex_iterator token(processed.begin(), processed.end(), stringLiteral), end; token != end; ++token) {
        const size_t start = static_cast<size_t>(token->position());
        const size_t length = static_cast<size_t>(token->length());
        const bool doubleQuoted = processed[start] == '"';
        const bool whitespaceOnly = processed.find_first_not_of(" \t\r\n\f\v", previousEnd) >= start;
        if (previousWasDoubleQuoted && doubleQuoted && whitespaceOnly) {
            concatenated.pop_back();
            concatenated.append(processed, start + 1, length - 1);
        } else {
            concatenated.append(processed, previousEnd, start - previousEnd);
            concatenated.append(processed, start, length);
        }
        previousEnd = start + length;
        previousWasDoubleQuoted = doubleQuoted;
    }
    concatenated.append(processed, previousEnd, std::string::npos);
    processed = std::move(concatenated);
}

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forMinja() {
    return {{"min", "from_json"}, true, {"tojson"}};
}

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forJinja() {
    return {{}};
}

ChatTemplateAdjuster::ChatTemplateAdjuster(const ChatTemplateAdjusterConfig& config) {
    if (!config.removeFunctionsParams.empty()) {
        adjustments.emplace_back(std::make_unique<RemoveFunctionParamsAdjustment>(config.removeFunctionsParams));
    }
    if (!config.removeFunctions.empty()) {
        adjustments.emplace_back(std::make_unique<RemoveFiltersAdjustment>(config.removeFunctions));
    }
    if (config.concatMultiLineStrings) {
        adjustments.emplace_back(std::make_unique<ConcatMultiLineStringAdjustment>());
    }
}

void ChatTemplateAdjuster::adjust(std::string& templateSource) const {
    for (const auto& adjustment : adjustments) {
        adjustment->apply(templateSource);
    }
}

}  // namespace ovms
