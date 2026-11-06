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
#include <optional>
#include <utility>

namespace ovms {
namespace {

bool isIdentifierChar(char value) {
    return std::isalnum(static_cast<unsigned char>(value)) || value == '_';
}

struct FindFilterResult {
    size_t pipePosition;
    size_t functionEnd;
};

std::optional<FindFilterResult> findFilter(const std::string& source, const std::string& function,
    size_t searchFrom) {
    char quote = '\0';
    bool escaped = false;
    for (size_t pipe = 0; pipe < source.size(); ++pipe) {
        const char value = source[pipe];
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
        if (source.compare(pipe, 2, "{#") == 0) {
            const size_t commentEnd = source.find("#}", pipe + 2);
            if (commentEnd == std::string::npos) {
                return std::nullopt;
            }
            pipe = commentEnd + 1;
            continue;
        }
        if (value == '\'' || value == '"') {
            quote = value;
            continue;
        }
        if (pipe < searchFrom || value != '|') {
            continue;
        }

        size_t functionStart = pipe + 1;
        while (functionStart < source.size() && std::isspace(static_cast<unsigned char>(source[functionStart]))) {
            ++functionStart;
        }
        if (source.compare(functionStart, function.size(), function) != 0) {
            continue;
        }
        size_t functionEnd = functionStart + function.size();
        if (functionEnd < source.size() && isIdentifierChar(source[functionEnd])) {
            continue;
        }

        return FindFilterResult{pipe, functionEnd};
    }
    return std::nullopt;
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
        while (auto result = findFilter(processed, function, searchFrom)) {
            size_t openPosition = result->functionEnd;
            while (openPosition < processed.size() && std::isspace(static_cast<unsigned char>(processed[openPosition]))) {
                ++openPosition;
            }
            searchFrom = result->functionEnd;
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

RemoveFiltersAdjustment::RemoveFiltersAdjustment(std::vector<std::string> filters) :
    filters(std::move(filters)) {}

void RemoveFiltersAdjustment::apply(std::string& processed) const {
    for (const auto& filter : filters) {
        size_t searchFrom = 0;
        while (auto result = findFilter(processed, filter, searchFrom)) {
            size_t removalStart = result->pipePosition;
            while (removalStart > 0 && (processed[removalStart - 1] == ' ' || processed[removalStart - 1] == '\t')) {
                --removalStart;
            }
            processed.erase(removalStart, result->functionEnd - removalStart);
            searchFrom = removalStart;
        }
    }
}

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forMinja() {
    ChatTemplateAdjusterConfig config;
    config.removeFunctions = {"min", "from_json"};
    config.removeFunctionsParams = {"tojson"};
    return config;
}

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forJinja() {
    ChatTemplateAdjusterConfig config;
    return config;
}

ChatTemplateAdjuster::ChatTemplateAdjuster(const ChatTemplateAdjusterConfig& config) {
    if (!config.removeFunctionsParams.empty() || !config.removeFunctions.empty()) {
        std::vector<std::string> removeFunctionsParams = config.removeFunctionsParams;
        removeFunctionsParams.insert(removeFunctionsParams.end(), config.removeFunctions.begin(), config.removeFunctions.end());
        if (!removeFunctionsParams.empty()) {
            adjustments.emplace_back(std::make_unique<RemoveFunctionParamsAdjustment>(removeFunctionsParams));
        }
    }
    if (!config.removeFunctions.empty()) {
        adjustments.emplace_back(std::make_unique<RemoveFiltersAdjustment>(config.removeFunctions));
    }
}

void ChatTemplateAdjuster::adjust(std::string& templateSource) const {
    for (const auto& adjustment : adjustments) {
        adjustment->apply(templateSource);
    }
}

}  // namespace ovms
