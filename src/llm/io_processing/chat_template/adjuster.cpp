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

}  // namespace

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

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forMinja() {
    return {{"min", "from_json"}};
}

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forJinja() {
    return {};
}

ChatTemplateAdjuster::ChatTemplateAdjuster(const ChatTemplateAdjusterConfig& config) {
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