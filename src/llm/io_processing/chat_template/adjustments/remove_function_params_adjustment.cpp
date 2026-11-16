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
#include "remove_function_params_adjustment.hpp"

#include "adjustment_utils.hpp"

#include <cctype>
#include <utility>

namespace ovms {

RemoveFunctionParamsAdjustment::RemoveFunctionParamsAdjustment(std::vector<std::string> functions) :
    functions(std::move(functions)) {}

void RemoveFunctionParamsAdjustment::apply(std::string& processed) const {
    for (const auto& function : functions) {
        size_t searchFrom = 0;
        while (auto result = adjustment_detail::findFilter(processed, function, searchFrom)) {
            size_t openPosition = result->functionEnd;
            while (openPosition < processed.size() && std::isspace(static_cast<unsigned char>(processed[openPosition]))) {
                ++openPosition;
            }
            searchFrom = result->functionEnd;
            if (openPosition >= processed.size() || processed[openPosition] != '(') {
                continue;
            }
            const size_t closePosition = adjustment_detail::findClosingParenthesis(processed, openPosition);
            if (closePosition != std::string::npos) {
                processed.erase(openPosition, closePosition - openPosition + 1);
            }
        }
    }
}

}  // namespace ovms
