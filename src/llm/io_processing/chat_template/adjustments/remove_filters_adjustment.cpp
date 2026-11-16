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
#include "remove_filters_adjustment.hpp"

#include "adjustment_utils.hpp"

#include <utility>

namespace ovms {

RemoveFiltersAdjustment::RemoveFiltersAdjustment(std::vector<std::string> filters) :
    filters(std::move(filters)) {}

void RemoveFiltersAdjustment::apply(std::string& processed) const {
    for (const auto& filter : filters) {
        size_t searchFrom = 0;
        while (auto result = adjustment_detail::findFilter(processed, filter, searchFrom)) {
            size_t removalStart = result->pipePosition;
            while (removalStart > 0 && (processed[removalStart - 1] == ' ' || processed[removalStart - 1] == '\t')) {
                --removalStart;
            }
            processed.erase(removalStart, result->functionEnd - removalStart);
            searchFrom = removalStart;
        }
    }
}

}  // namespace ovms