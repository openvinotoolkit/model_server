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
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace ovms {

// Joins any iterable container of strings (or types appendable to std::string)
// with '\n' as the separator. Empty container produces an empty string; there
// is no trailing newline. Works with std::vector<std::string>,
// std::set<std::string>, or any range whose elements support `std::string +=`.
template <typename Container>
std::string joinWithNewlines(const Container& values) {
    std::string joined;
    bool first = true;
    for (const auto& value : values) {
        if (!first) {
            joined += '\n';
        }
        joined += value;
        first = false;
    }
    return joined;
}

// Splits a newline-delimited C string into non-empty tokens. Accepts nullptr
// and empty input (both return an empty vector).
inline std::vector<std::string> splitNewlineDelimited(const char* values) {
    std::vector<std::string> parsed;
    if (values == nullptr || values[0] == '\0') {
        return parsed;
    }

    std::string data(values);
    size_t start = 0;
    while (start <= data.size()) {
        size_t end = data.find('\n', start);
        std::string value = (end == std::string::npos)
                                ? data.substr(start)
                                : data.substr(start, end - start);
        if (!value.empty()) {
            parsed.push_back(std::move(value));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return parsed;
}

}  // namespace ovms
