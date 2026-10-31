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

#include <cstddef>
#include <memory>
#include <string_view>

namespace ovms {

class CalculatorCounter;

/// Example per-request matcher: ASCII case-insensitive, whole-word "tomato".
/// Non-ASCII bytes, ASCII letters, digits and underscore are word characters.
class TomatoMetric {
    std::shared_ptr<CalculatorCounter> counter;
    size_t matched = 0;
    bool matches = true;

public:
    explicit TomatoMetric(std::shared_ptr<CalculatorCounter> counter);
    /// Consume generated content; finish flushes the last word of a sequence.
    void consume(std::string_view text, bool finish);
};

}  // namespace ovms
