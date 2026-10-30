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
#include "tomato_metric.hpp"

#include <stdexcept>
#include <utility>

#include "../metrics/calculator_metrics.hpp"

namespace ovms {

TomatoMetric::TomatoMetric(std::shared_ptr<CalculatorCounter> counter) :
    counter(std::move(counter)) {
    if (!this->counter) {
        throw std::invalid_argument("Tomato metric requires a counter");
    }
}

void TomatoMetric::consume(std::string_view text, bool finish) {
    constexpr std::string_view word = "tomato";
    size_t occurrences = 0;
    const auto finishWord = [&]() {
        if (matches && matched == word.size()) {
            ++occurrences;
        }
        matched = 0;
        matches = true;
    };
    for (unsigned char c : text) {
        const bool wordCharacter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                   (c >= '0' && c <= '9') || c == '_' || c >= 128;
        if (!wordCharacter) {
            finishWord();
        } else if (matches) {
            const unsigned char lower = (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
            matches = matched < word.size() && lower == static_cast<unsigned char>(word[matched]);
            if (matches) {
                ++matched;
            }
        }
    }
    if (finish) {
        finishWord();
    }
    if (occurrences != 0) {
        counter->increment(static_cast<double>(occurrences));
    }
}

}  // namespace ovms
