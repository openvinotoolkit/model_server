//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "layout.hpp"

#include <tuple>

namespace ovms {
const Layout& Layout::getDefaultLayout() {
    static const Layout defaultLayout{"N..."};
    return defaultLayout;
}

Layout::Layout(const std::string& str) :
    std::string(str) {
    this->batchIndex = retrieveBatchIndex();
}

const std::optional<size_t>& Layout::getBatchIndex() const {
    return this->batchIndex;
}

std::optional<size_t> Layout::retrieveBatchIndex() const {
    auto status = validate();
    if (!status.ok()) {
        return std::nullopt;
    }
    auto batchPos = this->find(BATCH_DIMENSION_LETTER);
    auto etcPos = this->find(ETC_LAYOUT_DELIMETER);
    if (batchPos == std::string::npos) {
        return std::nullopt;
    }
    if (etcPos != std::string::npos && batchPos > etcPos) {
        return std::nullopt;
    }
    return batchPos;
}

Status Layout::validate() const {
    if (this->find_first_not_of(ALLOWED_DIMENSION_LETTERS_AND_CHARS) != std::string::npos)
        return StatusCode::LAYOUT_WRONG_FORMAT;  // Cannot contain other letters

    for (char c : ALLOWED_DIMENSION_LETTERS) {
        if (std::count(this->begin(), this->end(), c) > 1) {
            return StatusCode::LAYOUT_WRONG_FORMAT;  // Can contain NCHWD only single time
        }
    }

    size_t dotCount = 0;
    bool firstEtcAppeared = false;
    bool fullEtcAppeared = false;
    for (char c : (*this)) {
        if (c == ETC_CHAR) {
            if (fullEtcAppeared) {
                return StatusCode::LAYOUT_WRONG_FORMAT;  // Cannot appear multiple times
            }
            firstEtcAppeared = true;
            dotCount++;
            if (dotCount >= 3) {
                fullEtcAppeared = true;
                firstEtcAppeared = false;
            }
        } else if (firstEtcAppeared) {
            return StatusCode::LAYOUT_WRONG_FORMAT;  // Dots separated
        }
    }
    if (firstEtcAppeared && !fullEtcAppeared) {
        return StatusCode::LAYOUT_WRONG_FORMAT;  // Dots not completed
    }

    return StatusCode::OK;
}

std::optional<Layout> Layout::createIntersection(const Layout& other) const {
    if (*this == other ||
        other == Layout::getDefaultLayout())
        return *this;
    if (*this == Layout::getDefaultLayout())
        return other;
    return std::nullopt;
}
}  // namespace ovms
