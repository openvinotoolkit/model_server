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

bool Layout::containsEtc() const {
    return this->find(ETC_LAYOUT_DELIMETER) != std::string::npos;
}

const int DUMB = 1000;

std::vector<std::tuple<char, int, int>> calculateDefinedDimensionsMinMaxPositions(const Layout& l) {
    std::vector<std::tuple<char, int, int>> dimensionPosThis;
    int etcCount = 0;
    for (int i = 0; i < l.size(); ++i) {
        char currentChar = l[i];
        if (ALLOWED_DIMENSION_LETTERS.find(currentChar) != std::string::npos) {
            dimensionPosThis.emplace_back(std::make_tuple(currentChar, i - 3 * etcCount, etcCount > 0 ? DUMB : i));
        }
    }
    return dimensionPosThis;
}

Layout Layout::createIntersection(const Layout& other) const {
    return Layout("N...");  // TODO
    if (!this->validate().ok() ||
        !other.validate().ok()) {
        return Layout("");  // TODO
    }
    std::cout << "this: ";
    std::cout << std::endl;
    auto dimensionPosThis = calculateDefinedDimensionsMinMaxPositions(*this);
    for (auto [c, min, max] : dimensionPosThis) {
        std::cout << "c:" << c << " [" << min << "," << max << "]" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "other: ";
    std::cout << std::endl;
    auto dimensionPosOther = calculateDefinedDimensionsMinMaxPositions(*this);
    for (auto [c, min, max] : dimensionPosOther) {
        std::cout << "c:" << c << " [" << min << "," << max << "]" << std::endl;
    }
    std::cout << std::endl;
    // final
    std::vector<std::tuple<char, int, int>> finalPos;
    // znajdz pierwszy
    auto itThis = dimensionPosThis.begin();
    auto itOther = dimensionPosOther.begin();
    int fmin = -1;
    int fmax = -1;
    // both share first char
    if (std::get<0>(*itThis) == std::get<0>(*itOther)) {
        // both begin at the same position
        fmin = std::max(std::get<1>(*itThis), std::get<1>(*itOther));
        fmax = std::min(std::get<2>(*itThis), std::get<2>(*itOther));
        // to do check if that fmax >= fmin
        if (fmin > fmax) {
            // there is no overlap position for both
            return Layout("");
        }
        char c = std::get<0>(*itThis);
        finalPos.emplace_back(std::make_tuple(c, fmin, fmax));
        ++itThis;
        ++itOther;
    } else {
        // first chars are different
        if (std::get<1>(*itThis) == std::get<1>(*itOther)) {
            fmin = std::get<1>(*itThis);
        }
    }
    return other;
}
}  // namespace ovms
