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

#include <algorithm>
#include <optional>
#include <string>
#include <tuple>

#include "logging.hpp"
#include "status.hpp"

const char* DEFAULT_LAYOUT = "N...";
const char* UNSPECIFIED_LAYOUT = "...";

namespace ovms {
const std::string Layout::ALLOWED_DIMENSION_LETTERS = "NCHWD";
const char Layout::ETC_CHAR = '.';
const char Layout::UNDEFINED_DIMENSION_CHAR = '?';
const std::string Layout::ALLOWED_DIMENSION_LETTERS_AND_CHARS = ALLOWED_DIMENSION_LETTERS + ETC_CHAR + UNDEFINED_DIMENSION_CHAR;
const std::string Layout::ETC_LAYOUT_DELIMETER = "...";
const std::string Layout::BATCH_DIMENSION_LETTER = "N";
const Layout& Layout::getDefaultLayout(size_t numOfDimensions) {
    static const Layout defaultLayout{DEFAULT_LAYOUT};
    return numOfDimensions ? defaultLayout : getUnspecifiedLayout();
}

const Layout& Layout::getUnspecifiedLayout() {
    static const Layout unspecifiedLayout{UNSPECIFIED_LAYOUT};
    return unspecifiedLayout;
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
    if (static_cast<std::string>(*this) == UNSPECIFIED_LAYOUT) {
        return std::nullopt;
    }
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

std::optional<Layout> Layout::createIntersection(const Layout& other, size_t numberOfDimensions) const {
    Layout lhs = (*this);
    Layout rhs = other;

    if (lhs.containsEtc()) {
        size_t knownDimensions = std::count_if(lhs.begin(), lhs.end(), [](char c) { return c != ETC_CHAR; });
        if (knownDimensions > numberOfDimensions)
            return std::nullopt;
        size_t unknownDimensions = numberOfDimensions - knownDimensions;
        lhs.replace(lhs.find(ETC_LAYOUT_DELIMETER), ETC_LAYOUT_DELIMETER.size(), std::string(unknownDimensions, UNDEFINED_DIMENSION_CHAR));
    }

    if (rhs.containsEtc()) {
        size_t knownDimensions = std::count_if(rhs.begin(), rhs.end(), [](char c) { return c != ETC_CHAR; });
        if (knownDimensions > numberOfDimensions)
            return std::nullopt;
        size_t unknownDimensions = numberOfDimensions - knownDimensions;
        rhs.replace(rhs.find(ETC_LAYOUT_DELIMETER), ETC_LAYOUT_DELIMETER.size(), std::string(unknownDimensions, UNDEFINED_DIMENSION_CHAR));
    }

    if (lhs.size() != rhs.size() || lhs.size() != numberOfDimensions)
        return std::nullopt;

    for (size_t i = 0; i < lhs.size(); i++) {
        if (lhs[i] == rhs[i])
            continue;
        if (rhs[i] != UNDEFINED_DIMENSION_CHAR && lhs.find(rhs[i]) != std::string::npos)
            return std::nullopt;
        if (lhs[i] == UNDEFINED_DIMENSION_CHAR) {
            lhs[i] = rhs[i];
            continue;
        }
        if (rhs[i] == UNDEFINED_DIMENSION_CHAR)
            continue;
        return std::nullopt;
    }

    return lhs;
}

Layout Layout::fromOvLayout(const ov::Layout& layout) {
    std::string strCopy = layout.to_string();
    strCopy.erase(std::remove_if(strCopy.begin(), strCopy.end(),
                      [](char c) -> bool {
                          return c == '[' || c == ']' || c == ',';
                      }),
        strCopy.end());
    return Layout(strCopy);
}

bool Layout::containsEtc() const {
    return this->find(ETC_LAYOUT_DELIMETER) != std::string::npos;
}

std::string::size_type Layout::getNumberOfKnownDimensions() const {
    return std::count_if(this->begin(), this->end(), [](char c) { return ALLOWED_DIMENSION_LETTERS.find(c) != std::string::npos || c == UNDEFINED_DIMENSION_CHAR; });
}

bool Layout::isCompatible(const Shape& shape) const {
    if (this->containsEtc()) {
        return this->getNumberOfKnownDimensions() <= shape.size();
    }
    return this->getNumberOfKnownDimensions() == shape.size();
}

}  // namespace ovms
