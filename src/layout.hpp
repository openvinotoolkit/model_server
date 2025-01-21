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
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <openvino/openvino.hpp>

#include "shape.hpp"

namespace ovms {

class Status;

class Layout : public std::string {
    std::optional<size_t> batchIndex = std::nullopt;

    std::optional<size_t> retrieveBatchIndex() const;
    bool containsEtc() const;

    std::string::size_type getNumberOfKnownDimensions() const;

public:
    static const std::string ALLOWED_DIMENSION_LETTERS;
    static const char ETC_CHAR;
    static const char UNDEFINED_DIMENSION_CHAR;
    static const std::string ALLOWED_DIMENSION_LETTERS_AND_CHARS;
    static const std::string ETC_LAYOUT_DELIMETER;
    static const std::string BATCH_DIMENSION_LETTER;

    Layout() = default;
    Layout(const std::string& str);

    static Layout fromOvLayout(const ov::Layout& layout);

    const std::optional<size_t>& getBatchIndex() const;
    Status validate() const;
    std::optional<Layout> createIntersection(const Layout& other, size_t numberOfDimensions) const;
    static const Layout& getDefaultLayout(size_t numOfDimensions);
    static const Layout& getUnspecifiedLayout();

    bool isCompatible(const Shape& shape) const;
};

}  // namespace ovms
