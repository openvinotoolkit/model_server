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

#include <string>
#include <utility>

#include "status.hpp"

namespace ovms {

static const std::string ALLOWED_DIMENSION_LETTERS = "NCHWD";
static const char ETC_CHAR = '.';
static const char UNDEFINED_DIMENSION_CHAR = '?';
static const std::string ALLOWED_DIMENSION_LETTERS_AND_CHARS = ALLOWED_DIMENSION_LETTERS + ETC_CHAR + UNDEFINED_DIMENSION_CHAR;
static const std::string ETC_LAYOUT_DELIMETER = "...";
static const std::string BATCH_DIMENSION_LETTER = "N";

class Layout : public std::string {
    std::optional<size_t> batchIndex = std::nullopt;

    std::optional<size_t> retrieveBatchIndex() const;
    bool containsEtc() const;

public:
    Layout() = default;
    Layout(const std::string& str);

    const std::optional<size_t>& getBatchIndex() const;
    Status validate() const;
    std::optional<Layout> createIntersection(const Layout& other) const;
    static const Layout& getDefaultLayout();
    static const Layout& getUnspecifiedLayout();
};

}  // namespace ovms
