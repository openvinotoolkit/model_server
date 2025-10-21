//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

using namespace rapidjson;
namespace ovms {
class PartialJsonBuilder {
private:
    enum class IteratorState {
        BEGIN,
        AWAITING_KEY,
        PROCESSING_KEY,
        AWAITING_COLON,
        AWAITING_VALUE,
        PROCESSING_NUMBER,
        PROCESSING_KEYWORD,
        PROCESSING_STRING,
        PROCESSING_OBJECT,
        PROCESSING_ARRAY,
        AWAITING_ARRAY_ELEMENT,
        END
    };

    struct LastSeparatorInfo {
        size_t position;
        IteratorState state;
    };

    // Incrementally built JSON string
    std::string buffer = "";
    // Current position in the buffer
    size_t currentPosition = 0;
    // Current state of the iterator
    IteratorState state = IteratorState::BEGIN;
    // Position of the last separator (comma) in the buffer
    LastSeparatorInfo lastSeparator = {0, IteratorState::BEGIN};
    // Open/close stack to track nested structures and open quotes
    std::vector<std::pair<char, size_t>> openCloseStack;

public:
    PartialJsonBuilder() = default;
    // Clear the internal state of the parser
    void clear();
    // Add new chunk to the buffer return current parsed JSON document (incremental parsing)
    Document add(const std::string& chunk);
    // Check if the current state is END (i.e. we have a complete JSON)
    bool isComplete() const;

    // Get the unprocessed part of the buffer (from current position to the end)
    std::string getUnprocessedBuffer() const;

    static Document computeDelta(const Document& previous, const Document& current);
};

}  // namespace ovms
