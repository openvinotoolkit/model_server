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

namespace ovms {

enum class IteratorState {
    BEGIN,
    AWAITING_KEY,
    PROCESSING_KEY,
    AWAITING_COLON,
    AWAITING_VALUE,
    PROCESSING_VALUE,
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

class JsonBuilder {
private:
    // Incrementally built JSON string
    std::string buffer = "";
    // String cache for the part that has been rejected in the last call to we can add it to the new chunk
    std::string cache = "";
    // Current position in the buffer
    size_t currentPosition = 0;
    // Current state of the iterator
    IteratorState state = IteratorState::BEGIN;
    // Position of the last separator (comma) in the buffer
    LastSeparatorInfo lastSeparator = {0, IteratorState::BEGIN};
    // Open/close stack to track nested structures and open quotes
    std::vector<std::pair<char, size_t>> openCloseStack;

public:

    void clear() {
        buffer.clear();
        cache.clear();
        currentPosition = 0;
        state = IteratorState::BEGIN;
        lastSeparator = {0, IteratorState::BEGIN};
        openCloseStack.clear();
    }

    rapidjson::Document partialParseToJson(const std::string& chunk) {
        // Adding chunk to buffer
        buffer += (cache + chunk);
        cache.clear();

        // Process only the new part of the buffer
        auto beginIt = buffer.begin() + currentPosition;
        auto endIt = buffer.end();

        for (auto it = beginIt; it != endIt; ++it, currentPosition++) {
            char c = *it;

            if (state != IteratorState::PROCESSING_STRING && state != IteratorState::PROCESSING_KEY) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    if (state == IteratorState::AWAITING_VALUE) {
                        state = IteratorState::PROCESSING_VALUE;
                    } else if (state == IteratorState::AWAITING_ARRAY_ELEMENT) {
                        state = IteratorState::PROCESSING_ARRAY;
                    }
                }
                if (c == '{') {
                    openCloseStack.emplace_back(c, currentPosition);
                    state = IteratorState::AWAITING_KEY;
                } else if (c == '[') {
                    openCloseStack.emplace_back(c, currentPosition);
                    state = IteratorState::PROCESSING_ARRAY;
                } else if (c == '}') {
                    if (!openCloseStack.empty() && openCloseStack.back().first == '{') {
                        openCloseStack.pop_back();
                        if (!openCloseStack.empty()) {
                            if (openCloseStack.back().first == '{') {
                                state = IteratorState::PROCESSING_OBJECT;
                            } else if (openCloseStack.back().first == '[') {
                                state = IteratorState::PROCESSING_ARRAY;
                            }
                        } else {
                            state = IteratorState::END;
                        }
                    }
                } else if (c == ']') {
                    if (!openCloseStack.empty() && openCloseStack.back().first == '[') {
                        openCloseStack.pop_back();
                        if (!openCloseStack.empty()) {
                            if (openCloseStack.back().first == '{') {
                                state = IteratorState::PROCESSING_OBJECT;
                            } else if (openCloseStack.back().first == '[') {
                                state = IteratorState::PROCESSING_ARRAY;
                            }
                        } else {
                            state = IteratorState::END;
                        }
                    }
                } else if (c == ':') {
                    state = IteratorState::AWAITING_VALUE;
                } else if (c == ',') {
                    lastSeparator = {currentPosition, state};
                    if (state == IteratorState::PROCESSING_OBJECT) {
                        state = IteratorState::AWAITING_KEY;
                    } else if (state == IteratorState::PROCESSING_ARRAY) {
                        state = IteratorState::AWAITING_ARRAY_ELEMENT;
                    }
                } else if (c == '"') {
                    if (state == IteratorState::AWAITING_KEY) {
                        state = IteratorState::PROCESSING_KEY;
                    } else {
                        state = IteratorState::PROCESSING_STRING;
                        // We want to close incomplete strings in values
                        openCloseStack.emplace_back('"', currentPosition);
                    }
                }
            } else {
                if (c == '"') {
                    if (it != buffer.begin() && *(it - 1) == '\\') {
                        continue;
                    } else {
                        if (state == IteratorState::PROCESSING_KEY) {
                            // We processed a key, now we expect a colon
                            state = IteratorState::AWAITING_COLON;
                        } else if (state == IteratorState::PROCESSING_STRING) {
                            assert (!openCloseStack.empty() && openCloseStack.back().first == '"');
                            openCloseStack.pop_back();
                            if (!openCloseStack.empty() && openCloseStack.back().first == '[') {
                                state = IteratorState::PROCESSING_ARRAY;
                            } else if (!openCloseStack.empty() && openCloseStack.back().first == '{') {
                                state = IteratorState::PROCESSING_OBJECT;
                            }
                        }
                    }
                }
            }
        }

        if (state == IteratorState::END && openCloseStack.empty()) {
            rapidjson::Document doc;
            doc.Parse(buffer.c_str());
            if (doc.HasParseError()) {
                throw std::runtime_error("Internal error: Failed to parse partial JSON.");
            }
            return doc;
        }

        // Change is required so we need a copy of the buffer
        std::string closedInput = buffer;

        if (state == IteratorState::AWAITING_KEY || state == IteratorState::PROCESSING_KEY || state == IteratorState::AWAITING_COLON ||
                state == IteratorState::AWAITING_VALUE || state == IteratorState::AWAITING_ARRAY_ELEMENT) {
            if (lastSeparator.position != std::string::npos && lastSeparator.position < closedInput.size()) {
                while (!openCloseStack.empty() && openCloseStack.back().second >= lastSeparator.position) {
                    openCloseStack.pop_back();
                }
                // Store the rejected part in cache before erasing
                //cache = closedInput.substr(lastSeparator.position);
                //std::cout << "Rejected part: " << cache << std::endl;
                std::cout << "Removing rejected part: " << closedInput.substr(lastSeparator.position) << std::endl;
                closedInput.erase(lastSeparator.position);
                // Reset current position and state to the last separator, so we parse rejected part again with new chunk
                currentPosition = lastSeparator.position;
                state = lastSeparator.state;
            }
        }

        for (auto it = openCloseStack.rbegin(); it != openCloseStack.rend(); ++it) {
            if (it->first == '{') {
                closedInput += '}';
            } else if (it->first == '[') {
                closedInput += ']';
            } else if (it->first == '"') {
                closedInput += '"';
            }
        }
        std::cout << "Iteration state: " << static_cast<int>(state) << std::endl;
        std::cout << "Partial JSON closed: " << closedInput << std::endl;
        rapidjson::Document doc;
        if (closedInput.empty()) {
            doc.SetObject();
            return doc;
        }
        doc.Parse(closedInput.c_str());
        if (doc.HasParseError()) {
            throw std::runtime_error("Internal error: Failed to parse partial JSON.");
        }
        return doc;
    }
};

}  // namespace ovms