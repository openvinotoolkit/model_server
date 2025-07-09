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

class JsonBuilder {
private:
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
    void clear() {
        buffer.clear();
        currentPosition = 0;
        state = IteratorState::BEGIN;
        lastSeparator = {0, IteratorState::BEGIN};
        openCloseStack.clear();
    }

    rapidjson::Document partialParseToJson(const std::string& chunk) {
        bool finishedWithEscapeCharacter = false;

        // Adding chunk to buffer
        buffer += chunk;

        // Process only the new part of the buffer
        auto beginIt = buffer.begin() + currentPosition;
        auto endIt = buffer.end();

        for (auto it = beginIt; it != endIt; ++it, currentPosition++) {
            finishedWithEscapeCharacter = false;
            char c = *it;

            if (state != IteratorState::PROCESSING_STRING && state != IteratorState::PROCESSING_KEY) {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    continue;
                }
                if (state == IteratorState::AWAITING_VALUE || state == IteratorState::AWAITING_ARRAY_ELEMENT || state == IteratorState::PROCESSING_ARRAY) {
                    // We either start a dict value, start a new array or continue processing an array
                    if (c == 't' || c == 'f' || c == 'n') {
                        state = IteratorState::PROCESSING_KEYWORD;
                    } else {
                        state = IteratorState::PROCESSING_NUMBER;
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
                    if (!openCloseStack.empty() && openCloseStack.back().first == '{') {
                        state = IteratorState::AWAITING_KEY;
                    } else if (!openCloseStack.empty() && openCloseStack.back().first == '[') {
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
                            assert(!openCloseStack.empty() && openCloseStack.back().first == '"');
                            openCloseStack.pop_back();
                            if (!openCloseStack.empty() && openCloseStack.back().first == '[') {
                                state = IteratorState::PROCESSING_ARRAY;
                            } else if (!openCloseStack.empty() && openCloseStack.back().first == '{') {
                                state = IteratorState::PROCESSING_OBJECT;
                            }
                        }
                    }
                } else if (c == '\\') {
                    finishedWithEscapeCharacter = true;
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
            state == IteratorState::AWAITING_VALUE || state == IteratorState::AWAITING_ARRAY_ELEMENT || state == IteratorState::PROCESSING_KEYWORD) {
            if (lastSeparator.position != std::string::npos && lastSeparator.position < closedInput.size()) {
                while (!openCloseStack.empty() && openCloseStack.back().second >= lastSeparator.position) {
                    openCloseStack.pop_back();
                }
                closedInput.erase(lastSeparator.position);
                // Reset current position and state to the last separator, so we parse rejected part again with new chunk
                currentPosition = lastSeparator.position;
                state = lastSeparator.state;
            }
        } else if (state == IteratorState::PROCESSING_STRING && finishedWithEscapeCharacter) {
            // If we are processing a string value and we finished with an escape character, we need to remove, so we can close the string properly
            closedInput.erase(closedInput.size() - 1);
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
