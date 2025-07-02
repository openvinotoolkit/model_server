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

enum class IteratorState {
    AWAITING_KEY,
    PROCESSING_KEY,
    AWAITING_VALUE,
    PROCESSING_VALUE,
    PROCESSING_STRING,
    PROCESSING_OBJECT,
    PROCESSING_ARRAY,
    AWAITING_ARRAY_ELEMENT,
    END
};

rapidjson::Document partialParseToJson(const std::string& input) {
    IteratorState state = IteratorState::AWAITING_KEY;
    size_t lastSeparatorPos = std::string::npos;
    std::vector<std::pair<char, size_t>> openCloseStack;
    std::string closedInput = input;

    for (auto it = closedInput.begin(); it != closedInput.end(); ++it) {
        char c = *it;
        size_t currentPos = std::distance(closedInput.begin(), it);

        if (state != IteratorState::PROCESSING_STRING && state != IteratorState::PROCESSING_KEY) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                if (state == IteratorState::AWAITING_VALUE) {
                    state = IteratorState::PROCESSING_VALUE;
                } else if (state == IteratorState::AWAITING_ARRAY_ELEMENT) {
                    state = IteratorState::PROCESSING_ARRAY;
                }
            }
            if (c == '{') {
                openCloseStack.emplace_back(c, currentPos);
                state = IteratorState::AWAITING_KEY;
            } else if (c == '[') {
                openCloseStack.emplace_back(c, currentPos);
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
                lastSeparatorPos = currentPos;
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
                }
            }
        } else {
            if (c == '"') {
                if (it != closedInput.begin() && *(it - 1) == '\\') {
                    continue;
                } else {
                    if (state == IteratorState::PROCESSING_KEY) {
                        state = IteratorState::PROCESSING_OBJECT;
                    } else if (state == IteratorState::PROCESSING_STRING) {
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
        doc.Parse(closedInput.c_str());
        if (doc.HasParseError()) {
            throw std::runtime_error("Internal error: Failed to parse partial JSON.");
        }
        return doc;
    }

    if (state == IteratorState::PROCESSING_STRING) {
        openCloseStack.emplace_back('"', closedInput.size());
    } else if (state == IteratorState::AWAITING_KEY || state == IteratorState::PROCESSING_KEY ||
               state == IteratorState::AWAITING_VALUE || state == IteratorState::AWAITING_ARRAY_ELEMENT) {
        if (lastSeparatorPos != std::string::npos && lastSeparatorPos < closedInput.size()) {
            while (!openCloseStack.empty() && openCloseStack.back().second >= lastSeparatorPos) {
                openCloseStack.pop_back();
            }
            closedInput.erase(lastSeparatorPos);
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
    rapidjson::Document doc;
    doc.Parse(closedInput.c_str());
    if (doc.HasParseError()) {
        throw std::runtime_error("Internal error: Failed to parse partial JSON.");
    }
    return doc;
}
