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

#include <string>
#include <vector>
#include <iostream>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

enum class IteratorState {
    AWAITING_KEY,
    INSIDE_KEY,
    AWAITING_VALUE,
    INSIDE_VALUE,
    INSIDE_STRING,
    INSIDE_OBJECT,
    INSIDE_ARRAY,
    END  // indicates that JSON is completed
};

// Partial parsing of the chunk to JSON.
// This function is used to handle the case where the chunk is not a complete JSON object,
// but we still want to extract content from it.
// It modifies the chunk to be a valid JSON by adding closures and dropping incomplete elements.
rapidjson::Document partialParseToJson(const std::string& input) {
    IteratorState state = IteratorState::AWAITING_KEY;
    size_t lastSeparatorPos = std::string::npos;
    std::vector<char> openCloseStack;
    std::string closedInput = input;  // Start with the original input
    for (auto it = closedInput.begin(); it != closedInput.end(); ++it) {
        char c = *it;
        // Check if we are inside a string (key or value)
        if (state != IteratorState::INSIDE_STRING && state != IteratorState::INSIDE_KEY) {
            // We are not inside a string. We either await key, value or process non-string value
            if (state == IteratorState::AWAITING_VALUE && !std::isspace(static_cast<unsigned char>(c))) {
                // We were awaiting value and now we encounter a non-whitespace character indicating we start processing a value
                state = IteratorState::INSIDE_VALUE;
            }
            // JSON openings and closures can be counted only when we are not inside a string
            if (c == '{') {
                openCloseStack.push_back(c);
                state = IteratorState::INSIDE_OBJECT;
            } else if (c == '[') {
                openCloseStack.push_back(c);
                state = IteratorState::INSIDE_ARRAY;
            } else if (c == '}') {
                if (!openCloseStack.empty() && openCloseStack.back() == '{') {
                    openCloseStack.pop_back();
                    if (!openCloseStack.empty()) {
                        if (openCloseStack.back() == '{') {
                            state = IteratorState::INSIDE_OBJECT;
                        } else if (openCloseStack.back() == '[') {
                            state = IteratorState::INSIDE_ARRAY;
                        }
                    } else {
                        state = IteratorState::END;  // We exited the last object, so we are done
                    }
                }
            } else if (c == ']') {
                if (!openCloseStack.empty() && openCloseStack.back() == '[') {
                    openCloseStack.pop_back();
                    if (!openCloseStack.empty()) {
                        if (openCloseStack.back() == '{') {
                            // We exit array, but we might still be inside another object, so now we are awaiting a key
                            state = IteratorState::INSIDE_OBJECT;
                        } else if (openCloseStack.back() == '[') {
                            // We exit array, but we might still be inside another array, so we keep reading values from it
                            state = IteratorState::INSIDE_ARRAY;
                        }
                    } else {
                        state = IteratorState::END;  // We exited the last object, so we are done
                    }
                }
            } else if (c == ':') {
                // Encountering a colon outside of a string indicates a key-value pair
                state = IteratorState::AWAITING_VALUE;
            } else if (c == ',') {
                if (state == IteratorState::INSIDE_OBJECT) {
                    // If we are inside an object, comma indicates the end of a key-value pair
                    state = IteratorState::AWAITING_KEY;
                    // Store the position of the last comma, so we can get back in case of incomplete key
                    lastSeparatorPos = std::distance(closedInput.begin(), it);
                }
            } else if (c == '"') {
                if (state == IteratorState::AWAITING_KEY) {
                    // If we are awaiting a key and encounter a quote, we are starting a new key
                    state = IteratorState::INSIDE_KEY;
                } else {
                    // If we are not awaiting a key, we are starting a string value
                    state = IteratorState::INSIDE_STRING;
                }
            }
        } else {
            if (c == '"') {
                // Check if the quote is escaped
                if (it != closedInput.begin() && *(it - 1) == '\\') {
                    // If the quote is escaped, we ignore it as it is valid part of the string
                    continue;
                } else {
                    // If the quote is not escaped we are exiting the string
                    if (state == IteratorState::INSIDE_KEY) {
                        state = IteratorState::INSIDE_OBJECT;  // We finished processing a key, now we are inside an object
                    } else if (state == IteratorState::INSIDE_STRING) {
                        // We are exiting a string value so we are either inside an object or an array
                        if (!openCloseStack.empty() && openCloseStack.back() == '[') {
                            state = IteratorState::INSIDE_ARRAY;
                        } else if (!openCloseStack.empty() && openCloseStack.back() == '{') {
                            state = IteratorState::INSIDE_OBJECT;
                        }
                    }
                }
            }
        }
    }

    if (state == IteratorState::INSIDE_STRING) {
        // The partial JSON ends with a string value that is not closed
        // We can close it by adding a closing quote
        openCloseStack.push_back('"');
    } else if (state == IteratorState::AWAITING_KEY || state == IteratorState::INSIDE_KEY || state == IteratorState::AWAITING_VALUE) {
        /*  
            Handling cases when:
            - partial JSON ends during or immediately after key processing
            - partial JSON ends when we are awaiting a value after a key
            In such cases we need to drop the incomplete part, get back to the last separator position and remove 
            all the content after it, including the last comma if it exists.
        */
        if (lastSeparatorPos != std::string::npos && lastSeparatorPos < closedInput.size()) {
            closedInput.erase(lastSeparatorPos);
        }
    }

    // Close any unclosed objects/arrays/strings in reverse order
    for (auto it = openCloseStack.rbegin(); it != openCloseStack.rend(); ++it) {
        if (*it == '{') {
            closedInput += '}';
        } else if (*it == '[') {
            closedInput += ']';
        } else if (*it == '"') {
            closedInput += '"';
        }
    }

    rapidjson::Document doc;
    doc.Parse(closedInput.c_str());
    if (doc.HasParseError()) {
        // Throw an exception to indicate an internal error
        throw std::runtime_error("Internal error: Failed to parse partial JSON.");
    }
    return doc;
}
