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

// Partial parsing of the chunk to JSON.
// This function is used to handle the case where the chunk is not a complete JSON object,
// but we still want to extract content from it.
// It modifies the chunk to be a valid JSON by adding closures and dropping incomplete elements.
rapidjson::Document partialParseToJson(const std::string& input) {
    bool insideString = false;
    bool insideArray = false;
    bool insideObject = false;
    bool awaitingValue = false;  // Indicates if we are waiting for a value after a key
    bool processingValue = false;
    bool processingKey = false;
    bool recentlyFinishedKey = false;
    size_t lastSeparatorPos = std::string::npos;
    std::vector<char> openCloseStack;
    std::string closedInput = input;  // Start with the original input
    for (auto it = closedInput.begin(); it != closedInput.end(); ++it) {
        char c = *it;
        if (!insideString) {
            if (awaitingValue) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    processingValue = true;
                    awaitingValue = false;
                }
            }
            // JSON openings and closures can be counted only when we are not inside a string
            if (c == '{') {
                openCloseStack.push_back(c);
                insideObject = true;
                insideArray = false;
            } else if (c == '[') {
                openCloseStack.push_back(c);
                insideArray = true;
                insideObject = false;
            } else if (c == '}') {
                if (!openCloseStack.empty() && openCloseStack.back() == '{') {
                    openCloseStack.pop_back();
                    if (!openCloseStack.empty() && openCloseStack.back() == '[') {
                        insideArray = true;  // We are still inside an array
                    } else {
                        insideObject = false;  // We are exiting an object
                    }
                }
            } else if (c == ']') {
                if (!openCloseStack.empty() && openCloseStack.back() == '[') {
                    openCloseStack.pop_back();
                    insideArray = false;  // We are exiting an array
                    if (!openCloseStack.empty() && openCloseStack.back() == '{') {
                        insideObject = true;  // We are still inside an object
                    } else if (!openCloseStack.empty() && openCloseStack.back() == '[') {
                        insideArray = true;  // We are still inside an array
                    } else {
                        processingValue = false;  // We are not processing a value anymore
                    }
                }
            } else if (c == ':') {
                // Encountering a colon outside of a string indicates a key-value pair
                awaitingValue = true;
                recentlyFinishedKey = false;  // We are now awaiting a value for the key
            } else if (c == ',') {
                // Store the position of the last comma
                lastSeparatorPos = std::distance(closedInput.begin(), it);
                if (insideObject) {
                    // If we are inside an object, comma indicates the end of a key-value pair
                    processingValue = false;
                    processingKey = true;  // Next part should be a key
                }
            } else if (c == '"') {
                // If we encounter a quote outside of a string, we set insideString to true
                insideString = true;
                if (processingValue) {
                    // We start a string inside value part
                    processingKey = false;
                } else {
                    // If we are not processing a value, we are starting a new key
                    processingKey = true;
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
                    insideString = false;
                    if (processingValue && !insideArray) {
                        // If we were processing a string that was not in the array, we are done with it
                        processingValue = false;
                    } else if (processingKey) {
                        // If we were processing a key, we are done with it
                        processingKey = false;
                        recentlyFinishedKey = true;  // We just finished processing a key
                    }
                }
            }
        }
    }

    if (processingValue && insideString) {
        // The partial JSON ends with a string value that is not closed
        // We can close it by adding a closing quote
        openCloseStack.push_back('"');
    } else if ((processingValue && insideArray) || processingKey || recentlyFinishedKey || awaitingValue) {
        /*  
            Handling cases when:
            - partial JSON ends with incomplete array
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
