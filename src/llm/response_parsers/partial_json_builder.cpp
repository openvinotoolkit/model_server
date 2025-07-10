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

#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "partial_json_builder.hpp"

using namespace rapidjson;
namespace ovms {

Document computeDeltaImpl(const Value::ConstObject& previous, const Value::ConstObject& current) {
    Document delta;
    delta.SetObject();

    for (const auto& m : current) {
        if (!previous.HasMember(m.name) || previous[m.name].IsNull()) {
            Value copiedValue;
            copiedValue.CopyFrom(m.value, delta.GetAllocator());
            Value key;
            key.CopyFrom(m.name, delta.GetAllocator());
            delta.AddMember(key, copiedValue, delta.GetAllocator());
        } else if (m.value.IsObject() && previous[m.name].IsObject()) {
            Document nestedDelta = computeDeltaImpl(previous[m.name].GetObject(), m.value.GetObject());
            if (!nestedDelta.Empty()) {
                Value nestedDeltaValue;
                nestedDeltaValue.CopyFrom(nestedDelta, delta.GetAllocator());
                Value key;
                key.CopyFrom(m.name, delta.GetAllocator());
                delta.AddMember(key, nestedDeltaValue, delta.GetAllocator());
            }
        } else if (m.value.IsArray() && previous[m.name].IsArray()) {
            const auto& currArray = m.value.GetArray();
            const auto& prevArray = previous[m.name].GetArray();
            if (currArray.Size() > prevArray.Size()) {
                Value diffArray(kArrayType);
                for (SizeType i = prevArray.Size(); i < currArray.Size(); ++i) {
                    Value copiedElement;
                    copiedElement.CopyFrom(currArray[i], delta.GetAllocator());
                    diffArray.PushBack(copiedElement, delta.GetAllocator());
                }
                Value key;
                key.CopyFrom(m.name, delta.GetAllocator());
                delta.AddMember(key, diffArray, delta.GetAllocator());
            }
            // Supporting modifications only for string values
        } else if (previous[m.name] != m.value && (m.value.IsString() && previous[m.name].IsString())) {
            std::string prevStr = previous[m.name].GetString();
            std::string currStr = m.value.GetString();
            if (currStr.size() > prevStr.size()) {
                std::string diffStr = currStr.substr(prevStr.size());
                Value diffValue;
                diffValue.SetString(diffStr.c_str(), diffStr.size(), delta.GetAllocator());
                Value key;
                key.CopyFrom(m.name, delta.GetAllocator());
                delta.AddMember(key, diffValue, delta.GetAllocator());
            }
        }
    }
    return delta;
}

Document PartialJsonBuilder::computeDelta(const Document& previous, const Document& current) {
    return computeDeltaImpl(previous.GetObject(), current.GetObject());
}

void PartialJsonBuilder::clear() {
    buffer.clear();
    currentPosition = 0;
    state = IteratorState::BEGIN;
    lastSeparator = {0, IteratorState::BEGIN};
    openCloseStack.clear();
}

Document PartialJsonBuilder::add(const std::string& chunk) {
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
        Document doc;
        doc.Parse(buffer.c_str());
        if (doc.HasParseError()) {
            throw std::runtime_error("Internal error: Failed to parse partial JSON.");
        }
        return doc;
    }

    // Change is required so we need a copy of the buffer
    std::string closedInput = buffer;

    if (state == IteratorState::AWAITING_VALUE) {
        closedInput += "null";
    } else if (state == IteratorState::AWAITING_KEY || state == IteratorState::PROCESSING_KEY || state == IteratorState::AWAITING_COLON ||
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
    } else if ((state == IteratorState::PROCESSING_STRING && finishedWithEscapeCharacter) ||
               (state == IteratorState::PROCESSING_NUMBER && closedInput.back() == '.')) {
        // If we are processing a string value and we finished with an escape character, we need to remove, so we can close the string properly
        // also if we are processing a number and the last character is a dot, we need to remove it so as we close JSON the entire object is valid
        // (e.g. we are processing a float value and we finished with a dot)
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

    Document doc;
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

}  // namespace ovms
