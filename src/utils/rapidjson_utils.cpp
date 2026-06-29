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
#include "rapidjson_utils.hpp"

#include <cstddef>
#include <string>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
#include <rapidjson/reader.h>
#include <rapidjson/stream.h>
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#pragma warning(pop)

namespace ovms {
std::string documentToString(const rapidjson::Document& doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

// Lightweight SAX handler that only tracks nesting depth.
// No DOM allocation — all SAX events are accepted and discarded.
struct DepthOnlyHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DepthOnlyHandler> {
    std::size_t depth{0};
    const std::size_t maxDepth;

    explicit DepthOnlyHandler(std::size_t m) :
        maxDepth(m) {}

    bool StartObject() {
        if (++depth > maxDepth)
            return false;
        return true;
    }
    bool StartArray() {
        if (++depth > maxDepth)
            return false;
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        --depth;
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        --depth;
        return true;
    }
    // All other events (Null, Bool, Int, …) accepted by BaseReaderHandler defaults.
};

Status parseJsonWithDepthLimit(
    rapidjson::Document& doc,
    const char* json,
    std::size_t maxDepth) {
    // Pass 1: depth-only scan — no DOM allocation.
    {
        rapidjson::Reader reader;
        rapidjson::StringStream ss(json);
        DepthOnlyHandler depthHandler(maxDepth);
        if (!reader.Parse<rapidjson::kParseIterativeFlag>(ss, depthHandler)) {
            if (reader.GetParseErrorCode() == rapidjson::kParseErrorTermination) {
                return StatusCode::JSON_NESTING_DEPTH_EXCEEDED;
            }
            std::string details = std::string("Error: ") +
                                  rapidjson::GetParseError_En(reader.GetParseErrorCode()) +
                                  " Offset: " + std::to_string(reader.GetErrorOffset());
            return Status(StatusCode::JSON_INVALID, details);
        }
    }

    // Pass 2: real DOM parse (depth is guaranteed safe).
    doc.Parse<rapidjson::kParseIterativeFlag>(json);
    if (doc.HasParseError()) {
        std::string details = std::string("Error: ") +
                              rapidjson::GetParseError_En(doc.GetParseError()) +
                              " Offset: " + std::to_string(doc.GetErrorOffset());
        return Status(StatusCode::JSON_INVALID, details);
    }
    return StatusCode::OK;
}
}  // namespace ovms
