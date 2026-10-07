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
#include <cstring>
#include <string>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
#include <rapidjson/memorystream.h>
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

void addJsonOrStringMember(rapidjson::Value& obj, const char* key, const std::string& value, rapidjson::Document::AllocatorType& alloc) {
    rapidjson::Document parsed(&alloc);
    if (!parsed.Parse(value.c_str()).HasParseError() && parsed.IsObject()) {
        rapidjson::Value jsonValue(parsed, alloc);
        obj.AddMember(rapidjson::Value(key, alloc), jsonValue, alloc);
    } else {
        obj.AddMember(rapidjson::Value(key, alloc), rapidjson::Value(value.c_str(), alloc), alloc);
    }
}

enum class JsonLimitViolation {
    NONE,
    DEPTH,
    COMPLEXITY,
};

// Lightweight SAX handler that tracks nesting depth and total JSON complexity.
// No DOM allocation — all SAX events are accepted and discarded until a limit trips.
struct DepthOnlyHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DepthOnlyHandler> {
    std::size_t depth{0};
    std::size_t complexity{0};
    const std::size_t maxDepth;
    const std::size_t maxComplexity;
    JsonLimitViolation violation{JsonLimitViolation::NONE};

    DepthOnlyHandler(std::size_t maxDepth, std::size_t maxComplexity) :
        maxDepth(maxDepth),
        maxComplexity(maxComplexity) {}

    bool incrementComplexity() {
        if (complexity >= maxComplexity) {
            violation = JsonLimitViolation::COMPLEXITY;
            return false;
        }
        ++complexity;
        return true;
    }

    bool StartObject() {
        if (depth >= maxDepth) {
            violation = JsonLimitViolation::DEPTH;
            return false;
        }
        ++depth;
        if (!incrementComplexity())
            return false;
        return true;
    }
    bool StartArray() {
        if (depth >= maxDepth) {
            violation = JsonLimitViolation::DEPTH;
            return false;
        }
        ++depth;
        if (!incrementComplexity())
            return false;
        return true;
    }
    bool Key(const char*, rapidjson::SizeType, bool) {
        return incrementComplexity();
    }
    bool Null() {
        return incrementComplexity();
    }
    bool Bool(bool) {
        return incrementComplexity();
    }
    bool Int(int) {
        return incrementComplexity();
    }
    bool Uint(unsigned) {
        return incrementComplexity();
    }
    bool Int64(int64_t) {
        return incrementComplexity();
    }
    bool Uint64(uint64_t) {
        return incrementComplexity();
    }
    bool Double(double) {
        return incrementComplexity();
    }
    bool RawNumber(const char*, rapidjson::SizeType, bool) {
        return incrementComplexity();
    }
    bool String(const char*, rapidjson::SizeType, bool) {
        return incrementComplexity();
    }
    bool EndObject(rapidjson::SizeType) {
        if (!incrementComplexity())
            return false;
        --depth;
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        if (!incrementComplexity())
            return false;
        --depth;
        return true;
    }
};

Status parseJsonWithDepthLimit(
    rapidjson::Document& doc,
    const char* json,
    std::size_t maxDepth,
    std::size_t maxComplexity) {
    return parseJsonWithDepthLimit(doc, json, std::strlen(json), maxDepth, maxComplexity);
}

Status parseJsonWithDepthLimit(
    rapidjson::Document& doc,
    const char* json,
    std::size_t jsonLength,
    std::size_t maxDepth,
    std::size_t maxComplexity) {
    // Pass 1: depth-only scan — no DOM allocation.
    {
        rapidjson::Reader reader;
        rapidjson::MemoryStream ss(json, jsonLength);
        DepthOnlyHandler depthHandler(maxDepth, maxComplexity);
        if (!reader.Parse<rapidjson::kParseIterativeFlag>(ss, depthHandler)) {
            if (reader.GetParseErrorCode() == rapidjson::kParseErrorTermination) {
                if (depthHandler.violation == JsonLimitViolation::DEPTH) {
                    return StatusCode::JSON_NESTING_DEPTH_EXCEEDED;
                }
                if (depthHandler.violation == JsonLimitViolation::COMPLEXITY) {
                    return StatusCode::JSON_COMPLEXITY_EXCEEDED;
                }
            }
            std::string details = std::string("Error: ") +
                                  rapidjson::GetParseError_En(reader.GetParseErrorCode()) +
                                  " Offset: " + std::to_string(reader.GetErrorOffset());
            return Status(StatusCode::JSON_INVALID, details);
        }
    }

    // Pass 2: real DOM parse (depth and complexity are guaranteed safe).
    rapidjson::MemoryStream ss(json, jsonLength);
    doc.ParseStream<rapidjson::kParseIterativeFlag>(ss);
    if (doc.HasParseError()) {
        std::string details = std::string("Error: ") +
                              rapidjson::GetParseError_En(doc.GetParseError()) +
                              " Offset: " + std::to_string(doc.GetErrorOffset());
        return Status(StatusCode::JSON_INVALID, details);
    }
    return StatusCode::OK;
}
}  // namespace ovms
