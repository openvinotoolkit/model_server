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

JsonParseOutcome parseJsonWithDepthLimit(
    rapidjson::Document& doc,
    const char* json,
    std::size_t maxDepth,
    int* errorCode,
    std::size_t* errorOffset) {
    rapidjson::Reader reader;
    rapidjson::StringStream ss(json);
    DepthLimitFilter<rapidjson::Document> filter(doc, maxDepth);
    if (!reader.Parse<rapidjson::kParseIterativeFlag>(ss, filter)) {
        if (errorCode != nullptr) {
            *errorCode = static_cast<int>(reader.GetParseErrorCode());
        }
        if (errorOffset != nullptr) {
            *errorOffset = reader.GetErrorOffset();
        }
        if (reader.GetParseErrorCode() == rapidjson::kParseErrorTermination) {
            return JsonParseOutcome::DepthExceeded;
        }
        return JsonParseOutcome::ParseError;
    }
    return JsonParseOutcome::Ok;
}
}  // namespace ovms
