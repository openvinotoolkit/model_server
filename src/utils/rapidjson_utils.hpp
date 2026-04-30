#pragma once
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
#include <cstdint>
#include <string>

#include "src/port/rapidjson_document.hpp"

namespace ovms {
std::string documentToString(const rapidjson::Document& doc);

// Default maximum nesting depth allowed for incoming JSON request bodies.
inline constexpr std::size_t DEFAULT_MAX_JSON_NESTING_DEPTH = 100;

// SAX filter that forwards events to an inner handler while enforcing a
// maximum nesting depth. Returning false from StartObject/StartArray aborts
// parsing with rapidjson::kParseErrorTermination, which avoids materializing
// the entire DOM for pathologically nested payloads.
template <typename Inner>
struct DepthLimitFilter {
    Inner& inner;
    std::size_t depth{0};
    const std::size_t maxDepth;

    DepthLimitFilter(Inner& i, std::size_t m) :
        inner(i),
        maxDepth(m) {}
    DepthLimitFilter(const DepthLimitFilter&) = delete;
    DepthLimitFilter& operator=(const DepthLimitFilter&) = delete;

    bool Null() { return inner.Null(); }
    bool Bool(bool b) { return inner.Bool(b); }
    bool Int(int v) { return inner.Int(v); }
    bool Uint(unsigned v) { return inner.Uint(v); }
    bool Int64(int64_t v) { return inner.Int64(v); }
    bool Uint64(uint64_t v) { return inner.Uint64(v); }
    bool Double(double v) { return inner.Double(v); }
    bool RawNumber(const char* s, rapidjson::SizeType l, bool c) { return inner.RawNumber(s, l, c); }
    bool String(const char* s, rapidjson::SizeType l, bool c) { return inner.String(s, l, c); }
    bool Key(const char* s, rapidjson::SizeType l, bool c) { return inner.Key(s, l, c); }
    bool StartObject() {
        if (++depth > maxDepth)
            return false;
        return inner.StartObject();
    }
    bool StartArray() {
        if (++depth > maxDepth)
            return false;
        return inner.StartArray();
    }
    bool EndObject(rapidjson::SizeType n) {
        --depth;
        return inner.EndObject(n);
    }
    bool EndArray(rapidjson::SizeType n) {
        --depth;
        return inner.EndArray(n);
    }
};

enum class JsonParseOutcome {
    Ok,
    DepthExceeded,
    ParseError,
};

// Populates `doc` from `json` using rapidjson's iterative parser wrapped with a
// DepthLimitFilter. On DepthExceeded / ParseError the rapidjson error details
// (code, offset) are accessible through `errorCode` / `errorOffset` if non-null.
// Iterative parsing avoids native stack recursion regardless of input depth.
JsonParseOutcome parseJsonWithDepthLimit(
    rapidjson::Document& doc,
    const char* json,
    std::size_t maxDepth = DEFAULT_MAX_JSON_NESTING_DEPTH,
    int* errorCode = nullptr,
    std::size_t* errorOffset = nullptr);
}  // namespace ovms
