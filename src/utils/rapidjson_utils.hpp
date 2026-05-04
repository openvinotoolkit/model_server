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
#include <cstddef>
#include <string>

#include "src/port/rapidjson_document.hpp"
#include "src/status.hpp"

namespace ovms {
std::string documentToString(const rapidjson::Document& doc);

// Default maximum nesting depth allowed for incoming JSON request bodies.
inline constexpr std::size_t DEFAULT_MAX_JSON_NESTING_DEPTH = 100;

// Parses `json` into `doc` with iterative parsing and a nesting depth limit.
// Pass 1: lightweight depth-only scan (no DOM allocation) — aborts early on
//   depth > maxDepth (JSON_NESTING_DEPTH_EXCEEDED) or malformed JSON (JSON_INVALID).
// Pass 2: normal Document::Parse to build the DOM (depth is guaranteed safe).
// Returns StatusCode::OK on success.
Status parseJsonWithDepthLimit(
    rapidjson::Document& doc,
    const char* json,
    std::size_t maxDepth = DEFAULT_MAX_JSON_NESTING_DEPTH);
}  // namespace ovms
