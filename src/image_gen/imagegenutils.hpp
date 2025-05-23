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
#include <string>
#include <memory>
#include <variant>
#include <utility>

#include <openvino/openvino.hpp>
#include "absl/status/status.h"

#define SET_OR_RETURN(TYPE, NAME, RHS)                      \
    auto NAME##_OPT = RHS;                                  \
    if (std::holds_alternative<absl::Status>(NAME##_OPT)) { \
        return std::get<absl::Status>(NAME##_OPT);          \
    }                                                       \
    TYPE NAME = std::get<TYPE>(NAME##_OPT);

namespace ovms {
class HttpPayload;
using dims_t = std::pair<int64_t, int64_t>;
std::variant<absl::Status, std::optional<dims_t>> getDimensions(const HttpPayload& payload);

std::variant<absl::Status, std::string> getPromptField(const HttpPayload& payload);

std::variant<absl::Status, std::optional<std::string>> getStringFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<int64_t>> getInt64FromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<int>> getIntFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<size_t>> getSizetFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<float>> getFloatFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);

std::variant<absl::Status, ov::AnyMap> getImageGenerationRequestOptions(const HttpPayload& payload);
std::variant<absl::Status, ov::AnyMap> getImageVariationRequestOptions(const HttpPayload& payload);
std::variant<absl::Status, ov::AnyMap> getImageEditRequestOptions(const HttpPayload& payload);

std::unique_ptr<std::string> generateJSONResponseFromB64Image(const std::string& base64_image);
}  // namespace ovms

