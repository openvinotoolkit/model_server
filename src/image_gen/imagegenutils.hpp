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
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/status/status.h"
#pragma warning(pop)

#include "imagegenpipelineargs.hpp"

#define SET_OR_RETURN(TYPE, NAME, RHS)                                                                                          \
    auto NAME##_OPT = RHS;                                                                                                      \
    if (std::holds_alternative<absl::Status>(NAME##_OPT)) {                                                                     \
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get {}: {}", #NAME, std::get<absl::Status>(NAME##_OPT).ToString()); \
        return std::get<absl::Status>(NAME##_OPT);                                                                              \
    }                                                                                                                           \
    auto NAME = std::get<TYPE>(NAME##_OPT);

namespace ovms {
struct HttpPayload;
std::variant<absl::Status, std::optional<resolution_t>> getDimensions(const std::string& dimensions);
std::variant<absl::Status, std::optional<resolution_t>> getDimensions(const HttpPayload& payload);

std::variant<absl::Status, std::string> getPromptField(const HttpPayload& payload);

std::variant<absl::Status, std::optional<std::string>> getStringFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<int64_t>> getInt64FromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<int>> getIntFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<size_t>> getSizetFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<float>> getFloatFromPayload(const ovms::HttpPayload& payload, const std::string& keyName);

std::variant<absl::Status, ov::AnyMap> getImageGenerationRequestOptions(const HttpPayload& payload, const ImageGenPipelineArgs& args);
std::variant<absl::Status, ov::AnyMap> getImageVariationRequestOptions(const HttpPayload& payload, const ImageGenPipelineArgs& args);
std::variant<absl::Status, ov::AnyMap> getImageEditRequestOptions(const HttpPayload& payload, const ImageGenPipelineArgs& args);

std::unique_ptr<std::string> generateJSONResponseFromB64Image(const std::string& base64_image);
}  // namespace ovms
