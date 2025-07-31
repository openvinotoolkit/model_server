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
#include <vector>
#include <utility>

#include <openvino/openvino.hpp>
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/status/status.h"
#pragma warning(pop)

#include "imagegenpipelineargs.hpp"

#pragma warning(push)
#pragma warning(disable : 4996 4244 4267 4127)
#include "rapidjson/document.h"
#pragma warning(pop)

#define RETURN_IF_HOLDS_STATUS(NAME)                  \
    if (std::holds_alternative<absl::Status>(NAME)) { \
        return std::get<absl::Status>(NAME);          \
    }

#define SET_OR_RETURN(TYPE, NAME, RHS) \
    auto NAME##_OPT = RHS;             \
    RETURN_IF_HOLDS_STATUS(NAME##_OPT) \
    auto NAME = std::get<TYPE>(NAME##_OPT);

namespace ovms {
struct MultiPartParser;
std::variant<absl::Status, std::optional<resolution_t>> getDimensions(const std::string& dimensions);
std::variant<absl::Status, std::optional<resolution_t>> getDimensions(const rapidjson::Document& doc);
std::variant<absl::Status, std::optional<resolution_t>> getDimensions(const ovms::MultiPartParser& parser);

std::variant<absl::Status, std::string> getPromptField(const rapidjson::Document& doc);
std::variant<absl::Status, std::string> getPromptField(const ovms::MultiPartParser& payload);

std::variant<absl::Status, std::optional<std::string>> getStringFromPayload(const rapidjson::Document& doc, const std::string& keyName);
std::variant<absl::Status, std::optional<std::string>> getStringFromPayload(const ovms::MultiPartParser& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<std::string_view>> getFileFromPayload(const ovms::MultiPartParser& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<int64_t>> getInt64FromPayload(const rapidjson::Document& doc, const std::string& keyName);
std::variant<absl::Status, std::optional<int64_t>> getInt64FromPayload(const ovms::MultiPartParser& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<int>> getIntFromPayload(const rapidjson::Document& doc, const std::string& keyName);
std::variant<absl::Status, std::optional<int>> getIntFromPayload(const ovms::MultiPartParser& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<size_t>> getSizetFromPayload(const rapidjson::Document& doc, const std::string& keyName);
std::variant<absl::Status, std::optional<size_t>> getSizetFromPayload(const ovms::MultiPartParser& payload, const std::string& keyName);
std::variant<absl::Status, std::optional<float>> getFloatFromPayload(const rapidjson::Document& doc, const std::string& keyName);
std::variant<absl::Status, std::optional<float>> getFloatFromPayload(const ovms::MultiPartParser& payload, const std::string& keyName);

std::variant<absl::Status, ov::AnyMap> getImageGenerationRequestOptions(const rapidjson::Document& doc, const ImageGenPipelineArgs& args);
std::variant<absl::Status, ov::AnyMap> getImageEditRequestOptions(const ovms::MultiPartParser& payload, const ImageGenPipelineArgs& args);

std::unique_ptr<std::string> generateJSONResponseFromB64Images(const std::vector<std::string>& base64Images);

std::variant<absl::Status, std::unique_ptr<std::string>> generateJSONResponseFromOvTensor(const ov::Tensor& tensor);
}  // namespace ovms
