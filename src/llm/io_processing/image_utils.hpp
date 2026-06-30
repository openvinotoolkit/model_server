//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <optional>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/status/statusor.h"
#pragma warning(pop)

#include "openvino/runtime/tensor.hpp"

namespace ovms {

constexpr std::string_view BASE64_PREFIX = "base64,";
constexpr int64_t MAX_IMAGE_SIZE_BYTES = 20000000;  // 20MB

// Loads an image from a base64 data URI, HTTP/HTTPS URL, or local file path.
// Returns the decoded image as an ov::Tensor (RGB, u8).
absl::StatusOr<ov::Tensor> loadImage(const std::string& imageSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains);

}  // namespace ovms
