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
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <openvino/runtime/tensor.hpp>

namespace ovms {

ov::Tensor loadImageStbi(unsigned char* image, const int x, const int y, const int desiredChannels);
ov::Tensor loadImageStbiFromMemory(std::string_view imageBytes);
ov::Tensor loadImageStbiFromFile(const char* filename);
std::vector<std::string> saveImagesStbi(const ov::Tensor& tensor);

// Reads only the image header (no pixel decode or allocation) via stb_image.
// Returns true and fills width/height/bytesPerPixel for formats stb recognizes
// (PNG, JPEG, BMP, GIF, ...). bytesPerPixel already accounts for channel count and
// 8- vs 16-bit sample depth. Returns false for empty/oversized input or formats stb
// does not support (e.g. WebP), in which case the caller must use its own fallback.
[[nodiscard]] bool tryReadImageHeaderStbi(std::string_view imageBytes, int& width, int& height, int& bytesPerPixel);

}  // namespace ovms
