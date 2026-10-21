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

#include <cstddef>
#include <string>
#include <string_view>

namespace ov {
class Tensor;
namespace element {
class Type;
}
}  // namespace ov
namespace ovms {
class Status;
namespace image_utils {

enum class ImageDecoder { AUTO,
    STB,
    OPENCV };

/// Decode into a single-image NHWC tensor, retaining depth and BGR/BGRA channel order.
/// AUTO prefers the optional OpenCV plugin, falling back to stb only if it is unavailable.
Status decodeImage(std::string_view encoded, ov::Tensor& tensor, ImageDecoder decoder = ImageDecoder::AUTO);

/// Return the runtime loader diagnostic, or an empty string when OpenCV decoding is available.
const std::string& openCvImageDecoderUnavailableReason();

/// Convert precision, then resize using a cached OpenVINO CPU preprocessing graph.
Status preprocessImage(const ov::Tensor& source, ov::Tensor& destination,
    const ov::element::Type& precision, size_t height, size_t width);

}  // namespace image_utils
}  // namespace ovms
