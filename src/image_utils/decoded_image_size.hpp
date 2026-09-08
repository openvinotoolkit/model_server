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

#include <cstdint>
#include <string_view>

namespace ovms {
namespace image_utils {

// Estimates the decoded (uncompressed) size in bytes of an encoded image by reading ONLY its
// header - it never allocates a decode buffer. The estimate uses the header-declared dimensions,
// which are the same values a decoder uses to size its allocation, so it upper-bounds the memory
// a full decode would need.
//
// Returns true and sets outDecodedBytes when the format is recognized and the header parses within
// bounds. Returns false when the format is unrecognized or the header is malformed/truncated; in
// that case the caller decides how to proceed (fail-open, relying on the OpenCV pre-decode pixel
// guard and the post-decode byte budget). The function is bounds-checked and must never read past
// the provided buffer nor crash on malformed input.
bool tryEstimateDecodedImageSize(std::string_view data, uint64_t& outDecodedBytes);

}  // namespace image_utils
}  // namespace ovms
