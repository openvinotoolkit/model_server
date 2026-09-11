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
#include "decoded_image_size.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#pragma warning(push)
#pragma warning(disable : 6262 6386 6385)
#include "stb_image.h"  // NOLINT
#pragma warning(pop)

namespace ovms {
namespace image_utils {
namespace {

uint8_t byteAt(std::string_view d, size_t off) {
    return static_cast<uint8_t>(d[off]);
}

bool readLE16(std::string_view d, size_t off, uint32_t& out) {
    if (off + 2 > d.size())
        return false;
    out = static_cast<uint32_t>(byteAt(d, off)) | (static_cast<uint32_t>(byteAt(d, off + 1)) << 8);
    return true;
}

bool readLE24(std::string_view d, size_t off, uint32_t& out) {
    if (off + 3 > d.size())
        return false;
    out = static_cast<uint32_t>(byteAt(d, off)) |
          (static_cast<uint32_t>(byteAt(d, off + 1)) << 8) |
          (static_cast<uint32_t>(byteAt(d, off + 2)) << 16);
    return true;
}

bool readLE32(std::string_view d, size_t off, uint32_t& out) {
    if (off + 4 > d.size())
        return false;
    out = static_cast<uint32_t>(byteAt(d, off)) |
          (static_cast<uint32_t>(byteAt(d, off + 1)) << 8) |
          (static_cast<uint32_t>(byteAt(d, off + 2)) << 16) |
          (static_cast<uint32_t>(byteAt(d, off + 3)) << 24);
    return true;
}

uint64_t satMul(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0)
        return 0;
    if (a > std::numeric_limits<uint64_t>::max() / b)
        return std::numeric_limits<uint64_t>::max();
    return a * b;
}

[[nodiscard]] bool startsWith(std::string_view d, const char* prefix, size_t n) {
    if (d.size() < n)
        return false;
    for (size_t i = 0; i < n; ++i) {
        if (byteAt(d, i) != static_cast<uint8_t>(prefix[i]))
            return false;
    }
    return true;
}

uint64_t decodedBytes(uint64_t width, uint64_t height, uint64_t bytesPerPixel) {
    return satMul(satMul(width, height), bytesPerPixel);
}

// Fallback header parser for WebP, which stb_image does not support but OpenCV decodes.
[[nodiscard]] bool parseWebp(std::string_view d, uint64_t& out) {
    if (d.size() < 30 || !startsWith(d, "RIFF", 4) || byteAt(d, 8) != 'W' ||
        byteAt(d, 9) != 'E' || byteAt(d, 10) != 'B' || byteAt(d, 11) != 'P')
        return false;
    uint64_t width = 0, height = 0;
    if (startsWith(d.substr(12), "VP8X", 4)) {
        // Extended: flags(1) at 20, canvas width-1 (3 LE) at 24, height-1 (3 LE) at 27.
        uint32_t w1 = 0, h1 = 0;
        if (!readLE24(d, 24, w1) || !readLE24(d, 27, h1))
            return false;
        width = static_cast<uint64_t>(w1) + 1;
        height = static_cast<uint64_t>(h1) + 1;
    } else if (startsWith(d.substr(12), "VP8L", 4)) {
        // Lossless: signature byte 0x2F at 20, then 14-bit width-1 and height-1 packed.
        if (byteAt(d, 20) != 0x2F)
            return false;
        uint32_t bits = 0;
        if (!readLE32(d, 21, bits))
            return false;
        width = static_cast<uint64_t>(bits & 0x3FFF) + 1;
        height = static_cast<uint64_t>((bits >> 14) & 0x3FFF) + 1;
    } else if (startsWith(d.substr(12), "VP8 ", 4)) {
        // Lossy: keyframe start code 0x9D 0x01 0x2A at 23, then 14-bit width and height.
        if (byteAt(d, 23) != 0x9D || byteAt(d, 24) != 0x01 || byteAt(d, 25) != 0x2A)
            return false;
        uint32_t w = 0, h = 0;
        if (!readLE16(d, 26, w) || !readLE16(d, 28, h))
            return false;
        width = w & 0x3FFF;
        height = h & 0x3FFF;
    } else {
        return false;
    }
    if (width == 0 || height == 0)
        return false;
    // WebP frames are expanded to BGRA by decoders.
    out = decodedBytes(width, height, 4);
    return true;
}

    // Reads only the image header via stb_image (no pixel decode or allocation). Fills
    // width/height/bytesPerPixel (channel count times 8- vs 16-bit sample width) for formats stb
    // recognizes (PNG, JPEG, BMP, GIF, ...). Returns false for empty/oversized input or formats stb
    // does not support (e.g. WebP), in which case the caller uses its own fallback.
    [[nodiscard]] bool tryReadImageHeaderStbi(std::string_view imageBytes, int& width, int& height, int& bytesPerPixel) {
    width = 0;
    height = 0;
    bytesPerPixel = 0;
    if (imageBytes.empty() || imageBytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    const stbi_uc* buffer = reinterpret_cast<const stbi_uc*>(imageBytes.data());
    int len = static_cast<int>(imageBytes.size());
    int channels = 0;
    if (!stbi_info_from_memory(buffer, len, &width, &height, &channels))
        return false;
    int bytesPerSample = stbi_is_16_bit_from_memory(buffer, len) != 0 ? 2 : 1;
    bytesPerPixel = channels * bytesPerSample;
    return true;
}

}  // namespace

[[nodiscard]] DecodedSizeEstimate estimateDecodedImageSize(std::string_view data, uint64_t& outDecodedBytes) {
    // Buffer too large for stb's int-based API and far beyond any sane budget.
    if (data.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return DecodedSizeEstimate::InputTooLarge;
    // Primary: reuse the stb_image header parser (PNG, JPEG, BMP, GIF, ...) instead of
    // maintaining per-format header parsers ourselves.
    int width = 0, height = 0, bytesPerPixel = 0;
    if (tryReadImageHeaderStbi(data, width, height, bytesPerPixel)) {
        if (width <= 0 || height <= 0 || bytesPerPixel <= 0)
            return DecodedSizeEstimate::UnsupportedFormat;
        outDecodedBytes = decodedBytes(static_cast<uint64_t>(width),
            static_cast<uint64_t>(height),
            static_cast<uint64_t>(bytesPerPixel));
        return DecodedSizeEstimate::Estimated;
    }
    // Fallback for formats stb_image does not support but OpenCV decodes (e.g. WebP).
    if (startsWith(data, "RIFF", 4) && parseWebp(data, outDecodedBytes))
        return DecodedSizeEstimate::Estimated;
    return DecodedSizeEstimate::UnsupportedFormat;
}

}  // namespace image_utils
}  // namespace ovms
