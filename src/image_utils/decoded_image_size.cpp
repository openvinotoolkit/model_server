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

bool readLittleEndian16(std::string_view d, size_t off, uint32_t& out) {
    if (off + 2 > d.size())
        return false;
    out = static_cast<uint32_t>(byteAt(d, off)) | (static_cast<uint32_t>(byteAt(d, off + 1)) << 8);
    return true;
}

bool readLittleEndian24(std::string_view d, size_t off, uint32_t& out) {
    if (off + 3 > d.size())
        return false;
    out = static_cast<uint32_t>(byteAt(d, off)) |
          (static_cast<uint32_t>(byteAt(d, off + 1)) << 8) |
          (static_cast<uint32_t>(byteAt(d, off + 2)) << 16);
    return true;
}

bool readLittleEndian32(std::string_view d, size_t off, uint32_t& out) {
    if (off + 4 > d.size())
        return false;
    out = static_cast<uint32_t>(byteAt(d, off)) |
          (static_cast<uint32_t>(byteAt(d, off + 1)) << 8) |
          (static_cast<uint32_t>(byteAt(d, off + 2)) << 16) |
          (static_cast<uint32_t>(byteAt(d, off + 3)) << 24);
    return true;
}

uint64_t saturatingMul(uint64_t a, uint64_t b) {
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

uint64_t decodedPixels(uint64_t width, uint64_t height) {
    return saturatingMul(width, height);
}

// WebP header magic values. Specs:
//   WebP container (RIFF, VP8X):  https://developers.google.com/speed/webp/docs/riff_container
//   VP8 lossy frame (start code): https://www.rfc-editor.org/info/rfc6386  (section 9.1)
//   VP8L lossless bitstream:      https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification
constexpr uint8_t VP8L_SIGNATURE_BYTE = 0x2F;        // first byte of a VP8L bitstream
constexpr uint8_t VP8_KEYFRAME_START_CODE_0 = 0x9D;  // VP8 lossy keyframe start code (3 bytes)
constexpr uint8_t VP8_KEYFRAME_START_CODE_1 = 0x01;
constexpr uint8_t VP8_KEYFRAME_START_CODE_2 = 0x2A;
constexpr uint32_t WEBP_DIMENSION_MASK_14BIT = 0x3FFF;  // width/height stored in 14-bit fields
constexpr int WEBP_DIMENSION_BITS_14 = 14;

// Fallback header parser for WebP, which stb_image does not support but OpenCV decodes.
[[nodiscard]] bool parseWebp(std::string_view d, uint64_t& out) {
    if (d.size() < 30 || !startsWith(d, "RIFF", 4) || byteAt(d, 8) != 'W' ||
        byteAt(d, 9) != 'E' || byteAt(d, 10) != 'B' || byteAt(d, 11) != 'P')
        return false;
    uint64_t width = 0, height = 0;
    if (startsWith(d.substr(12), "VP8X", 4)) {
        // Extended: flags(1) at 20, canvas width-1 (3 bytes little-endian) at 24, height-1 at 27.
        uint32_t w1 = 0, h1 = 0;
        if (!readLittleEndian24(d, 24, w1) || !readLittleEndian24(d, 27, h1))
            return false;
        width = static_cast<uint64_t>(w1) + 1;
        height = static_cast<uint64_t>(h1) + 1;
    } else if (startsWith(d.substr(12), "VP8L", 4)) {
        // Lossless: VP8L signature byte at 20, then packed 14-bit width-1 and height-1.
        if (byteAt(d, 20) != VP8L_SIGNATURE_BYTE)
            return false;
        uint32_t bits = 0;
        if (!readLittleEndian32(d, 21, bits))
            return false;
        width = static_cast<uint64_t>(bits & WEBP_DIMENSION_MASK_14BIT) + 1;
        height = static_cast<uint64_t>((bits >> WEBP_DIMENSION_BITS_14) & WEBP_DIMENSION_MASK_14BIT) + 1;
    } else if (startsWith(d.substr(12), "VP8 ", 4)) {
        // Lossy: 3-byte keyframe start code at 23, then 14-bit width and height.
        if (byteAt(d, 23) != VP8_KEYFRAME_START_CODE_0 || byteAt(d, 24) != VP8_KEYFRAME_START_CODE_1 ||
            byteAt(d, 25) != VP8_KEYFRAME_START_CODE_2)
            return false;
        uint32_t w = 0, h = 0;
        if (!readLittleEndian16(d, 26, w) || !readLittleEndian16(d, 28, h))
            return false;
        width = w & WEBP_DIMENSION_MASK_14BIT;
        height = h & WEBP_DIMENSION_MASK_14BIT;
    } else {
        return false;
    }
    if (width == 0 || height == 0)
        return false;
    out = decodedPixels(width, height);
    return true;
}

    // Reads only the image header via stb_image (no pixel decode or allocation). Fills width/height
    // for formats stb recognizes (PNG, JPEG, BMP, GIF, ...). Returns false for empty/oversized input
    // or formats stb does not support (e.g. WebP), in which case the caller uses its own fallback.
    [[nodiscard]] bool tryReadImageHeaderStbi(std::string_view imageBytes, int& width, int& height) {
    width = 0;
    height = 0;
    if (imageBytes.empty() || imageBytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    const stbi_uc* buffer = reinterpret_cast<const stbi_uc*>(imageBytes.data());
    int len = static_cast<int>(imageBytes.size());
    int channels = 0;
    if (!stbi_info_from_memory(buffer, len, &width, &height, &channels))
        return false;
    return true;
}

}  // namespace

[[nodiscard]] DecodedSizeEstimate estimateDecodedImageSize(std::string_view data, uint64_t& outDecodedPixels) {
    // Buffer too large for stb's int-based API and far beyond any sane budget.
    if (data.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return DecodedSizeEstimate::InputTooLarge;
    // Primary: reuse the stb_image header parser (PNG, JPEG, BMP, GIF, ...) instead of
    // maintaining per-format header parsers ourselves.
    int width = 0, height = 0;
    if (tryReadImageHeaderStbi(data, width, height)) {
        if (width <= 0 || height <= 0)
            return DecodedSizeEstimate::UnsupportedFormat;
        outDecodedPixels = decodedPixels(static_cast<uint64_t>(width), static_cast<uint64_t>(height));
        return DecodedSizeEstimate::Estimated;
    }
    // Fallback for formats stb_image does not support but OpenCV decodes (e.g. WebP).
    if (startsWith(data, "RIFF", 4) && parseWebp(data, outDecodedPixels))
        return DecodedSizeEstimate::Estimated;
    return DecodedSizeEstimate::UnsupportedFormat;
}

}  // namespace image_utils
}  // namespace ovms
