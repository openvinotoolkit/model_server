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

namespace ovms {
namespace image_utils {
namespace {

uint8_t byteAt(std::string_view d, size_t off) {
    return static_cast<uint8_t>(d[off]);
}

bool readBE16(std::string_view d, size_t off, uint32_t& out) {
    if (off + 2 > d.size())
        return false;
    out = (static_cast<uint32_t>(byteAt(d, off)) << 8) | byteAt(d, off + 1);
    return true;
}

bool readBE32(std::string_view d, size_t off, uint32_t& out) {
    if (off + 4 > d.size())
        return false;
    out = (static_cast<uint32_t>(byteAt(d, off)) << 24) |
          (static_cast<uint32_t>(byteAt(d, off + 1)) << 16) |
          (static_cast<uint32_t>(byteAt(d, off + 2)) << 8) |
          byteAt(d, off + 3);
    return true;
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

bool startsWith(std::string_view d, const char* prefix, size_t n) {
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

bool parsePng(std::string_view d, uint64_t& out) {
    // Signature (8) + IHDR: length(4) + "IHDR"(4) + width(4) + height(4) + bitDepth(1) + colorType(1)
    if (d.size() < 26)
        return false;
    if (byteAt(d, 12) != 'I' || byteAt(d, 13) != 'H' || byteAt(d, 14) != 'D' || byteAt(d, 15) != 'R')
        return false;
    uint32_t width = 0, height = 0;
    if (!readBE32(d, 16, width) || !readBE32(d, 20, height))
        return false;
    if (width == 0 || height == 0)
        return false;
    uint32_t bitDepth = byteAt(d, 24);
    uint32_t colorType = byteAt(d, 25);
    uint64_t channels = 0;
    switch (colorType) {
    case 0:  // grayscale
    case 3:  // palette (decoders expand to RGB, but stay conservative with per-sample math below)
        channels = 1;
        break;
    case 2:  // RGB
        channels = 3;
        break;
    case 4:  // grayscale + alpha
        channels = 2;
        break;
    case 6:  // RGBA
        channels = 4;
        break;
    default:
        return false;
    }
    // Sub-8-bit depths expand to at least 1 byte per sample after decode.
    uint64_t bytesPerSample = bitDepth >= 8 ? (bitDepth / 8) : 1;
    // Palette images can be expanded to 3-4 channels by decoders; bound conservatively.
    if (colorType == 3)
        channels = 4;
    out = decodedBytes(width, height, satMul(channels, bytesPerSample));
    return true;
}

bool parseJpeg(std::string_view d, uint64_t& out) {
    if (d.size() < 4 || byteAt(d, 0) != 0xFF || byteAt(d, 1) != 0xD8)
        return false;
    size_t pos = 2;
    // Bound the number of segments scanned to avoid pathological loops.
    for (int guard = 0; guard < 4096; ++guard) {
        if (pos + 1 >= d.size())
            return false;
        if (byteAt(d, pos) != 0xFF)
            return false;
        // Skip any fill bytes (0xFF).
        while (pos < d.size() && byteAt(d, pos) == 0xFF)
            ++pos;
        if (pos >= d.size())
            return false;
        uint8_t marker = byteAt(d, pos);
        ++pos;
        // Standalone markers without a length field.
        if (marker == 0xD8 || marker == 0xD9 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01)
            continue;
        uint32_t segLen = 0;
        if (!readBE16(d, pos, segLen))
            return false;
        if (segLen < 2)
            return false;
        // SOF markers carry the frame dimensions. Exclude DHT(C4), JPG(C8), DAC(CC).
        bool isSof = (marker >= 0xC0 && marker <= 0xCF) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (isSof) {
            // segment: len(2) precision(1) height(2) width(2) components(1)
            uint32_t height = 0, width = 0;
            if (!readBE16(d, pos + 3, height) || !readBE16(d, pos + 5, width))
                return false;
            if (pos + 7 >= d.size())
                return false;
            uint32_t precision = byteAt(d, pos + 2);
            uint32_t components = byteAt(d, pos + 7);
            if (width == 0 || height == 0 || components == 0)
                return false;
            uint64_t bytesPerSample = precision >= 8 ? (precision / 8) : 1;
            out = decodedBytes(width, height, satMul(components, bytesPerSample));
            return true;
        }
        pos += segLen;  // length includes the 2 length bytes
    }
    return false;
}

bool parseBmp(std::string_view d, uint64_t& out) {
    // "BM" + file header(14) + DIB header. Support BITMAPINFOHEADER (size >= 40).
    if (d.size() < 30 || byteAt(d, 0) != 'B' || byteAt(d, 1) != 'M')
        return false;
    uint32_t dibSize = 0;
    if (!readLE32(d, 14, dibSize) || dibSize < 40)
        return false;
    uint32_t widthRaw = 0, heightRaw = 0, bitCount = 0;
    if (!readLE32(d, 18, widthRaw) || !readLE32(d, 22, heightRaw) || !readLE16(d, 28, bitCount))
        return false;
    int32_t widthS = static_cast<int32_t>(widthRaw);
    int32_t heightS = static_cast<int32_t>(heightRaw);
    if (widthS <= 0 || heightS == 0)
        return false;
    uint64_t width = static_cast<uint64_t>(widthS);
    uint64_t height = static_cast<uint64_t>(heightS < 0 ? -static_cast<int64_t>(heightS) : heightS);
    uint64_t bytesPerPixel = bitCount >= 8 ? (bitCount / 8) : 1;
    out = decodedBytes(width, height, bytesPerPixel);
    return true;
}

bool parseGif(std::string_view d, uint64_t& out) {
    if (!(startsWith(d, "GIF87a", 6) || startsWith(d, "GIF89a", 6)))
        return false;
    uint32_t width = 0, height = 0;
    if (!readLE16(d, 6, width) || !readLE16(d, 8, height))
        return false;
    if (width == 0 || height == 0)
        return false;
    // GIF frames are expanded to BGRA by decoders.
    out = decodedBytes(width, height, 4);
    return true;
}

bool parseWebp(std::string_view d, uint64_t& out) {
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

}  // namespace

bool tryEstimateDecodedImageSize(std::string_view data, uint64_t& outDecodedBytes) {
    if (data.size() >= 8 && byteAt(data, 0) == 0x89 && byteAt(data, 1) == 'P' &&
        byteAt(data, 2) == 'N' && byteAt(data, 3) == 'G')
        return parsePng(data, outDecodedBytes);
    if (data.size() >= 3 && byteAt(data, 0) == 0xFF && byteAt(data, 1) == 0xD8 && byteAt(data, 2) == 0xFF)
        return parseJpeg(data, outDecodedBytes);
    if (startsWith(data, "BM", 2))
        return parseBmp(data, outDecodedBytes);
    if (startsWith(data, "GIF8", 4))
        return parseGif(data, outDecodedBytes);
    if (startsWith(data, "RIFF", 4))
        return parseWebp(data, outDecodedBytes);
    return false;
}

}  // namespace image_utils
}  // namespace ovms
