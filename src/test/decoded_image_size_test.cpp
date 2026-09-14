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
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "../image_utils/decoded_image_size.hpp"
#include "platform_utils.hpp"

using ovms::image_utils::DecodedSizeEstimate;
using ovms::image_utils::estimateDecodedImageSize;

namespace {

constexpr uint32_t kW = 7;
constexpr uint32_t kH = 11;

void putLE16(std::string& s, uint16_t v) {
    s.push_back(static_cast<char>(v & 0xFF));
    s.push_back(static_cast<char>((v >> 8) & 0xFF));
}
void putLE32(std::string& s, uint32_t v) {
    s.push_back(static_cast<char>(v & 0xFF));
    s.push_back(static_cast<char>((v >> 8) & 0xFF));
    s.push_back(static_cast<char>((v >> 16) & 0xFF));
    s.push_back(static_cast<char>((v >> 24) & 0xFF));
}
void putLE24(std::string& s, uint32_t v) {
    s.push_back(static_cast<char>(v & 0xFF));
    s.push_back(static_cast<char>((v >> 8) & 0xFF));
    s.push_back(static_cast<char>((v >> 16) & 0xFF));
}

std::string makeWebpVp8l(uint32_t w, uint32_t h) {
    std::string s;
    s.append("RIFF");
    putLE32(s, 0);
    s.append("WEBP");
    s.append("VP8L");
    putLE32(s, 5);
    s.push_back(static_cast<char>(0x2F));  // VP8L signature
    uint32_t bits = ((w - 1) & 0x3FFF) | (((h - 1) & 0x3FFF) << 14);
    putLE32(s, bits);
    while (s.size() < 30)  // parser requires at least 30 bytes
        s.push_back(0);
    return s;
}

std::string makeWebpVp8(uint32_t w, uint32_t h) {
    std::string s;
    s.append("RIFF");
    putLE32(s, 0);
    s.append("WEBP");
    s.append("VP8 ");
    putLE32(s, 10);  // chunk size
    s.push_back(0);  // frame tag
    s.push_back(0);
    s.push_back(0);
    s.push_back(static_cast<char>(0x9D));  // start code
    s.push_back(static_cast<char>(0x01));
    s.push_back(static_cast<char>(0x2A));
    putLE16(s, static_cast<uint16_t>(w & 0x3FFF));
    putLE16(s, static_cast<uint16_t>(h & 0x3FFF));
    return s;
}

std::string makeWebpVp8x(uint32_t w, uint32_t h) {
    std::string s;
    s.append("RIFF");
    putLE32(s, 0);  // riff size (unused)
    s.append("WEBP");
    s.append("VP8X");
    putLE32(s, 10);  // chunk size
    s.push_back(0);  // flags
    s.push_back(0);
    s.push_back(0);
    s.push_back(0);
    putLE24(s, w - 1);
    putLE24(s, h - 1);
    return s;
}

std::string readImageFixture(const std::string& name) {
    const std::string path = getGenericFullPathForSrcTest("/ovms/src/test/images/" + name);
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

struct RealImageCase {
    const char* file;
    uint64_t expectedPixels;
};

class DecodedImageSizeRealFileTest : public ::testing::TestWithParam<RealImageCase> {};

// Real images produced by a genuine encoder (see generate_test_images.py): the estimate
// must match the decoded pixel count derived from the known fixture dimensions.
TEST_P(DecodedImageSizeRealFileTest, MatchesDecodedSize) {
    const RealImageCase& c = GetParam();
    const std::string data = readImageFixture(c.file);
    ASSERT_FALSE(data.empty()) << "fixture missing: " << c.file;
    uint64_t pixels = 0;
    ASSERT_EQ(estimateDecodedImageSize(data, pixels), DecodedSizeEstimate::Estimated) << "format not recognized: " << c.file;
    EXPECT_EQ(pixels, c.expectedPixels) << "file: " << c.file;
}

INSTANTIATE_TEST_SUITE_P(
    RealImages,
    DecodedImageSizeRealFileTest,
    ::testing::Values(
        RealImageCase{"rgb8.png", uint64_t(kW) * kH},
        RealImageCase{"rgba8.png", uint64_t(kW) * kH},
        RealImageCase{"gray16.png", uint64_t(kW) * kH},
        RealImageCase{"rgb.jpg", uint64_t(kW) * kH},
        RealImageCase{"rgb24.bmp", uint64_t(kW) * kH},
        RealImageCase{"sample.gif", uint64_t(kW) * kH},
        RealImageCase{"lossy.webp", uint64_t(kW) * kH},
        RealImageCase{"lossless.webp", uint64_t(kW) * kH}));

// WebP is handled by our own fallback parser (stb_image has no WebP support);
// exercise all three header variants directly.
TEST(DecodedImageSizeTest, WebpVp8xExtended) {
    uint64_t pixels = 0;
    ASSERT_EQ(estimateDecodedImageSize(makeWebpVp8x(1024, 768), pixels), DecodedSizeEstimate::Estimated);
    EXPECT_EQ(pixels, 1024ull * 768ull);
}

TEST(DecodedImageSizeTest, WebpVp8lLossless) {
    uint64_t pixels = 0;
    ASSERT_EQ(estimateDecodedImageSize(makeWebpVp8l(300, 200), pixels), DecodedSizeEstimate::Estimated);
    EXPECT_EQ(pixels, 300ull * 200ull);
}

TEST(DecodedImageSizeTest, WebpVp8Lossy) {
    uint64_t pixels = 0;
    ASSERT_EQ(estimateDecodedImageSize(makeWebpVp8(640, 480), pixels), DecodedSizeEstimate::Estimated);
    EXPECT_EQ(pixels, 640ull * 480ull);
}

TEST(DecodedImageSizeTest, UnrecognizedFormatUnsupported) {
    uint64_t bytes = 123;
    EXPECT_EQ(estimateDecodedImageSize(std::string("not an image at all"), bytes), DecodedSizeEstimate::UnsupportedFormat);
}

TEST(DecodedImageSizeTest, EmptyInputUnsupported) {
    uint64_t bytes = 0;
    EXPECT_EQ(estimateDecodedImageSize(std::string(), bytes), DecodedSizeEstimate::UnsupportedFormat);
}

// Truncated real files and synthetic WebP headers must never crash.
TEST(DecodedImageSizeTest, TruncatedInputsDoNotCrash) {
    std::vector<std::string> imgs = {
        readImageFixture("rgb8.png"),
        readImageFixture("rgb.jpg"),
        readImageFixture("rgb24.bmp"),
        readImageFixture("sample.gif"),
        readImageFixture("lossy.webp"),
        makeWebpVp8x(1024, 768),
        makeWebpVp8l(300, 200),
        makeWebpVp8(640, 480),
    };
    for (const auto& img : imgs) {
        for (size_t len = 0; len <= img.size(); ++len) {
            uint64_t bytes = 0;
            (void)estimateDecodedImageSize(std::string_view(img.data(), len), bytes);
        }
    }
    SUCCEED();
}

// Garbage after a valid magic must not crash.
TEST(DecodedImageSizeTest, GarbageAfterMagicDoesNotCrash) {
    std::string prefixes[] = {
        std::string("\x89PNG\r\n\x1a\n", 8),
        std::string("\xFF\xD8\xFF", 3),
        std::string("BM", 2),
        std::string("GIF89a", 6),
        std::string("RIFF", 4),
    };
    for (auto& p : prefixes) {
        std::string s = p;
        for (int i = 0; i < 64; ++i)
            s.push_back(static_cast<char>((i * 37 + 11) & 0xFF));
        uint64_t bytes = 0;
        (void)estimateDecodedImageSize(s, bytes);
    }
    SUCCEED();
}
