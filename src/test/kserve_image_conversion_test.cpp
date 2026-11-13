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
#include <array>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openvino/runtime/tensor.hpp>

#include "src/image_utils/kserve_image_conversion.hpp"
#include "src/status.hpp"
#include "stb_image_write.h"  // NOLINT

namespace ovms {
namespace {
using image_utils::decodeImage;
using image_utils::ImageDecoder;
using image_utils::preprocessImage;

std::string png(const unsigned char* pixels, int channels) {
    std::string encoded;
    auto write = [](void* context, void* data, int size) {
        static_cast<std::string*>(context)->append(static_cast<const char*>(data), size);
    };
    if (!stbi_write_png_to_func(write, &encoded, 1, 1, channels, pixels, channels))
        throw std::runtime_error("Unable to encode PNG test fixture");
    return encoded;
}

TEST(KserveImageConversion, StbPreservesChannelsAndBgrOrder) {
    const std::array<unsigned char, 4> pixels{10, 20, 30, 40};
    for (int channels : {1, 3, 4}) {
        ov::Tensor tensor;
        ASSERT_TRUE(decodeImage(png(pixels.data(), channels), tensor, ImageDecoder::STB).ok());
        EXPECT_EQ(tensor.get_shape(), (ov::Shape{1, 1, 1, static_cast<size_t>(channels)}));
        EXPECT_EQ(tensor.get_element_type(), ov::element::u8);
        EXPECT_EQ(tensor.data<unsigned char>()[0], channels >= 3 ? 30 : 10);
        if (channels >= 3) {
            EXPECT_EQ(tensor.data<unsigned char>()[1], 20);
            EXPECT_EQ(tensor.data<unsigned char>()[2], 10);
        }
        if (channels == 4) {
            EXPECT_EQ(tensor.data<unsigned char>()[3], 40);
        }
    }
}

TEST(KserveImageConversion, GrayAlphaExpandsToBgra) {
    const std::array<unsigned char, 2> pixels{23, 47};
    ov::Tensor tensor;
    ASSERT_TRUE(decodeImage(png(pixels.data(), 2), tensor, ImageDecoder::STB).ok());
    EXPECT_EQ(tensor.get_shape(), (ov::Shape{1, 1, 1, 4}));
    const std::array<unsigned char, 4> expected{23, 23, 23, 47};
    EXPECT_EQ(std::memcmp(tensor.data(), expected.data(), expected.size()), 0);
}

TEST(KserveImageConversion, StbPreserves16BitPng) {
    std::ifstream input("src/test/images/gray16.png", std::ios::binary);
    ASSERT_TRUE(input.good());
    const std::string encoded{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ov::Tensor tensor;
    ASSERT_TRUE(decodeImage(encoded, tensor, ImageDecoder::STB).ok());
    EXPECT_EQ(tensor.get_element_type(), ov::element::u16);
    EXPECT_EQ(tensor.get_shape(), (ov::Shape{1, 11, 7, 1}));
}

TEST(KserveImageConversion, MalformedImageIsNotASuccessfulFallback) {
    ov::Tensor tensor(ov::element::u8, {1, 1, 1, 3});
    const Status status = decodeImage("not an image", tensor);
    EXPECT_EQ(status.getCode(), StatusCode::IMAGE_PARSING_FAILED);
    EXPECT_FALSE(tensor);
    if (!image_utils::openCvImageDecoderUnavailableReason().empty()) {
        EXPECT_NE(status.string().find("OpenCV"), std::string::npos);
        EXPECT_NE(status.string().find("unavailable"), std::string::npos);
    }
}

TEST(KserveImageConversion, OpenCvOnlyFormatHasExplicitDependencyError) {
    const float value = 0.5;
    std::string pfm = "Pf\n1 1\n-1.0\n";
    pfm.append(reinterpret_cast<const char*>(&value), sizeof(value));
    ov::Tensor tensor;
    const Status status = decodeImage(pfm, tensor);
    if (image_utils::openCvImageDecoderUnavailableReason().empty()) {
        ASSERT_TRUE(status.ok()) << status.string();
        EXPECT_EQ(tensor.get_element_type(), ov::element::f32);
        EXPECT_FLOAT_EQ(tensor.data<float>()[0], value);
    } else {
        EXPECT_EQ(status.getCode(), StatusCode::IMAGE_PARSING_FAILED);
        EXPECT_NE(status.string().find("OpenCV"), std::string::npos);
        EXPECT_NE(status.string().find("missing"), std::string::npos);
        EXPECT_NE(status.string().find("Install OpenCV"), std::string::npos);
    }
}

TEST(KserveImageConversion, ExplicitOpenCvSelectionReportsMissingDependency) {
    if (image_utils::openCvImageDecoderUnavailableReason().empty()) {
        GTEST_SKIP() << "Requires isolated runtime without the OpenCV plugin";
    }
    const std::array<unsigned char, 3> pixels{10, 20, 30};
    ov::Tensor tensor;
    const Status status = decodeImage(png(pixels.data(), 3), tensor, ImageDecoder::OPENCV);
    EXPECT_EQ(status.getCode(), StatusCode::IMAGE_PARSING_FAILED);
    EXPECT_NE(status.string().find("OpenCV"), std::string::npos);
    EXPECT_NE(status.string().find("missing"), std::string::npos);
}

TEST(KserveImageConversion, ResizeAndPrecisionConversion) {
    ov::Tensor input(ov::element::u8, {1, 1, 1, 3});
    const std::array<unsigned char, 3> pixels{10, 20, 30};
    std::memcpy(input.data(), pixels.data(), pixels.size());
    ov::Tensor output;
    const Status status = preprocessImage(input, output, ov::element::f32, 2, 2);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_shape(), (ov::Shape{1, 2, 2, 3}));
    for (size_t i = 0; i < output.get_size(); ++i)
        EXPECT_FLOAT_EQ(output.data<float>()[i], pixels[i % 3]);
}

TEST(KserveImageConversion, SaturatesNarrowingConversion) {
    ov::Tensor input(ov::element::u16, {1, 1, 1, 3});
    const std::array<uint16_t, 3> pixels{0, 300, 65535};
    std::memcpy(input.data(), pixels.data(), sizeof(pixels));
    ov::Tensor output;
    const Status status = preprocessImage(input, output, ov::element::u8, 1, 1);
    ASSERT_TRUE(status.ok()) << status.string();
    const std::array<unsigned char, 3> expected{0, 255, 255};
    EXPECT_EQ(std::memcmp(output.data(), expected.data(), expected.size()), 0);
}

TEST(KserveImageConversion, SupportsExistingDestinationPrecisions) {
    const std::array<ov::element::Type, 8> precisions{
        ov::element::u8, ov::element::u16, ov::element::i8, ov::element::i16,
        ov::element::i32, ov::element::f16, ov::element::f32, ov::element::f64};
    ov::Tensor input(ov::element::u8, {1, 1, 1, 1});
    input.data<unsigned char>()[0] = 42;
    for (const auto& precision : precisions) {
        ov::Tensor output;
        const Status status = preprocessImage(input, output, precision, 2, 2);
        ASSERT_TRUE(status.ok()) << precision << ": " << status.string();
        EXPECT_EQ(output.get_element_type(), precision);
    }
}

TEST(KserveImageConversion, RoundsAndSaturatesSignedConversion) {
    ov::Tensor input(ov::element::f32, {1, 1, 1, 3});
    const std::array<float, 3> values{-2.6, 2.6, 300};
    std::memcpy(input.data(), values.data(), sizeof(values));
    ov::Tensor output;
    const Status status = preprocessImage(input, output, ov::element::i8, 1, 1);
    ASSERT_TRUE(status.ok()) << status.string();
    const std::array<int8_t, 3> expected{-3, 3, 127};
    EXPECT_EQ(std::memcmp(output.data(), expected.data(), expected.size()), 0);
}

TEST(KserveImageConversion, ConcurrentPreprocessingDoesNotAliasOutputs) {
    std::vector<std::future<ov::Tensor>> outputs;
    for (int value = 0; value < 8; ++value) {
        outputs.push_back(std::async(std::launch::async, [value] {
            ov::Tensor input(ov::element::u8, {1, 1, 1, 1});
            input.data<unsigned char>()[0] = value;
            ov::Tensor output;
            const Status status = preprocessImage(input, output, ov::element::f32, 2, 2);
            if (!status.ok())
                throw std::runtime_error(status.string());
            return output;
        }));
    }
    for (int value = 0; value < 8; ++value) {
        const ov::Tensor output = outputs[value].get();
        for (size_t i = 0; i < output.get_size(); ++i)
            EXPECT_FLOAT_EQ(output.data<float>()[i], value);
    }
}

TEST(KserveImageConversion, RejectsInvalidPreprocessingShape) {
    ov::Tensor input(ov::element::u8, {1});
    ov::Tensor output;
    EXPECT_EQ(preprocessImage(input, output, ov::element::u8, 1, 1).getCode(), StatusCode::INVALID_SHAPE);
}
}  // namespace
}  // namespace ovms
