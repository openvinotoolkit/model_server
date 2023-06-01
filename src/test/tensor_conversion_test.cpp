//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../tensor_conversion.hpp"
#include "opencv2/opencv.hpp"
#include "test_utils.hpp"

using namespace ovms;

namespace {
template <typename TensorType>
class NativeFileInputConversionTest : public ::testing::Test {
public:
    TensorType requestTensor;
    void SetUp() override {
        prepareBinaryTensor(requestTensor);
    }

    void prepareBinaryTensor(tensorflow::TensorProto& tensor, std::unique_ptr<char[]>& image_bytes, const size_t filesize, const size_t batchSize = 1) {
        for (size_t i = 0; i < batchSize; i++) {
            tensor.add_string_val(image_bytes.get(), filesize);
        }
        tensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
        tensor.set_dtype(tensorflow::DataType::DT_STRING);
    }
    void prepareBinaryTensor(::KFSRequest::InferInputTensor& tensor, std::unique_ptr<char[]>& image_bytes, const size_t filesize, const size_t batchSize = 1) {
        for (size_t i = 0; i < batchSize; i++) {
            tensor.mutable_contents()->add_bytes_contents(image_bytes.get(), filesize);
        }
        tensor.mutable_shape()->Add(batchSize);
        tensor.set_datatype("BYTES");
    }
    void prepareBinaryTensor(tensorflow::TensorProto& tensor) {
        size_t filesize;
        std::unique_ptr<char[]> image_bytes;

        readRgbJpg(filesize, image_bytes);
        prepareBinaryTensor(tensor, image_bytes, filesize);
    }
    void prepareBinaryTensor(::KFSRequest::InferInputTensor& tensor) {
        size_t filesize;
        std::unique_ptr<char[]> image_bytes;

        readRgbJpg(filesize, image_bytes);
        prepareBinaryTensor(tensor, image_bytes, filesize);
    }

    void prepareBinaryTensor(tensorflow::TensorProto& tensor, std::string input) {
        tensor.set_dtype(tensorflow::DataType::DT_STRING);
        tensor.add_string_val(input);
    }
    void prepareBinaryTensor(::KFSRequest::InferInputTensor& tensor, std::string input) {
        tensor.mutable_contents()->add_bytes_contents(input);
        tensor.set_datatype("BYTES");
    }
};

using MyTypes = ::testing::Types<tensorflow::TensorProto, ::KFSRequest::InferInputTensor>;
TYPED_TEST_SUITE(NativeFileInputConversionTest, MyTypes);

TYPED_TEST(NativeFileInputConversionTest, tensorWithNonMatchingBatchsize) {
    ov::Tensor tensor;
    auto tensorInfo = std::make_shared<const TensorInfo>(
        "", ovms::Precision::U8, ovms::Shape{5, 1, 1, 1}, Layout{"NHWC"});
    EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::INVALID_BATCH_SIZE);
}

TYPED_TEST(NativeFileInputConversionTest, tensorWithInvalidImage) {
    TypeParam requestTensorInvalidImage;
    std::string invalidImage = "INVALID IMAGE";
    this->prepareBinaryTensor(requestTensorInvalidImage, invalidImage);

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(requestTensorInvalidImage, tensor, tensorInfo, nullptr), ovms::StatusCode::IMAGE_PARSING_FAILED);
}

TYPED_TEST(NativeFileInputConversionTest, tensorWithEmptyTensor) {
    TypeParam requestTensorEmptyInput;
    std::string emptyInput = "";
    this->prepareBinaryTensor(requestTensorEmptyInput, emptyInput);

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});
    if (std::is_same<TypeParam, tensorflow::TensorProto>::value)
        EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(requestTensorEmptyInput, tensor, tensorInfo, nullptr), ovms::StatusCode::STRING_VAL_EMPTY);
    else
        EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(requestTensorEmptyInput, tensor, tensorInfo, nullptr), StatusCode::BYTES_CONTENTS_EMPTY);
}

TYPED_TEST(NativeFileInputConversionTest, tensorWithNonSupportedLayout) {
    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NCHW"});

    EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TYPED_TEST(NativeFileInputConversionTest, tensorWithNonSupportedPrecision) {
    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::MIXED, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::INVALID_PRECISION);
}

TYPED_TEST(NativeFileInputConversionTest, tensorWithNonMatchingShapeSize) {
    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1}, Layout{"NC"});

    EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TYPED_TEST(NativeFileInputConversionTest, tensorWithNonMatchingNumberOfChannelsNHWC) {
    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, Layout{"NHWC"});

    EXPECT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::INVALID_NO_OF_CHANNELS);
}

TYPED_TEST(NativeFileInputConversionTest, positive_rgb) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_grayscale) {
    uint8_t grayscale_expected_tensor[] = {0x00};

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/grayscale.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t grayscale_filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> grayscale_image_bytes(new char[grayscale_filesize]);
    DataFile.read(grayscale_image_bytes.get(), grayscale_filesize);

    TypeParam grayscaleRequestTensor;
    ov::Tensor tensor;
    this->prepareBinaryTensor(grayscaleRequestTensor, grayscale_image_bytes, grayscale_filesize);

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(grayscaleRequestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 1);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), grayscale_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_batch_size_2) {
    uint8_t rgb_batchsize_2_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};
    ov::Tensor tensor;

    TypeParam batchSize2RequestTensor;
    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    readRgbJpg(filesize, image_bytes);
    size_t batchsize = 2;
    this->prepareBinaryTensor(batchSize2RequestTensor, image_bytes, filesize, batchsize);

    for (const auto layout : std::vector<Layout>{Layout("NHWC"), Layout::getDefaultLayout()}) {
        auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{2, 1, 1, 3}, layout);

        ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(batchSize2RequestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
        ASSERT_EQ(tensor.get_size(), 6);
        uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
        EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_batchsize_2_tensor), true);
    }
}

TYPED_TEST(NativeFileInputConversionTest, positive_precision_changed) {
    uint8_t rgb_precision_changed_expected_tensor[] = {0x24, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0xed, 0x00, 0x00, 0x00};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::I32, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    int I32_size = 4;
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size() * I32_size, rgb_precision_changed_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_nhwc_layout) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());

    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, layout_default_resolution_mismatch) {
    ov::Tensor tensor;
    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 3, 1, 3}, Layout::getDefaultLayout());
    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::INVALID_SHAPE);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 2, 2, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 12);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_dynamic_shape_cols_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {2, 5}, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 2;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_dynamic_shape_cols_bigger) {
    uint8_t rgb_expected_tensor[] = {0x96, 0x8f, 0xf3, 0x98, 0x9a, 0x81, 0x9d, 0xa9, 0x12};

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    read4x4RgbJpg(filesize, image_bytes);

    TypeParam requestTensor4x4;
    this->prepareBinaryTensor(requestTensor4x4, image_bytes, filesize);

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(requestTensor4x4, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 3;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_dynamic_shape_cols_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_dynamic_shape_rows_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {2, 5}, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 2;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_dynamic_shape_rows_bigger) {
    uint8_t rgb_expected_tensor[] = {0x3f, 0x65, 0x88, 0x98, 0x9a, 0x81, 0xf5, 0xd2, 0x7c};

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    read4x4RgbJpg(filesize, image_bytes);

    TypeParam requestTensor4x4;
    this->prepareBinaryTensor(requestTensor4x4, image_bytes, filesize);

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(requestTensor4x4, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 3;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_dynamic_shape_rows_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_any_shape) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), ovms::Dimension::any(), 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TYPED_TEST(NativeFileInputConversionTest, negative_resizing_with_one_any_one_static_shape) {
    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), 4, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::INVALID_SHAPE);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_one_any_one_static_shape) {
    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
}

TYPED_TEST(NativeFileInputConversionTest, positive_resizing_with_demultiplexer_and_range_resolution) {
    ov::Tensor tensor;

    const int batchSize = 5;
    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, {1, 3}, 3}, Layout{"NHWC"});
    tensorInfo = tensorInfo->createCopyWithDemultiplexerDimensionPrefix(batchSize);

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    read4x4RgbJpg(filesize, image_bytes);

    TypeParam requestTensor4x4;
    this->prepareBinaryTensor(requestTensor4x4, image_bytes, filesize, batchSize);

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(requestTensor4x4, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    ASSERT_EQ(tensorDims[0], batchSize);
    EXPECT_EQ(tensorDims[1], 1);
    EXPECT_EQ(tensorDims[2], 3);
    EXPECT_EQ(tensorDims[3], 3);
    EXPECT_EQ(tensorDims[4], 3);
    ASSERT_EQ(tensor.get_size(), batchSize * 1 * 3 * 3 * 3);
}

TYPED_TEST(NativeFileInputConversionTest, positive_range_resolution_matching_in_between) {
    ov::Tensor tensor;

    const int batchSize = 5;
    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    read4x4RgbJpg(filesize, image_bytes);

    TypeParam requestTensor4x4;
    this->prepareBinaryTensor(requestTensor4x4, image_bytes, filesize, batchSize);

    for (const auto& batchDim : std::vector<Dimension>{Dimension::any(), Dimension(batchSize)}) {
        auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{batchDim, {1, 5}, {1, 5}, 3}, Layout{"NHWC"});

        ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(requestTensor4x4, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
        shape_t tensorDims = tensor.get_shape();
        ASSERT_EQ(tensorDims[0], batchSize);
        EXPECT_EQ(tensorDims[1], 4);
        EXPECT_EQ(tensorDims[2], 4);
        EXPECT_EQ(tensorDims[3], 3);
        ASSERT_EQ(tensor.get_size(), batchSize * 4 * 4 * 3);
    }
}

class NativeFileInputConversionTFSPrecisionTest : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        readRgbJpg(filesize, image_bytes);
        stringVal.set_dtype(tensorflow::DataType::DT_STRING);
        stringVal.add_string_val(image_bytes.get(), filesize);
    }

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    tensorflow::TensorProto stringVal;
};

class NativeFileInputConversionTFSValidPrecisionTest : public NativeFileInputConversionTFSPrecisionTest {};
class NativeFileInputConversionTFSInvalidPrecisionTest : public NativeFileInputConversionTFSPrecisionTest {};

TEST_P(NativeFileInputConversionTFSValidPrecisionTest, Valid) {
    ovms::Precision testedPrecision = GetParam();

    auto tensorInfo = std::make_shared<const TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(stringVal, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_shape(), (ov::Shape{1, 1, 1, 3}));
    ASSERT_EQ(tensor.get_size(), 3);
    ASSERT_EQ(tensor.get_element_type(), ovmsPrecisionToIE2Precision(testedPrecision));
}

TEST_P(NativeFileInputConversionTFSInvalidPrecisionTest, Invalid) {
    ovms::Precision testedPrecision = GetParam();

    auto tensorInfo = std::make_shared<const TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(stringVal, tensor, tensorInfo, nullptr), ovms::StatusCode::INVALID_PRECISION);
}

const std::vector<ovms::Precision> BINARY_SUPPORTED_INPUT_PRECISIONS{
    // ovms::Precision::UNSPECIFIED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    // InferenceEngine::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    // ovms::Precision::I64,
    // ovms::Precision::BIN,
    // ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

INSTANTIATE_TEST_SUITE_P(
    Test,
    NativeFileInputConversionTFSValidPrecisionTest,
    ::testing::ValuesIn(BINARY_SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<NativeFileInputConversionTFSValidPrecisionTest::ParamType>& info) {
        return toString(info.param);
    });

static const std::vector<ovms::Precision> BINARY_UNSUPPORTED_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    // ovms::Precision::FP16,
    ovms::Precision::Q78,
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

INSTANTIATE_TEST_SUITE_P(
    Test,
    NativeFileInputConversionTFSInvalidPrecisionTest,
    ::testing::ValuesIn(BINARY_UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<NativeFileInputConversionTFSInvalidPrecisionTest::ParamType>& info) {
        return toString(info.param);
    });

class NativeFileInputConversionKFSPrecisionTest : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        readRgbJpg(filesize, image_bytes);
        inferTensorContent.mutable_contents()->add_bytes_contents(image_bytes.get(), filesize);
    }

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    ::KFSRequest::InferInputTensor inferTensorContent;
};

class NativeFileInputConversionKFSValidPrecisionTest : public NativeFileInputConversionKFSPrecisionTest {};
class NativeFileInputConversionKFSInvalidPrecisionTest : public NativeFileInputConversionKFSPrecisionTest {};

TEST_P(NativeFileInputConversionKFSValidPrecisionTest, Valid) {
    ovms::Precision testedPrecision = GetParam();

    auto tensorInfo = std::make_shared<const TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(inferTensorContent, tensor, tensorInfo, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_shape(), (ov::Shape{1, 1, 1, 3}));
    ASSERT_EQ(tensor.get_size(), 3);
    ASSERT_EQ(tensor.get_element_type(), ovmsPrecisionToIE2Precision(testedPrecision));
}

TEST_P(NativeFileInputConversionKFSInvalidPrecisionTest, Invalid) {
    ovms::Precision testedPrecision = GetParam();

    auto tensorInfo = std::make_shared<const TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(inferTensorContent, tensor, tensorInfo, nullptr), ovms::StatusCode::INVALID_PRECISION);
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    NativeFileInputConversionKFSValidPrecisionTest,
    ::testing::ValuesIn(BINARY_SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<NativeFileInputConversionTFSValidPrecisionTest::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    NativeFileInputConversionKFSInvalidPrecisionTest,
    ::testing::ValuesIn(BINARY_UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<NativeFileInputConversionTFSInvalidPrecisionTest::ParamType>& info) {
        return toString(info.param);
    });

class NativeFileInputConversionTestKFSRawInputsContents : public ::testing::Test {
public:
    ::KFSRequest::InferInputTensor requestTensor;
    std::string buffer;
    void SetUp() override {
        requestTensor.mutable_shape()->Add(1);
        requestTensor.set_datatype("BYTES");

        size_t filesize;
        std::unique_ptr<char[]> image_bytes;

        readRgbJpg(filesize, image_bytes);
        uint8_t imageSize[] = {0x9E, 0x02, 0x00, 0x00};
        buffer.append((char*)imageSize, 4);
        buffer.append(image_bytes.get(), filesize);
    }
};

TEST_F(NativeFileInputConversionTestKFSRawInputsContents, Positive) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});
    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, &this->buffer), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(NativeFileInputConversionTestKFSRawInputsContents, Positive_batchSizeBiggerThan1) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};
    this->requestTensor.mutable_shape()->Clear();
    this->requestTensor.mutable_shape()->Add(2);

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
    readRgbJpg(filesize, image_bytes);
    uint8_t imageSize[] = {0x9E, 0x02, 0x00, 0x00};
    this->buffer.append((char*)imageSize, 4);
    this->buffer.append(image_bytes.get(), filesize);

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{2, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, &this->buffer), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(NativeFileInputConversionTestKFSRawInputsContents, Negative_batchSizeBiggerThan1WithEmptyString) {
    this->requestTensor.mutable_shape()->Clear();
    this->requestTensor.mutable_shape()->Add(2);

    uint8_t imageSize[] = {0x00, 0x00, 0x00, 0x00};
    this->buffer.append((char*)imageSize, 4);

    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{2, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, &this->buffer), ovms::StatusCode::IMAGE_PARSING_FAILED);
}

TEST_F(NativeFileInputConversionTestKFSRawInputsContents, Negative_emptyString) {
    ov::Tensor tensor;

    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    std::string empty;
    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, &empty), ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(NativeFileInputConversionTestKFSRawInputsContents, Negative_invalidFormat) {
    this->requestTensor.mutable_shape()->Clear();
    this->requestTensor.mutable_shape()->Add(2);

    uint8_t imageSize[] = {0x01, 0x00, 0x00, 0x00};
    this->buffer.append((char*)imageSize, 4);

    ov::Tensor tensor;
    auto tensorInfo = std::make_shared<const TensorInfo>("", ovms::Precision::U8, ovms::Shape{2, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertNativeFileFormatRequestTensorToOVTensor(this->requestTensor, tensor, tensorInfo, &this->buffer), ovms::StatusCode::INVALID_BATCH_SIZE);
}

template <typename TensorType>
class StringInputsConversionTest : public ::testing::Test {
public:
    TensorType requestTensor;
    void SetUp() override {}

    void prepareStringTensor(tensorflow::TensorProto& tensor, std::vector<std::string> inputStrings) {
        prepareInferStringTensor(tensor, "UNUSED", inputStrings, true, nullptr);
    }
    void prepareStringTensor(::KFSRequest::InferInputTensor& tensor, std::vector<std::string> inputStrings) {
        prepareInferStringTensor(tensor, "UNUSED", inputStrings, true, nullptr);
    }

    void prepareStringTensorWithRawInputContents(tensorflow::TensorProto& tensor, std::vector<std::string> inputStrings, std::string& buffer) {
        SPDLOG_ERROR("RawInputContents not supported for TFS API");
    }
    void prepareStringTensorWithRawInputContents(::KFSRequest::InferInputTensor& tensor, std::vector<std::string> inputStrings, std::string& buffer) {
        prepareInferStringTensor(tensor, "UNUSED", inputStrings, false, &buffer);
    }
};

using MyTypes = ::testing::Types<tensorflow::TensorProto, ::KFSRequest::InferInputTensor>;
TYPED_TEST_SUITE(StringInputsConversionTest, MyTypes);

TYPED_TEST(StringInputsConversionTest, positive) {
    std::vector<std::string> expectedStrings = {"String_123"};
    this->prepareStringTensor(this->requestTensor, expectedStrings);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, nullptr), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, rawInputContents_positive) {
    if (typeid(TypeParam) == typeid(TFSInputTensorType))
        GTEST_SKIP() << "String inputs in buffer not supported for TFS api";
    std::vector<std::string> expectedStrings = {"String_123"};
    std::string rawInputContents;
    this->prepareStringTensorWithRawInputContents(this->requestTensor, expectedStrings, rawInputContents);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, &rawInputContents), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, positive_batch_size_2) {
    std::vector<std::string> expectedStrings = {"String_123", "zebra"};
    this->prepareStringTensor(this->requestTensor, expectedStrings);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, nullptr), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, rawInputContents_positive_batch_size_2) {
    if (typeid(TypeParam) == typeid(TFSInputTensorType))
        GTEST_SKIP() << "String inputs in buffer not supported for TFS api";
    std::vector<std::string> expectedStrings = {"String_123", "zebra"};
    std::string rawInputContents;
    this->prepareStringTensorWithRawInputContents(this->requestTensor, expectedStrings, rawInputContents);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, &rawInputContents), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, positive_batch_size_3_one_string_empty) {
    std::vector<std::string> expectedStrings = {"String_123", "zebra", ""};
    this->prepareStringTensor(this->requestTensor, expectedStrings);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, nullptr), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, rawInputContents_positive_batch_size_3_one_string_empty) {
    if (typeid(TypeParam) == typeid(TFSInputTensorType))
        GTEST_SKIP() << "String inputs in buffer not supported for TFS api";
    std::vector<std::string> expectedStrings = {"String_123", "zebra", ""};
    std::string rawInputContents;
    this->prepareStringTensorWithRawInputContents(this->requestTensor, expectedStrings, rawInputContents);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, &rawInputContents), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, positive_empty_inputs) {
    // This case can't happen because request validation dont allow empty strings
    std::vector<std::string> expectedStrings = {};
    this->prepareStringTensor(this->requestTensor, expectedStrings);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, nullptr), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, rawInputContents_positive_empty_inputs) {
    // This case can't happen because request validation dont allow empty strings
    if (typeid(TypeParam) == typeid(TFSInputTensorType))
        GTEST_SKIP() << "String inputs in buffer not supported for TFS api";
    std::vector<std::string> expectedStrings = {};
    std::string rawInputContents;
    this->prepareStringTensorWithRawInputContents(this->requestTensor, expectedStrings, rawInputContents);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor2D(this->requestTensor, tensor, &rawInputContents), ovms::StatusCode::OK);
    assertOutputTensorMatchExpectations(tensor, expectedStrings);
}

TYPED_TEST(StringInputsConversionTest, u8_1d) {
    std::vector<std::string> expectedStrings = {"ala", "", "ma", "kota"};
    this->prepareStringTensor(this->requestTensor, expectedStrings);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor1D(this->requestTensor, tensor, nullptr), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_element_type(), ov::element::u8);
    ASSERT_EQ(tensor.get_size(), 33);
    std::vector<uint8_t> expectedData = {
        4, 0, 0, 0,  // batch size
        0, 0, 0, 0,  // first string start offset
        3, 0, 0, 0,  // end of "ala" in condensed content
        3, 0, 0, 0,  // end of "" in condensed content
        5, 0, 0, 0,  // end of "ma" in condensed content
        9, 0, 0, 0,  // end of "kota" in condensed content
        'a', 'l', 'a',
        'm', 'a',
        'k', 'o', 't', 'a'};
    ASSERT_EQ(std::memcmp(reinterpret_cast<uint8_t*>(tensor.data()), expectedData.data(), expectedData.size()), 0)
        << readableError(reinterpret_cast<uint8_t*>(tensor.data()), expectedData.data(), expectedData.size());
}

TYPED_TEST(StringInputsConversionTest, rawInputContents_u8_1d) {
    if (typeid(TypeParam) == typeid(TFSInputTensorType))
        GTEST_SKIP() << "String inputs in buffer not supported for TFS api";
    std::vector<std::string> expectedStrings = {"ala", "", "ma", "kota"};
    std::string rawInputContents;
    this->prepareStringTensorWithRawInputContents(this->requestTensor, expectedStrings, rawInputContents);
    ov::Tensor tensor;
    ASSERT_EQ(convertStringRequestToOVTensor1D(this->requestTensor, tensor, &rawInputContents), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_element_type(), ov::element::u8);
    ASSERT_EQ(tensor.get_size(), 33);
    std::vector<uint8_t> expectedData = {
        4, 0, 0, 0,  // batch size
        0, 0, 0, 0,  // first string start offset
        3, 0, 0, 0,  // end of "ala" in condensed content
        3, 0, 0, 0,  // end of "" in condensed content
        5, 0, 0, 0,  // end of "ma" in condensed content
        9, 0, 0, 0,  // end of "kota" in condensed content
        'a', 'l', 'a',
        'm', 'a',
        'k', 'o', 't', 'a'};
    ASSERT_EQ(std::memcmp(reinterpret_cast<uint8_t*>(tensor.data()), expectedData.data(), expectedData.size()), 0)
        << readableError(reinterpret_cast<uint8_t*>(tensor.data()), expectedData.data(), expectedData.size());
}

template <typename TensorType>
class StringOutputsConversionTest : public ::testing::Test {
public:
    TensorType responseTensor;
    void SetUp() override {}
};

using OutputTensorProtos = ::testing::Types<tensorflow::TensorProto, KFSTensorOutputProto>;
TYPED_TEST_SUITE(StringOutputsConversionTest, OutputTensorProtos);

TYPED_TEST(StringOutputsConversionTest, positive) {
    std::vector<std::uint8_t> _2dTensorData = {
        'S', 't', 'r', 'i', 'n', 'g', '_', '1', '2', '3', 0,  // String_123
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                      // ""
        'z', 'e', 'b', 'r', 'a', 0, 0, 0, 0, 0, 0};           // "zebra"
    ov::Shape _2dTensorShape = {3, 11};
    ov::Tensor tensor(ov::element::u8, _2dTensorShape, _2dTensorData.data());
    ASSERT_EQ(convertOVTensor2DToStringResponse(tensor, this->responseTensor), ovms::StatusCode::OK);
    assertStringOutputProto(this->responseTensor, {"String_123", "", "zebra"});
}

}  // namespace
