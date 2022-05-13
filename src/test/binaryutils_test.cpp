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

#include "../binaryutils.hpp"
#include "opencv2/opencv.hpp"
#include "test_utils.hpp"

using namespace ovms;

namespace {

class BinaryUtilsTest : public ::testing::Test {
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

TEST_F(BinaryUtilsTest, tensorWithNonMatchingBatchsize) {
    tensorflow::TensorProto stringValDummy;
    stringValDummy.add_string_val("dummy");
    ov::Tensor tensor;
    auto tensorInfo = std::make_shared<TensorInfo>();
    tensorInfo->setShape({5, 1, 1, 1});
    tensorInfo->setLayout(Layout{"NHWC"});
    auto status = convertStringValToTensor(stringVal, tensor, tensorInfo);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(BinaryUtilsTest, tensorWithInvalidImage) {
    tensorflow::TensorProto stringValInvalidImage;
    ov::Tensor tensor;
    stringValInvalidImage.set_dtype(tensorflow::DataType::DT_STRING);
    std::string invalidImage = "INVALID_IMAGE";
    stringValInvalidImage.add_string_val(invalidImage);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringValInvalidImage, tensor, tensorInfo), ovms::StatusCode::IMAGE_PARSING_FAILED);
}

TEST_F(BinaryUtilsTest, tensorWithEmptyStringVal) {
    tensorflow::TensorProto stringValEmptyImage;
    ov::Tensor tensor;
    stringValEmptyImage.set_dtype(tensorflow::DataType::DT_STRING);
    std::string invalidImage = "";
    stringValEmptyImage.add_string_val(invalidImage);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringValEmptyImage, tensor, tensorInfo), ovms::StatusCode::STRING_VAL_EMPTY);
}

TEST_F(BinaryUtilsTest, tensorWithNonSupportedLayout) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NCHW"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTest, tensorWithNonSupportedPrecision) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::MIXED, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(BinaryUtilsTest, tensorWithNonMatchingShapeSize) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1}, Layout{"NC"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTest, tensorWithNonMatchingNumberOfChannelsNHWC) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, Layout{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::INVALID_NO_OF_CHANNELS);
}

TEST_F(BinaryUtilsTest, positive_rgb) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_grayscale) {
    uint8_t grayscale_expected_tensor[] = {0x00};

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/grayscale.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t grayscale_filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> grayscale_image_bytes(new char[grayscale_filesize]);
    DataFile.read(grayscale_image_bytes.get(), grayscale_filesize);

    tensorflow::TensorProto grayscaleStringVal;
    ov::Tensor tensor;
    grayscaleStringVal.set_dtype(tensorflow::DataType::DT_STRING);
    grayscaleStringVal.add_string_val(grayscale_image_bytes.get(), grayscale_filesize);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(grayscaleStringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 1);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), grayscale_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_batch_size_2) {
    uint8_t rgb_batchsize_2_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;
    stringVal.add_string_val(image_bytes.get(), filesize);

    for (const auto layout : std::vector<Layout>{Layout("NHWC"), Layout::getDefaultLayout()}) {
        std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{2, 1, 1, 3}, layout);

        ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
        ASSERT_EQ(tensor.get_size(), 6);
        uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
        EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_batchsize_2_tensor), true);
    }
}

TEST_F(BinaryUtilsTest, positive_precision_changed) {
    uint8_t rgb_precision_changed_expected_tensor[] = {0x24, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0xed, 0x00, 0x00, 0x00};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::I32, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    int I32_size = 4;
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size() * I32_size, rgb_precision_changed_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_nhwc_layout) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());

    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, layout_default_resolution_mismatch) {
    ov::Tensor tensor;
    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 3, 1, 3}, Layout::getDefaultLayout());
    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(BinaryUtilsTest, positive_resizing) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 2, 2, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 12);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_cols_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {2, 5}, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 2;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_cols_bigger) {
    uint8_t rgb_expected_tensor[] = {0x96, 0x8f, 0xf3, 0x98, 0x9a, 0x81, 0x9d, 0xa9, 0x12};

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/rgb4x4.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> image_bytes(new char[filesize]);
    DataFile.read(image_bytes.get(), filesize);

    tensorflow::TensorProto stringVal2x2;
    stringVal2x2.set_dtype(tensorflow::DataType::DT_STRING);
    stringVal2x2.add_string_val(image_bytes.get(), filesize);

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal2x2, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 3;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_cols_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_rows_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {2, 5}, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 2;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_rows_bigger) {
    uint8_t rgb_expected_tensor[] = {0x3f, 0x65, 0x88, 0x98, 0x9a, 0x81, 0xf5, 0xd2, 0x7c};

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/rgb4x4.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> image_bytes(new char[filesize]);
    DataFile.read(image_bytes.get(), filesize);

    tensorflow::TensorProto stringVal2x2;
    stringVal2x2.set_dtype(tensorflow::DataType::DT_STRING);
    stringVal2x2.add_string_val(image_bytes.get(), filesize);

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal2x2, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 3;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_rows_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_any_shape) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), ovms::Dimension::any(), 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, negative_resizing_with_one_any_one_static_shape) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), 4, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_one_any_one_static_shape) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_demultiplexer_and_range_resolution) {
    ov::Tensor tensor;

    const int batchSize = 5;
    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, {1, 3}, 3}, Layout{"NHWC"});
    tensorInfo = tensorInfo->createCopyWithDemultiplexerDimensionPrefix(batchSize);

    stringVal.Clear();
    read4x4RgbJpg(filesize, image_bytes);

    stringVal.set_dtype(tensorflow::DataType::DT_STRING);
    for (int i = 0; i < batchSize; i++) {
        stringVal.add_string_val(image_bytes.get(), filesize);
    }
    stringVal.mutable_tensor_shape()->add_dim()->set_size(batchSize);

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    ASSERT_EQ(tensorDims[0], batchSize);
    EXPECT_EQ(tensorDims[1], 1);
    EXPECT_EQ(tensorDims[2], 3);
    EXPECT_EQ(tensorDims[3], 3);
    EXPECT_EQ(tensorDims[4], 3);
    ASSERT_EQ(tensor.get_size(), batchSize * 1 * 3 * 3 * 3);
}

TEST_F(BinaryUtilsTest, positive_range_resolution_matching_in_between) {
    ov::Tensor tensor;

    const int batchSize = 5;
    for (const auto& batchDim : std::vector<Dimension>{Dimension::any(), Dimension(batchSize)}) {
        std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{batchDim, {1, 5}, {1, 5}, 3}, Layout{"NHWC"});

        stringVal.Clear();
        read4x4RgbJpg(filesize, image_bytes);

        stringVal.set_dtype(tensorflow::DataType::DT_STRING);
        for (int i = 0; i < batchSize; i++) {
            stringVal.add_string_val(image_bytes.get(), filesize);
        }
        stringVal.mutable_tensor_shape()->add_dim()->set_size(batchSize);

        ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
        shape_t tensorDims = tensor.get_shape();
        ASSERT_EQ(tensorDims[0], batchSize);
        EXPECT_EQ(tensorDims[1], 4);
        EXPECT_EQ(tensorDims[2], 4);
        EXPECT_EQ(tensorDims[3], 3);
        ASSERT_EQ(tensor.get_size(), batchSize * 4 * 4 * 3);
    }
}


class BinaryUtilsTestKFS : public ::testing::Test {
protected:
    void SetUp() override {
        readRgbJpg(filesize, image_bytes);
    }

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
};

// TEST_F(BinaryUtilsTestKFS, tensorWithNonMatchingBatchsize) {
//     tensorflow::TensorProto stringValDummy;
//     stringValDummy.add_string_val("dummy");
//     ov::Tensor tensor;
//     auto tensorInfo = std::make_shared<TensorInfo>();
//     tensorInfo->setShape({5, 1, 1, 1});
//     tensorInfo->setLayout(Layout{"NHWC"});
//     auto status = convertStringValToTensor(stringVal, tensor, tensorInfo);
//     EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
// }

TEST_F(BinaryUtilsTestKFS, tensorWithInvalidImage) {
    ov::Tensor tensor;
    std::string invalidImage = "INVALID_IMAGE";

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});
    EXPECT_EQ(convertStringToTensor(invalidImage, tensor, tensorInfo), ovms::StatusCode::IMAGE_PARSING_FAILED);
}

TEST_F(BinaryUtilsTestKFS, tensorWithEmptyStringVal) {
    ov::Tensor tensor;
    std::string invalidImage = "";

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    EXPECT_EQ(convertStringToTensor(invalidImage, tensor, tensorInfo), ovms::StatusCode::BYTES_CONTENTS_EMPTY);
}

TEST_F(BinaryUtilsTestKFS, tensorWithNonSupportedLayout) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NCHW"});
    std::string image(image_bytes.get(), filesize);
    EXPECT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTestKFS, tensorWithNonSupportedPrecision) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::MIXED, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    EXPECT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(BinaryUtilsTestKFS, tensorWithNonMatchingShapeSize) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1}, Layout{"NC"});
    std::string image(image_bytes.get(), filesize);
    EXPECT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTestKFS, tensorWithNonMatchingNumberOfChannelsNHWC) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    EXPECT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::INVALID_NO_OF_CHANNELS);
}

TEST_F(BinaryUtilsTestKFS, positive_rgb) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_grayscale) {
    uint8_t grayscale_expected_tensor[] = {0x00};

    ov::Tensor tensor;

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/grayscale.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t grayscale_filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> grayscale_image_bytes(new char[grayscale_filesize]);
    DataFile.read(grayscale_image_bytes.get(), grayscale_filesize);

    std::string image(grayscale_image_bytes.get(), filesize);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, Layout{"NHWC"});

    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 1);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), grayscale_expected_tensor), true);
}

// TEST_F(BinaryUtilsTestKFS, positive_batch_size_2) {
//     uint8_t rgb_batchsize_2_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

//     ov::Tensor tensor;
//     stringVal.add_string_val(image_bytes.get(), filesize);

//     for (const auto layout : std::vector<Layout>{Layout("NHWC"), Layout::getDefaultLayout()}) {
//         std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{2, 1, 1, 3}, layout);

//         ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
//         ASSERT_EQ(tensor.get_size(), 6);
//         uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
//         EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_batchsize_2_tensor), true);
//     }
// }

TEST_F(BinaryUtilsTestKFS, positive_precision_changed) {
    uint8_t rgb_precision_changed_expected_tensor[] = {0x24, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0xed, 0x00, 0x00, 0x00};

    ov::Tensor tensor;

    std::string image(image_bytes.get(), filesize);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::I32, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});

    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    int I32_size = 4;
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size() * I32_size, rgb_precision_changed_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_nhwc_layout) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());

    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, layout_default_resolution_mismatch) {
    ov::Tensor tensor;
    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 3, 1, 3}, Layout::getDefaultLayout());
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 2, 2, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 12);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_dynamic_shape_cols_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {2, 5}, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 2;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_dynamic_shape_cols_bigger) {
    uint8_t rgb_expected_tensor[] = {0x96, 0x8f, 0xf3, 0x98, 0x9a, 0x81, 0x9d, 0xa9, 0x12};

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/rgb4x4.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> image_bytes(new char[filesize]);
    DataFile.read(image_bytes.get(), filesize);

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 3;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_dynamic_shape_cols_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_dynamic_shape_rows_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {2, 5}, 1, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 2;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_dynamic_shape_rows_bigger) {
    uint8_t rgb_expected_tensor[] = {0x3f, 0x65, 0x88, 0x98, 0x9a, 0x81, 0xf5, 0xd2, 0x7c};

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/rgb4x4.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> image_bytes(new char[filesize]);
    DataFile.read(image_bytes.get(), filesize);

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 3;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_dynamic_shape_rows_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_any_shape) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), ovms::Dimension::any(), 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTestKFS, negative_resizing_with_one_any_one_static_shape) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), 4, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::INVALID_SHAPE);
}

TEST_F(BinaryUtilsTestKFS, positive_resizing_with_one_any_one_static_shape) {
    ov::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), 1, 3}, Layout{"NHWC"});
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
}

// TEST_F(BinaryUtilsTestKFS, positive_resizing_with_demultiplexer_and_range_resolution) {
//     ov::Tensor tensor;

//     const int batchSize = 5;
//     std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, {1, 3}, 3}, Layout{"NHWC"});
//     tensorInfo = tensorInfo->createCopyWithDemultiplexerDimensionPrefix(batchSize);

//     stringVal.Clear();
//     read4x4RgbJpg(filesize, image_bytes);

//     stringVal.set_dtype(tensorflow::DataType::DT_STRING);
//     for (int i = 0; i < batchSize; i++) {
//         stringVal.add_string_val(image_bytes.get(), filesize);
//     }
//     stringVal.mutable_tensor_shape()->add_dim()->set_size(batchSize);

//     ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
//     shape_t tensorDims = tensor.get_shape();
//     ASSERT_EQ(tensorDims[0], batchSize);
//     EXPECT_EQ(tensorDims[1], 1);
//     EXPECT_EQ(tensorDims[2], 3);
//     EXPECT_EQ(tensorDims[3], 3);
//     EXPECT_EQ(tensorDims[4], 3);
//     ASSERT_EQ(tensor.get_size(), batchSize * 1 * 3 * 3 * 3);
// }

// TEST_F(BinaryUtilsTestKFS, positive_range_resolution_matching_in_between) {
//     ov::Tensor tensor;

//     const int batchSize = 5;
//     for (const auto& batchDim : std::vector<Dimension>{Dimension::any(), Dimension(batchSize)}) {
//         std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{batchDim, {1, 5}, {1, 5}, 3}, Layout{"NHWC"});

//         stringVal.Clear();
//         read4x4RgbJpg(filesize, image_bytes);

//         stringVal.set_dtype(tensorflow::DataType::DT_STRING);
//         for (int i = 0; i < batchSize; i++) {
//             stringVal.add_string_val(image_bytes.get(), filesize);
//         }
//         stringVal.mutable_tensor_shape()->add_dim()->set_size(batchSize);

//         ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
//         shape_t tensorDims = tensor.get_shape();
//         ASSERT_EQ(tensorDims[0], batchSize);
//         EXPECT_EQ(tensorDims[1], 4);
//         EXPECT_EQ(tensorDims[2], 4);
//         EXPECT_EQ(tensorDims[3], 3);
//         ASSERT_EQ(tensor.get_size(), batchSize * 4 * 4 * 3);
//     }
// }

class BinaryUtilsPrecisionTest : public ::testing::TestWithParam<ovms::Precision> {
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

class BinaryUtilsValidPrecisionTest : public BinaryUtilsPrecisionTest {};
class BinaryUtilsInvalidPrecisionTest : public BinaryUtilsPrecisionTest {};

TEST_P(BinaryUtilsValidPrecisionTest, Valid) {
    ovms::Precision testedPrecision = GetParam();

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_shape(), (ov::Shape{1, 1, 1, 3}));
    ASSERT_EQ(tensor.get_size(), 3);
    ASSERT_EQ(tensor.get_element_type(), ovmsPrecisionToIE2Precision(testedPrecision));
}

TEST_P(BinaryUtilsInvalidPrecisionTest, Invalid) {
    ovms::Precision testedPrecision = GetParam();

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo), ovms::StatusCode::INVALID_PRECISION);
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
    BinaryUtilsValidPrecisionTest,
    ::testing::ValuesIn(BINARY_SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<BinaryUtilsValidPrecisionTest::ParamType>& info) {
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
    BinaryUtilsInvalidPrecisionTest,
    ::testing::ValuesIn(BINARY_UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<BinaryUtilsInvalidPrecisionTest::ParamType>& info) {
        return toString(info.param);
    });

class BinaryUtilsKFSPrecisionTest : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        readRgbJpg(filesize, image_bytes);
    }

    size_t filesize;
    std::unique_ptr<char[]> image_bytes;
};

class BinaryUtilsKFSValidPrecisionTest : public BinaryUtilsKFSPrecisionTest {};
class BinaryUtilsKFSInvalidPrecisionTest : public BinaryUtilsKFSPrecisionTest {};

TEST_P(BinaryUtilsKFSValidPrecisionTest, Valid) {
    ovms::Precision testedPrecision = GetParam();

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_shape(), (ov::Shape{1, 1, 1, 3}));
    ASSERT_EQ(tensor.get_size(), 3);
    ASSERT_EQ(tensor.get_element_type(), ovmsPrecisionToIE2Precision(testedPrecision));
}

TEST_P(BinaryUtilsKFSInvalidPrecisionTest, Invalid) {
    ovms::Precision testedPrecision = GetParam();

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("",
        testedPrecision,
        ovms::Shape{1, 1, 1, 3},
        Layout{"NHWC"});

    ov::runtime::Tensor tensor;
    std::string image(image_bytes.get(), filesize);
    ASSERT_EQ(convertStringToTensor(image, tensor, tensorInfo), ovms::StatusCode::INVALID_PRECISION);
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    BinaryUtilsKFSValidPrecisionTest,
    ::testing::ValuesIn(BINARY_SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<BinaryUtilsValidPrecisionTest::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    BinaryUtilsKFSInvalidPrecisionTest,
    ::testing::ValuesIn(BINARY_UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<BinaryUtilsInvalidPrecisionTest::ParamType>& info) {
        return toString(info.param);
    });

}  // namespace
