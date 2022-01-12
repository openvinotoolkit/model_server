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

#include "../binaryutils.hpp"
#include "gtest/gtest.h"
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
    ov::runtime::Tensor tensor;
    auto tensorInfo = std::make_shared<TensorInfo>();
    tensorInfo->setShape({5, 1, 1, 1});
    tensorInfo->setLayout(layout_t{"NHWC"});
    auto status = convertStringValToTensor(stringVal, tensor, tensorInfo, false);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(BinaryUtilsTest, tensorWithInvalidImage) {
    tensorflow::TensorProto stringValInvalidImage;
    ov::runtime::Tensor tensor;
    stringValInvalidImage.set_dtype(tensorflow::DataType::DT_STRING);
    std::string invalidImage = "INVALID_IMAGE";
    stringValInvalidImage.add_string_val(invalidImage);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, layout_t{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringValInvalidImage, tensor, tensorInfo, false), ovms::StatusCode::IMAGE_PARSING_FAILED);
}

TEST_F(BinaryUtilsTest, tensorWithEmptyStringVal) {
    tensorflow::TensorProto stringValEmptyImage;
    ov::runtime::Tensor tensor;
    stringValEmptyImage.set_dtype(tensorflow::DataType::DT_STRING);
    std::string invalidImage = "";
    stringValEmptyImage.add_string_val(invalidImage);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, layout_t{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringValEmptyImage, tensor, tensorInfo, false), ovms::StatusCode::STRING_VAL_EMPTY);
}

TEST_F(BinaryUtilsTest, tensorWithNonSupportedLayout) {
    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, layout_t{"NCHW"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTest, tensorWithNonSupportedPrecision) {
    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::MIXED, ovms::Shape{1, 1, 1, 3}, layout_t{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(BinaryUtilsTest, tensorWithNonMatchingShapeSize) {
    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1}, layout_t{"NC"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTest, tensorWithNonMatchingNumberOfChannelsNHWC) {
    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, layout_t{"NHWC"});

    EXPECT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::INVALID_NO_OF_CHANNELS);
}

TEST_F(BinaryUtilsTest, positive_rgb) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
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
    ov::runtime::Tensor tensor;
    grayscaleStringVal.set_dtype(tensorflow::DataType::DT_STRING);
    grayscaleStringVal.add_string_val(grayscale_image_bytes.get(), grayscale_filesize);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 1}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(grayscaleStringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 1);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), grayscale_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_batch_size_2) {
    uint8_t rgb_batchsize_2_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;
    stringVal.add_string_val(image_bytes.get(), filesize);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{2, 1, 1, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 6);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_batchsize_2_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_precision_changed) {
    uint8_t rgb_precision_changed_expected_tensor[] = {0x24, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0xed, 0x00, 0x00, 0x00};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::I32, ovms::Shape{1, 1, 1, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    int I32_size = 4;
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size() * I32_size, rgb_precision_changed_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_nhwc_layout) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, 1, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());

    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 2, 2, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(tensor.get_size(), 12);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_cols_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {2, 5}, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
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

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal2x2, tensor, tensorInfo, false), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 3;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_cols_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, 1, {1, 3}, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_rows_smaller) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {2, 5}, 1, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
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

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal2x2, tensor, tensorInfo, false), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 3;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 9);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_dynamic_shape_rows_in_range) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, {1, 3}, 1, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_any_shape) {
    uint8_t rgb_expected_tensor[] = {0x24, 0x1b, 0xed};

    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), ovms::Dimension::any(), 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 1;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 3);
    uint8_t* ptr = static_cast<uint8_t*>(tensor.data());
    EXPECT_EQ(std::equal(ptr, ptr + tensor.get_size(), rgb_expected_tensor), true);
}

TEST_F(BinaryUtilsTest, positive_resizing_with_one_any_one_static_shape) {
    ov::runtime::Tensor tensor;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", ovms::Precision::U8, ovms::Shape{1, ovms::Dimension::any(), 4, 3}, layout_t{"NHWC"});

    ASSERT_EQ(convertStringValToTensor(stringVal, tensor, tensorInfo, false), ovms::StatusCode::OK);
    shape_t tensorDims = tensor.get_shape();
    size_t expectedRowsNumber = 1;
    EXPECT_EQ(tensorDims[1], expectedRowsNumber);
    size_t expectedColsNumber = 4;
    EXPECT_EQ(tensorDims[2], expectedColsNumber);
    ASSERT_EQ(tensor.get_size(), 12);
}
}  // namespace
