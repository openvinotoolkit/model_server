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
    InferenceEngine::Blob::Ptr blob;
    auto tensorInfo = std::make_shared<TensorInfo>();
    tensorInfo->setShape({5, 1, 1, 1});
    tensorInfo->setLayout(InferenceEngine::Layout::NHWC);

    EXPECT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(BinaryUtilsTest, tensorWithInvalidImage) {
    tensorflow::TensorProto stringValInvalidImage;
    InferenceEngine::Blob::Ptr blob;
    stringValInvalidImage.set_dtype(tensorflow::DataType::DT_STRING);
    std::string invalidImage = "INVALID_IMAGE";
    stringValInvalidImage.add_string_val(invalidImage);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NHWC);

    EXPECT_EQ(convertStringValToBlob(stringValInvalidImage, blob, tensorInfo, false), ovms::StatusCode::IMAGE_PARSING_FAILED);
}

TEST_F(BinaryUtilsTest, tensorWithEmptyStringVal) {
    tensorflow::TensorProto stringValEmptyImage;
    InferenceEngine::Blob::Ptr blob;
    stringValEmptyImage.set_dtype(tensorflow::DataType::DT_STRING);
    std::string invalidImage = "";
    stringValEmptyImage.add_string_val(invalidImage);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NHWC);

    EXPECT_EQ(convertStringValToBlob(stringValEmptyImage, blob, tensorInfo, false), ovms::StatusCode::STRING_VAL_EMPTY);
}

TEST_F(BinaryUtilsTest, tensorWithNonSupportedLayout) {
    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NCHW);

    EXPECT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTest, tensorWithNonSupportedPrecision) {
    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::MIXED, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NHWC);

    EXPECT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::INVALID_PRECISION);
}

TEST_F(BinaryUtilsTest, tensorWithNonMatchingShapeSize) {
    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1}, InferenceEngine::Layout::NC);

    EXPECT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

TEST_F(BinaryUtilsTest, tensorWithNonMatchingNumberOfChannelsNHWC) {
    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1, 1, 1}, InferenceEngine::Layout::NHWC);

    EXPECT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::INVALID_NO_OF_CHANNELS);
}

TEST_F(BinaryUtilsTest, positive_rgb) {
    uint8_t rgb_expected_blob[] = {0x24, 0x1b, 0xed};

    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NHWC);

    ASSERT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(blob->size(), 3);
    uint8_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();
    EXPECT_EQ(std::equal(ptr, ptr + blob->size(), rgb_expected_blob), true);
}

TEST_F(BinaryUtilsTest, positive_grayscale) {
    uint8_t grayscale_expected_blob[] = {0x00};

    std::ifstream DataFile;
    DataFile.open("/ovms/src/test/binaryutils/grayscale.jpg", std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    size_t grayscale_filesize = DataFile.tellg();
    DataFile.seekg(0);
    std::unique_ptr<char[]> grayscale_image_bytes(new char[grayscale_filesize]);
    DataFile.read(grayscale_image_bytes.get(), grayscale_filesize);

    tensorflow::TensorProto grayscaleStringVal;
    InferenceEngine::Blob::Ptr blob;
    grayscaleStringVal.set_dtype(tensorflow::DataType::DT_STRING);
    grayscaleStringVal.add_string_val(grayscale_image_bytes.get(), grayscale_filesize);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1, 1, 1}, InferenceEngine::Layout::NHWC);

    ASSERT_EQ(convertStringValToBlob(grayscaleStringVal, blob, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(blob->size(), 1);
    uint8_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();
    EXPECT_EQ(std::equal(ptr, ptr + blob->size(), grayscale_expected_blob), true);
}

TEST_F(BinaryUtilsTest, positive_batch_size_2) {
    uint8_t rgb_batchsize_2_blob[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    InferenceEngine::Blob::Ptr blob;
    stringVal.add_string_val(image_bytes.get(), filesize);

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{2, 1, 1, 3}, InferenceEngine::Layout::NHWC);

    ASSERT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(blob->size(), 6);
    uint8_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();
    EXPECT_EQ(std::equal(ptr, ptr + blob->size(), rgb_batchsize_2_blob), true);
}

TEST_F(BinaryUtilsTest, positive_precision_changed) {
    uint8_t rgb_precision_changed_expected_blob[] = {0x24, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0xed, 0x00, 0x00, 0x00};

    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::I32, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NHWC);

    ASSERT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(blob->size(), 3);
    uint8_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();
    int I32_size = 4;
    EXPECT_EQ(std::equal(ptr, ptr + blob->size() * I32_size, rgb_precision_changed_expected_blob), true);
}

TEST_F(BinaryUtilsTest, positive_nhwc_layout) {
    uint8_t rgb_expected_blob[] = {0x24, 0x1b, 0xed};

    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 1, 1, 3}, InferenceEngine::Layout::NHWC);

    ASSERT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(blob->size(), 3);
    uint8_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();

    EXPECT_EQ(std::equal(ptr, ptr + blob->size(), rgb_expected_blob), true);
}

TEST_F(BinaryUtilsTest, positive_resizing) {
    uint8_t rgb_expected_blob[] = {0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed, 0x24, 0x1b, 0xed};

    InferenceEngine::Blob::Ptr blob;

    std::shared_ptr<TensorInfo> tensorInfo = std::make_shared<TensorInfo>("", InferenceEngine::Precision::U8, shape_t{1, 2, 2, 3}, InferenceEngine::Layout::NHWC);

    ASSERT_EQ(convertStringValToBlob(stringVal, blob, tensorInfo, false), ovms::StatusCode::OK);
    ASSERT_EQ(blob->size(), 12);
    uint8_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();
    EXPECT_EQ(std::equal(ptr, ptr + blob->size(), rgb_expected_blob), true);
}
}  // namespace
