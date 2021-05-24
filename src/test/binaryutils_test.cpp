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

using namespace ovms;

namespace {
class BinaryUtilsTest : public ::testing::Test {};
//convertStringValToBlob(const tensorflow::TensorProto& src, InferenceEngine::Blob::Ptr* blob, const std::shared_ptr<TensorInfo>& tensorInfo)

TEST_F(BinaryUtilsTest, tensorWithNonMatchingBatchsize) {
    tensorflow::TensorProto stringVal;
    stringVal.add_string_val("dummy");
    InferenceEngine::Blob::Ptr blob;
    auto tensorInfo = std::make_shared<TensorInfo>();
    tensorInfo->setShape({5,1,1,1});
    auto status = convertStringValToBlob(stringVal, &blob, tensorInfo);
    EXPECT_EQ(status, ovms::StatusCode::UNSUPPORTED_LAYOUT);
}

// TEST_F(BinaryUtilsTest, positive) {
//     std::ifstream DataFile("/ovms/example_client/images/bee.jpeg", std::ios::binary);
//     DataFile.seekg(0, std::ios::end);
//     size_t filesize = (int)DataFile.tellg();
//     DataFile.seekg(0);

//     char* image_bytes = new char[filesize];
//     DataFile.read((char*)image_bytes, filesize);

//     tensorflow::TensorProto stringVal;
//     tensorflow::TensorProto tensorContent;
//     stringVal.set_dtype(tensorflow::DataType::DT_STRING);
//     stringVal.add_string_val(image_bytes, filesize);
//     stringVal.mutable_tensor_shape()->add_dim()->set_size(1);

//     auto tensorInfo = std::make_shared<TensorInfo>();
//     auto status = convertStringValToTensorContent(stringVal, tensorContent, tensorInfo);
//     delete [] image_bytes;
//     EXPECT_EQ(status, ovms::StatusCode::OK);
// }
}  // namespace