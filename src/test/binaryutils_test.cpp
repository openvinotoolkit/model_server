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

#include "gtest/gtest.h"
#include "../binaryutils.hpp"

using namespace ovms;

namespace {
    class BinaryUtilsTest : public ::testing::Test {};
    TEST_F(BinaryUtilsTest, tensorWithNoStringVal) {
        tensorflow::TensorProto tensor;
        auto status = convertBinaryStringValToTensorContent(tensor);
        EXPECT_EQ(status, ovms::StatusCode::OK);
    }

    TEST_F(BinaryUtilsTest, positive) {
        std::ifstream DataFile("/ovms/example_client/images/bee.jpeg", std::ios::binary);
        DataFile.seekg(0, std::ios::end);
        size_t filesize = (int)DataFile.tellg();
        DataFile.seekg(0);

        char* image_bytes = new char[filesize];
        DataFile.read((char*)image_bytes, filesize);

        tensorflow::TensorProto tensor;
        tensor.set_dtype(tensorflow::DataType::DT_STRING);
        tensor.add_string_val(image_bytes, filesize);
        tensor.mutable_tensor_shape()->add_dim()->set_size(1);

        auto status = convertBinaryStringValToTensorContent(tensor);
        delete [] image_bytes;
        EXPECT_EQ(status, ovms::StatusCode::OK);
    }
}