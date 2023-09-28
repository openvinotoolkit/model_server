//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pybind11/pybind11.h>

#include "../../ovms_py_tensor.hpp"

using namespace ovms;

namespace py = pybind11;

TEST(OvmsPyTensor, BuildKnownFormatMultiDimShape) {
    // OvmsPyTensor(void *ptr, std::vector<py::ssize_t> shape, std::string datatype, size_t size);
    size_t INPUT_BUFFER_SIZE = 1 * 3 * 300 * 300 * sizeof(float);
    std::string data(INPUT_BUFFER_SIZE, '1');
    void* ptr = data.data();
    std::vector<py::ssize_t> shape{1, 3, 300, 300};
    std::string datatype = "FP32";
    OvmsPyTensor ovmsPyTensor("input", ptr, shape, datatype, INPUT_BUFFER_SIZE);

    std::vector<py::ssize_t> expectedStrides{1080000, 360000, 1200, 4};
    std::string expectedFormat = datatypeToBufferFormat.at(datatype);
    size_t expectedItemsize = bufferFormatToItemsize.at(expectedFormat);

    EXPECT_EQ(ovmsPyTensor.name, "input");
    EXPECT_EQ(ovmsPyTensor.ptr, ptr);
    EXPECT_EQ(ovmsPyTensor.shape, shape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);
}

TEST(OvmsPyTensor, BuildKnownFormatSingleDimShape) {
    size_t INPUT_BUFFER_SIZE = 1 * 3 * 300 * 300 * sizeof(float);
    std::string data(INPUT_BUFFER_SIZE, '1');
    void* ptr = data.data();
    std::vector<py::ssize_t> shape{1 * 3 * 300 * 300};
    std::string datatype = "FP32";
    OvmsPyTensor ovmsPyTensor("input", ptr, shape, datatype, INPUT_BUFFER_SIZE);

    std::vector<py::ssize_t> expectedStrides{sizeof(float)};
    std::string expectedFormat = datatypeToBufferFormat.at(datatype);
    size_t expectedItemsize = bufferFormatToItemsize.at(expectedFormat);

    EXPECT_EQ(ovmsPyTensor.name, "input");
    EXPECT_EQ(ovmsPyTensor.ptr, ptr);
    EXPECT_EQ(ovmsPyTensor.shape, shape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);
}

TEST(OvmsPyTensor, BuildUnknownFormatSingleDimShape) {
    size_t INPUT_BUFFER_SIZE = 3 * 1024;
    std::string data(INPUT_BUFFER_SIZE, '1');
    void* ptr = data.data();
    std::vector<py::ssize_t> shape{3 * 1024};
    std::string datatype = "my_string_type";
    OvmsPyTensor ovmsPyTensor("input", ptr, shape, datatype, INPUT_BUFFER_SIZE);

    std::vector<py::ssize_t> expectedStrides{1};
    std::string expectedFormat = datatypeToBufferFormat.at("UINT8");
    size_t expectedItemsize = bufferFormatToItemsize.at(expectedFormat);

    EXPECT_EQ(ovmsPyTensor.name, "input");
    EXPECT_EQ(ovmsPyTensor.ptr, ptr);
    EXPECT_EQ(ovmsPyTensor.shape, shape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);
}

TEST(OvmsPyTensor, BuildUnknownFormatMultiDimShape) {
    size_t INPUT_BUFFER_SIZE = 3 * 1024;
    std::string data(INPUT_BUFFER_SIZE, '1');
    void* ptr = data.data();
    std::vector<py::ssize_t> shape{3, 1024};
    std::string datatype = "my_string_type";
    OvmsPyTensor ovmsPyTensor("input", ptr, shape, datatype, INPUT_BUFFER_SIZE);

    std::vector<py::ssize_t> expectedStrides{1024, 1};
    std::string expectedFormat = datatypeToBufferFormat.at("UINT8");
    size_t expectedItemsize = bufferFormatToItemsize.at(expectedFormat);

    EXPECT_EQ(ovmsPyTensor.name, "input");
    EXPECT_EQ(ovmsPyTensor.ptr, ptr);
    EXPECT_EQ(ovmsPyTensor.shape, shape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);
}
