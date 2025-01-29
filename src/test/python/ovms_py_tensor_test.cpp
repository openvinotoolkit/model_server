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
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/pybind11.h>
#pragma warning(pop)

#include "bindings/python/ovms_py_tensor.hpp"

using namespace ovms;

namespace py = pybind11;

TEST(OvmsPyTensor, BuildKnownFormatMultiDimShape) {
    py::ssize_t INPUT_BUFFER_SIZE = 1 * 3 * 300 * 300 * sizeof(float);
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
    EXPECT_EQ(ovmsPyTensor.userShape, shape);
    EXPECT_EQ(ovmsPyTensor.bufferShape, shape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);

    OvmsPyTensor recreatedTensor("input", py::buffer_info{
                                              ovmsPyTensor.ptr,
                                              ovmsPyTensor.itemsize,
                                              ovmsPyTensor.format,
                                              ovmsPyTensor.ndim,
                                              ovmsPyTensor.bufferShape,
                                              ovmsPyTensor.strides});

    EXPECT_EQ(recreatedTensor.name, "input");
    EXPECT_EQ(recreatedTensor.ptr, ovmsPyTensor.ptr);
    EXPECT_EQ(recreatedTensor.userShape, ovmsPyTensor.userShape);
    EXPECT_EQ(recreatedTensor.bufferShape, ovmsPyTensor.bufferShape);
    EXPECT_EQ(recreatedTensor.strides, ovmsPyTensor.strides);
    EXPECT_EQ(recreatedTensor.format, ovmsPyTensor.format);
    EXPECT_EQ(recreatedTensor.datatype, ovmsPyTensor.datatype);
    EXPECT_EQ(recreatedTensor.itemsize, ovmsPyTensor.itemsize);
    EXPECT_EQ(recreatedTensor.size, ovmsPyTensor.size);
}

TEST(OvmsPyTensor, BuildKnownFormatSingleDimShape) {
    py::ssize_t INPUT_BUFFER_SIZE = 1 * 3 * 300 * 300 * sizeof(float);
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
    EXPECT_EQ(ovmsPyTensor.userShape, shape);
    EXPECT_EQ(ovmsPyTensor.bufferShape, shape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);
    EXPECT_EQ(ovmsPyTensor.size, INPUT_BUFFER_SIZE);

    OvmsPyTensor recreatedTensor("input", py::buffer_info{
                                              ovmsPyTensor.ptr,
                                              ovmsPyTensor.itemsize,
                                              ovmsPyTensor.format,
                                              ovmsPyTensor.ndim,
                                              ovmsPyTensor.bufferShape,
                                              ovmsPyTensor.strides});

    EXPECT_EQ(recreatedTensor.name, "input");
    EXPECT_EQ(recreatedTensor.ptr, ovmsPyTensor.ptr);
    EXPECT_EQ(recreatedTensor.userShape, ovmsPyTensor.userShape);
    EXPECT_EQ(recreatedTensor.bufferShape, ovmsPyTensor.bufferShape);
    EXPECT_EQ(recreatedTensor.strides, ovmsPyTensor.strides);
    EXPECT_EQ(recreatedTensor.format, ovmsPyTensor.format);
    EXPECT_EQ(recreatedTensor.datatype, ovmsPyTensor.datatype);
    EXPECT_EQ(recreatedTensor.itemsize, ovmsPyTensor.itemsize);
    EXPECT_EQ(recreatedTensor.size, ovmsPyTensor.size);
}

TEST(OvmsPyTensor, BuildUnknownFormatSingleDimShape) {
    py::ssize_t INPUT_BUFFER_SIZE = 3 * 1024;
    std::string data(INPUT_BUFFER_SIZE, '1');
    void* ptr = data.data();
    std::vector<py::ssize_t> shape{3};
    std::string datatype = "my_string_type";
    OvmsPyTensor ovmsPyTensor("input", ptr, shape, datatype, INPUT_BUFFER_SIZE);

    std::string expectedFormat = datatypeToBufferFormat.at("UINT8");
    size_t expectedItemsize = bufferFormatToItemsize.at(expectedFormat);
    // For unknown format the underlying buffer is UINT8 1-D with shape (num_bytes,) and strides (1,)
    std::vector<py::ssize_t> expectedBufferShape{INPUT_BUFFER_SIZE};
    std::vector<py::ssize_t> expectedStrides{1};

    EXPECT_EQ(ovmsPyTensor.name, "input");
    EXPECT_EQ(ovmsPyTensor.ptr, ptr);
    EXPECT_EQ(ovmsPyTensor.userShape, shape);
    EXPECT_EQ(ovmsPyTensor.bufferShape, expectedBufferShape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);
    EXPECT_EQ(ovmsPyTensor.size, INPUT_BUFFER_SIZE);

    OvmsPyTensor recreatedTensor("input", py::buffer_info{
                                              ovmsPyTensor.ptr,
                                              ovmsPyTensor.itemsize,
                                              ovmsPyTensor.format,
                                              ovmsPyTensor.ndim,
                                              ovmsPyTensor.bufferShape,
                                              ovmsPyTensor.strides});

    EXPECT_EQ(recreatedTensor.name, "input");
    EXPECT_EQ(recreatedTensor.ptr, ovmsPyTensor.ptr);
    // When creating from another buffer we assign bufferShape to userShape
    EXPECT_EQ(recreatedTensor.userShape, ovmsPyTensor.bufferShape);
    EXPECT_EQ(recreatedTensor.bufferShape, ovmsPyTensor.bufferShape);
    EXPECT_EQ(recreatedTensor.strides, ovmsPyTensor.strides);
    EXPECT_EQ(recreatedTensor.format, ovmsPyTensor.format);
    // We cannot recreate known datatype only from buffer info. For unknown types we assume to UINT8
    EXPECT_EQ(recreatedTensor.datatype, "UINT8");
    EXPECT_EQ(recreatedTensor.itemsize, ovmsPyTensor.itemsize);
    EXPECT_EQ(recreatedTensor.size, ovmsPyTensor.size);
}

TEST(OvmsPyTensor, BuildUnknownFormatMultiDimShape) {
    py::ssize_t INPUT_BUFFER_SIZE = 10 * 3 * 1024;
    std::string data(INPUT_BUFFER_SIZE, '1');
    void* ptr = data.data();
    std::vector<py::ssize_t> shape{10, 3};
    std::string datatype = "my_string_type";
    OvmsPyTensor ovmsPyTensor("input", ptr, shape, datatype, INPUT_BUFFER_SIZE);

    std::string expectedFormat = datatypeToBufferFormat.at("UINT8");
    size_t expectedItemsize = bufferFormatToItemsize.at(expectedFormat);
    // For unknown format the underlying buffer is UINT8 1-D with shape (num_bytes,) and strides (1,)
    std::vector<py::ssize_t> expectedBufferShape{INPUT_BUFFER_SIZE};
    std::vector<py::ssize_t> expectedStrides{1};

    EXPECT_EQ(ovmsPyTensor.name, "input");
    EXPECT_EQ(ovmsPyTensor.ptr, ptr);
    EXPECT_EQ(ovmsPyTensor.userShape, shape);
    EXPECT_EQ(ovmsPyTensor.bufferShape, expectedBufferShape);
    EXPECT_EQ(ovmsPyTensor.strides, expectedStrides);
    EXPECT_EQ(ovmsPyTensor.format, expectedFormat);
    EXPECT_EQ(ovmsPyTensor.datatype, datatype);
    EXPECT_EQ(ovmsPyTensor.itemsize, expectedItemsize);
    EXPECT_EQ(ovmsPyTensor.size, INPUT_BUFFER_SIZE);

    OvmsPyTensor recreatedTensor("input", py::buffer_info{
                                              ovmsPyTensor.ptr,
                                              ovmsPyTensor.itemsize,
                                              ovmsPyTensor.format,
                                              ovmsPyTensor.ndim,
                                              ovmsPyTensor.bufferShape,
                                              ovmsPyTensor.strides});

    EXPECT_EQ(recreatedTensor.name, "input");
    EXPECT_EQ(recreatedTensor.ptr, ovmsPyTensor.ptr);
    // When creating from another buffer we assign bufferShape to userShape
    EXPECT_EQ(recreatedTensor.userShape, ovmsPyTensor.bufferShape);
    EXPECT_EQ(recreatedTensor.bufferShape, ovmsPyTensor.bufferShape);
    EXPECT_EQ(recreatedTensor.strides, ovmsPyTensor.strides);
    EXPECT_EQ(recreatedTensor.format, ovmsPyTensor.format);
    // We cannot recreate known datatype only from buffer info. For unknown types we assume to UINT8
    EXPECT_EQ(recreatedTensor.datatype, "UINT8");
    EXPECT_EQ(recreatedTensor.itemsize, ovmsPyTensor.itemsize);
    EXPECT_EQ(recreatedTensor.size, ovmsPyTensor.size);
}
