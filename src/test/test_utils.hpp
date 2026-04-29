//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#pragma once

#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "src/dags/node_library.hpp"
#include "src/execution_context.hpp"
#include "src/modelconfig.hpp"
#include "src/precision.hpp"
#include "src/shape.hpp"
#include "src/status.hpp"
#include "src/tensorinfo.hpp"

#include "test_models.hpp"

// ============================================================================
// Core test utilities (frontend-agnostic)
// ============================================================================

using inputs_info_t = std::map<std::string, std::tuple<ovms::signed_shape_t, ovms::Precision>>;

void adjustConfigToAllowModelFileRemovalWhenLoaded(ovms::ModelConfig& modelConfig);

static const ovms::ExecutionContext DEFAULT_TEST_CONTEXT{ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::Predict};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
void printTensor(const ov::Tensor& tensor);

ovms::tensor_map_t prepareTensors(
    const std::unordered_map<std::string, ovms::Shape>&& tensors,
    ovms::Precision precision = ovms::Precision::FP32);

template <typename T>
std::string readableError(const T* expected_output, const T* actual_output, const size_t size) {
    std::stringstream ss;
    for (size_t i = 0; i < size; ++i) {
        if (actual_output[i] != expected_output[i]) {
            ss << "Expected:" << expected_output[i] << ", actual:" << actual_output[i] << " at place:" << i << std::endl;
            break;
        }
    }
    return ss.str();
}

std::string readableSetError(std::unordered_set<std::string> expected, std::unordered_set<std::string> actual);

template <typename T>
void checkBuffers(const T* expected, const T* actual, size_t bufferSize) {
    EXPECT_EQ(0, std::memcmp(actual, expected, bufferSize))
        << readableError(expected, actual, bufferSize / sizeof(T));
}

template <typename TensorType>
void prepareInvalidImageBinaryTensor(TensorType& tensor);

void assertOutputTensorMatchExpectations(const ov::Tensor& tensor, std::vector<std::string> expectedStrings);

template <typename T>
static std::vector<T> asVector(const std::string& tensor_content) {
    std::vector<T> v(tensor_content.size() / sizeof(T) + 1);
    std::memcpy(
        reinterpret_cast<char*>(v.data()),
        reinterpret_cast<const char*>(tensor_content.data()),
        tensor_content.size());
    v.resize(tensor_content.size() / sizeof(T));
    return v;
}
#pragma GCC diagnostic pop

void RemoveReadonlyFileAttributeFromDir(std::string& directoryPath);
void SetReadonlyFileAttributeFromDir(std::string& directoryPath);

void readRgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes);
void read4x4RgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes);
void readFile(const std::string& path, size_t& filesize, std::unique_ptr<char[]>& bytes);

template <typename T>
static ovms::NodeLibrary createLibraryMock() {
    return ovms::NodeLibrary{
        T::initialize,
        T::deinitialize,
        T::execute,
        T::getInputsInfo,
        T::getOutputsInfo,
        T::release};
}

std::shared_ptr<const ovms::TensorInfo> createTensorInfoCopyWithPrecision(std::shared_ptr<const ovms::TensorInfo> src, ovms::Precision precision);
