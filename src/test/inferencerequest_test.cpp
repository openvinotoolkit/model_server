//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <algorithm>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../buffer.hpp"
#include "../inferenceparameter.hpp"
#include "../inferencerequest.hpp"
#include "../inferenceresponse.hpp"
#include "../logging.hpp"
#include "../shape.hpp"
#include "../status.hpp"

using testing::ElementsAre;

using ovms::Buffer;
using ovms::InferenceParameter;
using ovms::InferenceRequest;
using ovms::InferenceResponse;
using ovms::InferenceTensor;
using ovms::Shape;
using ovms::Status;
using ovms::StatusCode;

namespace {
const std::string MODEL_NAME{"SomeModelName"};
const uint64_t MODEL_VERSION{42};
const std::string PARAMETER_NAME{"SEQUENCE_ID"};  // TODO check if in ovms there is such constant
const DataType PARAMETER_DATATYPE{OVMS_DATATYPE_I32};

const uint32_t PARAMETER_VALUE{13};
const uint32_t PRIORITY{7};
const uint64_t REQUEST_ID{3};

const std::string INPUT_NAME{"NOT_RANDOM_NAME"};
const ovms::shape_t INPUT_SHAPE{1, 3, 220, 230};
const std::array<float, 10> INPUT_DATA{1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
constexpr size_t INPUT_DATA_BYTESIZE{INPUT_DATA.size() * sizeof(float)};
const DataType DATATYPE{OVMS_DATATYPE_FP32};
}  // namespace

TEST(InferenceParameter, CreateParameter) {
    InferenceParameter parameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
}

TEST(InferenceRequest, CreateInferenceRequest) {
    InferenceRequest request(MODEL_NAME.c_str(), MODEL_VERSION);
    EXPECT_EQ(request.getServableName(), MODEL_NAME);
    EXPECT_EQ(request.getServableVersion(), MODEL_VERSION);

    auto status = request.addParameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
    ASSERT_EQ(status, StatusCode::OK) << status.string();

    // add parameter
    const InferenceParameter* parameter = request.getParameter(PARAMETER_NAME.c_str());
    ASSERT_NE(parameter, nullptr);
    EXPECT_EQ(parameter->getName(), PARAMETER_NAME);
    EXPECT_EQ(parameter->getDataType(), PARAMETER_DATATYPE);
    EXPECT_EQ(*(reinterpret_cast<uint32_t*>(const_cast<void*>(parameter->getData()))), PARAMETER_VALUE);
    // add same parameter second time should fail
    status = request.addParameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
    ASSERT_EQ(status, StatusCode::DOUBLE_PARAMETER_INSERT) << status.string();

    // add input
    status = request.addInput(INPUT_NAME.c_str(), DATATYPE, INPUT_SHAPE.data(), INPUT_SHAPE.size());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // add same input second time should fail
    status = request.addInput(INPUT_NAME.c_str(), DATATYPE, INPUT_SHAPE.data(), INPUT_SHAPE.size());
    ASSERT_EQ(status, StatusCode::DOUBLE_TENSOR_INSERT) << status.string();

    // set input buffer
    status = request.setInputBuffer(INPUT_NAME.c_str(), INPUT_DATA.data(), INPUT_DATA.size() * sizeof(float), OVMS_BUFFERTYPE_CPU, std::nullopt);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // set buffer second time should fail
    status = request.setInputBuffer(INPUT_NAME.c_str(), INPUT_DATA.data(), INPUT_DATA.size() * sizeof(float), OVMS_BUFFERTYPE_CPU, std::nullopt);
    ASSERT_EQ(status, StatusCode::DOUBLE_BUFFER_SET) << status.string();

    // get input & buffer
    const InferenceTensor* tensor;
    status = request.getInput(INPUT_NAME.c_str(), &tensor);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    ASSERT_NE(nullptr, tensor);
    EXPECT_EQ(tensor->getDataType(), DATATYPE);
    EXPECT_TRUE(Shape(tensor->getShape()).match(INPUT_SHAPE));
    const Buffer* buffer = tensor->getBuffer();
    ASSERT_NE(nullptr, buffer);

    // remove input buffer
    status = request.removeInputBuffer(INPUT_NAME.c_str());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    ASSERT_EQ(nullptr, tensor->getBuffer());
    // remove buffer twice
    status = request.removeInputBuffer(INPUT_NAME.c_str());
    ASSERT_EQ(status, StatusCode::NONEXISTENT_BUFFER_FOR_REMOVAL) << status.string();

    // remove input
    status = request.removeInput(INPUT_NAME.c_str());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // verify removing all inputs
    status = request.addInput(INPUT_NAME.c_str(), DATATYPE, INPUT_SHAPE.data(), INPUT_SHAPE.size());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    status = request.removeAllInputs();
    ASSERT_EQ(status, StatusCode::OK) << status.string();

    // verify removing parameter
    status = request.removeParameter(PARAMETER_NAME.c_str());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    ASSERT_EQ(nullptr, request.getParameter(PARAMETER_NAME.c_str()));
}
TEST(InferenceResponse, CreateAndReadData) {
    // create response
    InferenceResponse response{MODEL_NAME, MODEL_VERSION};
    EXPECT_EQ(response.getServableName(), MODEL_NAME);
    EXPECT_EQ(response.getServableVersion(), MODEL_VERSION);
    // add output
    auto status = response.addOutput(INPUT_NAME.c_str(), DATATYPE, INPUT_SHAPE.data(), INPUT_SHAPE.size());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // add 2nd output with the same name should fail
    status = response.addOutput(INPUT_NAME.c_str(), DATATYPE, INPUT_SHAPE.data(), INPUT_SHAPE.size());
    ASSERT_EQ(status, StatusCode::DOUBLE_TENSOR_INSERT) << status.string();
    // get nonexistent output
    InferenceTensor* tensor = nullptr;
    status = response.getOutput("SOME_NOT_RANDOM_NAME", &tensor);
    ASSERT_EQ(status, StatusCode::NONEXISTENT_INPUT) << status.string();
    // get output
    status = response.getOutput(INPUT_NAME.c_str(), &tensor);
    ASSERT_NE(nullptr, tensor);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // compare datatype
    ASSERT_EQ(tensor->getDataType(), DATATYPE);
    // compare shape
    auto shape = tensor->getShape();
    ASSERT_THAT(shape, ElementsAre(1, 3, 220, 230));

    // save data into output (it should have it's own copy in contrast to request)
    bool ownCopy = true;
    status = tensor->setBuffer(INPUT_DATA.data(), INPUT_DATA_BYTESIZE, OVMS_BUFFERTYPE_CPU, std::nullopt, ownCopy);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // savind data into output twice should fail
    std::array<float, 10> RANDOM_DATA{10., 9, 8, 7, 6, 5, 4, 3, 2, 1};
    status = tensor->setBuffer(RANDOM_DATA.data(), INPUT_DATA_BYTESIZE, OVMS_BUFFERTYPE_CPU, std::nullopt, ownCopy);
    ASSERT_EQ(status, StatusCode::DOUBLE_BUFFER_SET) << status.string();

    const Buffer* buffer = tensor->getBuffer();
    ASSERT_NE(nullptr, buffer);
    ASSERT_NE(nullptr, buffer->data());
    ASSERT_NE(buffer->data(), INPUT_DATA.data());
    // save data 2nd time should fail
    // compare data content with what was saved
    auto res = std::memcmp(buffer->data(), reinterpret_cast<void*>(const_cast<float*>(INPUT_DATA.data())), INPUT_DATA_BYTESIZE);
    EXPECT_EQ(0, res) << res;

    // verify parameter handling
    status = response.addParameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
    ASSERT_EQ(status, StatusCode::OK) << status.string();

    // add parameter
    const InferenceParameter* parameter = response.getParameter(PARAMETER_NAME.c_str());
    ASSERT_NE(parameter, nullptr);
    EXPECT_EQ(parameter->getName(), PARAMETER_NAME);
    EXPECT_EQ(parameter->getDataType(), PARAMETER_DATATYPE);
    EXPECT_EQ(*(reinterpret_cast<uint32_t*>(const_cast<void*>(parameter->getData()))), PARAMETER_VALUE);
    // add same parameter second time should fail
    status = response.addParameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
    ASSERT_EQ(status, StatusCode::DOUBLE_PARAMETER_INSERT) << status.string();
}
// TODO logging
