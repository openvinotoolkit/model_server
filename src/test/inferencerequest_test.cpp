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

#include "../capi_frontend/buffer.hpp"
#include "../capi_frontend/inferenceparameter.hpp"
#include "../capi_frontend/inferencerequest.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
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
const std::string PARAMETER_NAME{"SEQUENCE_ID"};
const OVMS_DataType PARAMETER_DATATYPE{OVMS_DATATYPE_I32};

const uint32_t PARAMETER_VALUE{13};
const uint32_t PRIORITY{7};
const uint64_t REQUEST_ID{3};

const std::string INPUT_NAME{"NOT_RANDOM_NAME"};
const ovms::signed_shape_t INPUT_SHAPE{1, 3, 220, 230};
const std::array<float, 10> INPUT_DATA{1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
constexpr size_t INPUT_DATA_BYTESIZE{INPUT_DATA.size() * sizeof(float)};
const OVMS_DataType DATATYPE{OVMS_DATATYPE_FP32};
}  // namespace

TEST(InferenceParameter, CreateParameter) {
    InferenceParameter parameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
}
TEST(Buffer, StringHandling) {
    using std::string;
    using vs_t = std::vector<string>;
    const string* stringPtr{nullptr};
    vs_t intelText{{"Intel"}, {"owns"}, {"OVMS"}};
    vs_t nvidiaText{{"NVIDIA"}, {"owns"}, {"Triton"}};
    std::unique_ptr<const Buffer> bufferWithCopy{nullptr};
    {
        vs_t text2BeDeleted = nvidiaText;
        const Buffer bufferWithNoCopy(&intelText[0], intelText.size() * sizeof(string), OVMS_BUFFERTYPE_CPU);
        auto text2BeMoved = std::make_unique<vs_t>(std::move(text2BeDeleted));
        bufferWithCopy = std::make_unique<const Buffer>(std::move(text2BeMoved));
        // check with nocopy
        EXPECT_EQ(intelText.data(), &intelText[0]);
        EXPECT_EQ(bufferWithNoCopy.data(), &intelText[0]);
        stringPtr = reinterpret_cast<const string*>(bufferWithNoCopy.data());
        EXPECT_EQ(intelText.size(), bufferWithNoCopy.getByteSize() / sizeof(string));
        EXPECT_TRUE(std::equal(intelText.begin(), intelText.end(), stringPtr));
        // now check with copy
        stringPtr = reinterpret_cast<const string*>(bufferWithCopy->data());
        EXPECT_EQ(nvidiaText.size(), bufferWithNoCopy.getByteSize() / sizeof(string));
        EXPECT_TRUE(std::equal(nvidiaText.begin(), nvidiaText.end(), stringPtr));
    }
    // now text is deleted but for buffer with copy still expect to work
    vs_t randomData{{"Akademia"}, {"Pana"}, {"Kleksa"}};
    stringPtr = reinterpret_cast<const string*>(bufferWithCopy->data());
    EXPECT_EQ(bufferWithCopy->getByteSize(), sizeof(string) * nvidiaText.size());
    EXPECT_EQ(bufferWithCopy->getBufferType(), OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(bufferWithCopy->getDeviceId(), std::nullopt);
    EXPECT_TRUE(std::equal(nvidiaText.begin(), nvidiaText.end(), stringPtr));
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
    EXPECT_EQ(*(reinterpret_cast<const uint32_t*>(parameter->getData())), PARAMETER_VALUE);
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
    ASSERT_EQ(tensor->getShape().size(), INPUT_SHAPE.size());
    EXPECT_EQ(std::memcmp(tensor->getShape().data(), INPUT_SHAPE.data(), INPUT_SHAPE.size() * sizeof(int64_t)), 0);
    const Buffer* buffer = tensor->getBuffer();
    ASSERT_NE(nullptr, buffer);
    ASSERT_NE(nullptr, buffer->data());
    ASSERT_EQ(buffer->data(), INPUT_DATA.data());
    ASSERT_EQ(buffer->getByteSize(), INPUT_DATA_BYTESIZE);
    // save data 2nd time should fail
    // compare data content with what was saved
    auto res = std::memcmp(buffer->data(), reinterpret_cast<const void*>(INPUT_DATA.data()), INPUT_DATA_BYTESIZE);
    EXPECT_EQ(0, res) << res;

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

    //////////////////////
    std::vector<int64_t> stringShape{4};
    std::vector<std::string> strings{"Intel", "OpenVINO", "Model", "Server"};
    status = request.addInput(INPUT_NAME.c_str(), OVMS_DATATYPE_STRING, stringShape.data(), stringShape.size());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // set input buffer
    status = request.setInputBuffer(INPUT_NAME.c_str(), strings.data(), strings.size() * sizeof(float), OVMS_BUFFERTYPE_CPU, std::nullopt);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
}
TEST(InferenceResponse, CreateAndReadData) {
    // create response
    InferenceResponse response{MODEL_NAME, MODEL_VERSION};
    EXPECT_EQ(response.getServableName(), MODEL_NAME);
    EXPECT_EQ(response.getServableVersion(), MODEL_VERSION);
    // add output
    auto status = response.addOutput(INPUT_NAME.c_str(), DATATYPE, INPUT_SHAPE.data(), INPUT_SHAPE.size());
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    EXPECT_EQ(response.getOutputCount(), 1);
    // add 2nd output with the same name should fail
    status = response.addOutput(INPUT_NAME.c_str(), DATATYPE, INPUT_SHAPE.data(), INPUT_SHAPE.size());
    ASSERT_EQ(status, StatusCode::DOUBLE_TENSOR_INSERT) << status.string();
    // get nonexistent output
    const std::string* wrongOutputName = nullptr;
    InferenceTensor* tensor = nullptr;
    status = response.getOutput(13, &wrongOutputName, &tensor);
    ASSERT_EQ(status, StatusCode::NONEXISTENT_TENSOR) << status.string();
    ASSERT_EQ(nullptr, tensor);
    ASSERT_EQ(nullptr, wrongOutputName);
    // get output
    const std::string* outputName = nullptr;
    status = response.getOutput(0, &outputName, &tensor);
    ASSERT_NE(nullptr, tensor);
    ASSERT_EQ(INPUT_NAME, *outputName);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // compare datatype
    ASSERT_EQ(tensor->getDataType(), DATATYPE);
    // compare shape
    auto shape = tensor->getShape();
    ASSERT_THAT(shape, ElementsAre(1, 3, 220, 230));

    // save data into output (it should have it's own copy in contrast to request)
    bool createCopy = true;
    status = tensor->setBuffer(INPUT_DATA.data(), INPUT_DATA_BYTESIZE, OVMS_BUFFERTYPE_CPU, std::nullopt, createCopy);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    // savind data into output twice should fail
    std::array<float, 10> RANDOM_DATA{10., 9, 8, 7, 6, 5, 4, 3, 2, 1};
    status = tensor->setBuffer(RANDOM_DATA.data(), INPUT_DATA_BYTESIZE, OVMS_BUFFERTYPE_CPU, std::nullopt, createCopy);
    ASSERT_EQ(status, StatusCode::DOUBLE_BUFFER_SET) << status.string();

    const Buffer* buffer = tensor->getBuffer();
    ASSERT_NE(nullptr, buffer);
    ASSERT_NE(nullptr, buffer->data());
    ASSERT_NE(buffer->data(), INPUT_DATA.data());
    ASSERT_EQ(buffer->getByteSize(), INPUT_DATA_BYTESIZE);
    // save data 2nd time should fail
    // compare data content with what was saved
    auto res = std::memcmp(buffer->data(), reinterpret_cast<const void*>(INPUT_DATA.data()), INPUT_DATA_BYTESIZE);
    EXPECT_EQ(0, res) << res;

    // verify parameter handling
    status = response.addParameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    EXPECT_EQ(response.getParameterCount(), 1);

    const InferenceParameter* parameter = response.getParameter(1);
    ASSERT_EQ(parameter, nullptr);
    parameter = response.getParameter(0);
    ASSERT_NE(parameter, nullptr);
    EXPECT_EQ(parameter->getName(), PARAMETER_NAME);
    EXPECT_EQ(parameter->getDataType(), PARAMETER_DATATYPE);
    EXPECT_EQ(*(reinterpret_cast<const uint32_t*>(parameter->getData())), PARAMETER_VALUE);
    // add same parameter second time should fail
    status = response.addParameter(PARAMETER_NAME.c_str(), PARAMETER_DATATYPE, reinterpret_cast<const void*>(&PARAMETER_VALUE));
    ASSERT_EQ(status, StatusCode::DOUBLE_PARAMETER_INSERT) << status.string();
}
