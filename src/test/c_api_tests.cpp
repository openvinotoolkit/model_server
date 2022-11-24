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

#include <gtest/gtest.h>

// TODO we should not include classes from OVMS here
// consider how to workaround test_utils
#include "../config.hpp"
#include "../inferenceresponse.hpp"
#include "../poc_api_impl.hpp"
#include "../pocapi.hpp"
#include "test_utils.hpp"

using namespace ovms;
using testing::ElementsAreArray;

class CapiConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(CapiConfigTest, Parse) {
    GeneralOptionsImpl go;
    MultiModelOptionsImpl mmo;

    go.grpcPort = 123;
    go.restPort = 234;
    mmo.configPath = "/path/config.json";

    ovms::Config::instance().parse(&go, &mmo);
    EXPECT_EQ(ovms::Config::instance().port(), 123);
    EXPECT_EQ(ovms::Config::instance().restPort(), 234);
    EXPECT_EQ(ovms::Config::instance().configPath(), "/path/config.json");
}

class CapiInferencePreparationTest : public ::testing::Test {};

TEST_F(CapiInferencePreparationTest, Basic) {
    // request creation
    OVMS_InferenceRequest* request{nullptr};
    OVMS_Status* status = OVMS_InferenceRequestNew(&request, "dummy", 1);
    ASSERT_EQ(nullptr, status);
    ASSERT_NE(nullptr, request);

    // adding input
    status = OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size());
    ASSERT_EQ(nullptr, status);
    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    status = OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum);
    ASSERT_EQ(nullptr, status);
    // add parameters
    const uint64_t sequenceId{42};
    status = OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId));
    ASSERT_EQ(nullptr, status);
    const uint32_t sequenceControl{1};  // SEQUENCE_START
    status = OVMS_InferenceRequestAddParameter(request, "sequence_control_input", OVMS_DATATYPE_U32, reinterpret_cast<const void*>(&sequenceControl), sizeof(sequenceControl));
    ASSERT_EQ(nullptr, status);

    //////////////////
    //  INFERENCE
    //////////////////

    ///////////////
    // CLEANUP
    ///////////////
    // here we will add additional inputs to verify 2 ways of cleanup
    // - direct call to remove whole request
    // - separate calls to remove partial data
    //
    // here we will just add inputs to remove them later
    // one original will be removed together with buffer during whole request removal
    // one will be removed together with request but without buffer attached
    // one will be removed with buffer directly
    // one will be removed without buffer directly
    status = OVMS_InferenceRequestAddInput(request, "INPUT_WITHOUT_BUFFER_REMOVED_WITH_REQUEST", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size());
    ASSERT_EQ(nullptr, status);
    status = OVMS_InferenceRequestAddInput(request, "INPUT_WITH_BUFFER_REMOVED_DIRECTLY", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size());
    ASSERT_EQ(nullptr, status);
    status = OVMS_InferenceRequestAddInput(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size());
    ASSERT_EQ(nullptr, status);
    status = OVMS_InferenceRequestInputSetData(request, "INPUT_WITH_BUFFER_REMOVED_DIRECTLY", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum);
    ASSERT_EQ(nullptr, status);
    // we will add buffer and remove it to check separate buffer removal
    status = OVMS_InferenceRequestInputSetData(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum);
    ASSERT_EQ(nullptr, status);

    status = OVMS_InferenceRequestInputRemoveData(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY");
    ASSERT_EQ(nullptr, status);
    status = OVMS_InferenceRequestRemoveInput(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY");
    ASSERT_EQ(nullptr, status);
    status = OVMS_InferenceRequestRemoveInput(request, "INPUT_WITH_BUFFER_REMOVED_DIRECTLY");
    ASSERT_EQ(nullptr, status);
    // we will remove 1 of two parameters
    status = OVMS_InferenceRequestRemoveParameter(request, "sequence_id");
    ASSERT_EQ(nullptr, status);

    status = OVMS_InferenceRequestDelete(request);
    ASSERT_EQ(nullptr, status);
}
// TODO negative test -> validate at the infer stage
// TODO flow with removel just request no separate input/buffer
// TODO reuse request after inference
namespace {
const std::string MODEL_NAME{"SomeModelName"};
const uint64_t MODEL_VERSION{42};
const std::string PARAMETER_NAME{"sequence_id"};  // TODO check if in ovms there is such constant
const OVMS_DataType PARAMETER_DATATYPE{OVMS_DATATYPE_I32};

const uint32_t PARAMETER_VALUE{13};
const uint32_t PRIORITY{7};
const uint64_t REQUEST_ID{3};

const std::string INPUT_NAME{"NOT_RANDOM_NAME"};
const ovms::shape_t INPUT_SHAPE{1, 3, 220, 230};
const std::array<float, DUMMY_MODEL_INPUT_SIZE> INPUT_DATA{1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
constexpr size_t INPUT_DATA_BYTESIZE{INPUT_DATA.size() * sizeof(float)};
const OVMS_DataType DATATYPE{OVMS_DATATYPE_FP32};
}  // namespace
class CapiInferenceRetrievalTest : public ::testing::Test {};
TEST_F(CapiInferenceRetrievalTest, Basic) {
    auto cppResponse = std::make_unique<InferenceResponse>(MODEL_NAME, MODEL_VERSION);
    // add output
    std::array<size_t, 2> cppOutputShape{1, DUMMY_MODEL_INPUT_SIZE};
    auto cppStatus = cppResponse->addOutput(INPUT_NAME.c_str(), DATATYPE, cppOutputShape.data(), cppOutputShape.size());
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();
    InferenceTensor* cpptensor = nullptr;
    const std::string* cppOutputName;
    cppStatus = cppResponse->getOutput(0, &cppOutputName, &cpptensor);
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();

    // save data into output (it should have it's own copy in contrast to request)
    bool createCopy = true;
    cppStatus = cpptensor->setBuffer(INPUT_DATA.data(), INPUT_DATA_BYTESIZE, OVMS_BUFFERTYPE_CPU, std::nullopt, createCopy);
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();
    // add parameter to response
    uint64_t seqId = 666;
    cppStatus = cppResponse->addParameter("sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<void*>(&seqId));
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();
    ///////////////////////////
    // now response is prepared so we can test C-API
    ///////////////////////////
    OVMS_InferenceResponse* response = reinterpret_cast<OVMS_InferenceResponse*>(cppResponse.get());
    uint32_t outputCount = -1;
    auto status = OVMS_InferenceResponseGetOutputCount(response, &outputCount);
    ASSERT_EQ(nullptr, status);
    ASSERT_EQ(outputCount, 1);

    uint32_t parameterCount = -1;
    status = OVMS_InferenceResponseGetParameterCount(response, &parameterCount);
    ASSERT_EQ(nullptr, status);
    ASSERT_EQ(1, parameterCount);
    // verify get Parameter
    OVMS_DataType parameterDatatype = OVMS_DATATYPE_FP32;
    const void* parameterData{nullptr};
    status = OVMS_InferenceResponseGetParameter(response, 0, &parameterDatatype, &parameterData);
    ASSERT_EQ(nullptr, status);
    ASSERT_EQ(parameterDatatype, OVMS_DATATYPE_U64);
    EXPECT_EQ(0, std::memcmp(parameterData, (void*)&seqId, sizeof(seqId)));
    // verify get Output
    void* voutputData;
    size_t bytesize = -1;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const uint64_t* shape{nullptr};
    uint32_t dimCount = -1;
    BufferType bufferType = (BufferType)199;
    uint32_t deviceId = -1;
    const char* outputName{nullptr};
    status = OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId);
    ASSERT_EQ(nullptr, status);
    ASSERT_EQ(INPUT_NAME, outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);

    for (size_t i = 0; i < cppOutputShape.size(); ++i) {
        EXPECT_EQ(cppOutputShape[i], shape[i]) << "Different at:" << i << " place.";
    }
    float* outputData = reinterpret_cast<float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < INPUT_DATA.size(); ++i) {
        EXPECT_EQ(INPUT_DATA[i], outputData[i]) << "Different at:" << i << " place.";
    }

    // we release unique_ptr ownership here so that we can free it safely via C-API
    cppResponse.release();
    status = OVMS_InferenceResponseDelete(response);
    ASSERT_EQ(nullptr, status);
}
// TODO make cleaner error codes reporting
// todo decide either use remove or delete for consistency
