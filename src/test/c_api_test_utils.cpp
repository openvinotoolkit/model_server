//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "c_api_test_utils.hpp"

#include <vector>

#include "../logging.hpp"
#include "../ovms.h"  // NOLINT

void callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPICorrectness(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    using ovms::StatusCode;
    SPDLOG_INFO("Using callback: callbackMarkingItWasUsedWith42!");
    uint32_t* usedFlag = reinterpret_cast<uint32_t*>(userStruct);
    *usedFlag = 42;
    OVMS_InferenceResponseDelete(response);
}
void callbackMarkingItWasUsedWith42(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    using ovms::StatusCode;
    SPDLOG_INFO("Using callback: callbackMarkingItWasUsedWith42!");
    uint32_t* usedFlag = reinterpret_cast<uint32_t*>(userStruct);
    *usedFlag = 42;
    OVMS_InferenceResponseDelete(response);
}
void checkDummyResponse(OVMS_InferenceResponse* response, double expectedValue, double tolerance) {
    const void* voutputData{nullptr};
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t ovmsDeviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &ovmsDeviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(ovmsDeviceId, 0);
    std::vector<int> expectedShape{1, 10};
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(expectedShape[i], shape[i]) << "Different at:" << i << " place.";
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    for (int i = 0; i < expectedShape[1]; ++i) {
        EXPECT_NEAR(expectedValue, outputData[i], tolerance) << "Different at:" << i << " place.";
    }
}
