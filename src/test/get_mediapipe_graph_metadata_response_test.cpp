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

#include <chrono>
#include <future>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>

#if (MEDIAPIPE_DISABLE == 0)
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#endif
#include "../get_model_metadata_impl.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace rapidjson;

class DummyMediapipeGraphDefinition : public MediapipeGraphDefinition {
public:
    std::string inputConfig;

public:
    DummyMediapipeGraphDefinition(const std::string name,
        const MediapipeGraphConfig& config,
        std::string inputConfig) :
        MediapipeGraphDefinition(name, config, nullptr, nullptr) {}

    // Do not read from path - use predefined config contents
    Status validateForConfigFileExistence() override {
        this->chosenConfig = this->inputConfig;
        return StatusCode::OK;
    }
};

class GetMediapipeGraphMetadataResponse : public ::testing::Test {
protected:
    KFSModelMetadataResponse response;
    ConstructorEnabledModelManager manager;
};

TEST_F(GetMediapipeGraphMetadataResponse, BasicResponseMetadata) {
    std::string testPbtxt = R"(
        input_stream: "TEST:in"
        input_stream: "TEST33:in2"
        output_stream: "TEST0:out"
        output_stream: "TEST1:out2"
        output_stream: "TEST3:out3"
            node {
            calculator: "OVMSOVCalculator"
            input_stream: "B:in"
            output_stream: "A:out"
            }
        )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeGraphDefinition("mediaDummy", mgc, testPbtxt);
    mediapipeGraphDefinition.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeGraphDefinition.validate(manager), StatusCode::OK);

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(mediapipeGraphDefinition, &response), ovms::StatusCode::OK);
    EXPECT_EQ(response.name(), "mediaDummy");

    EXPECT_EQ(response.versions_size(), 1);
    EXPECT_EQ(response.versions().at(0), "1");

    EXPECT_EQ(response.platform(), "OpenVINO");

    EXPECT_EQ(response.inputs_size(), 2);
    auto firstInput = response.inputs().at(0);
    EXPECT_EQ(firstInput.name(), "in");
    EXPECT_EQ(firstInput.datatype(), "INVALID");
    EXPECT_EQ(firstInput.shape_size(), 0);
    auto secondInput = response.inputs().at(1);
    EXPECT_EQ(secondInput.name(), "in2");
    EXPECT_EQ(secondInput.datatype(), "INVALID");
    EXPECT_EQ(secondInput.shape_size(), 0);

    EXPECT_EQ(response.outputs_size(), 3);
    auto firstOutput = response.outputs().at(0);
    EXPECT_EQ(firstOutput.name(), "out");
    EXPECT_EQ(firstOutput.datatype(), "INVALID");
    EXPECT_EQ(firstOutput.shape_size(), 0);
    auto secondOutput = response.outputs().at(1);
    EXPECT_EQ(secondOutput.name(), "out2");
    EXPECT_EQ(secondOutput.datatype(), "INVALID");
    EXPECT_EQ(secondOutput.shape_size(), 0);

    auto thirdOutput = response.outputs().at(2);
    EXPECT_EQ(thirdOutput.name(), "out3");
    EXPECT_EQ(thirdOutput.datatype(), "INVALID");
    EXPECT_EQ(thirdOutput.shape_size(), 0);
}
