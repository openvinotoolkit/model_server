//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#pragma GCC diagnostic pop

#include "../modelversionstatus.hpp"
#include "gtest/gtest.h"

using namespace ovms;

TEST(ModelVersionStatus, casting_to_protobuf_enum) {
    ModelVersionStatus mvs("SampleModelName", 15);
    tensorflow::serving::ModelVersionStatus_State tf_state = static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(mvs.getState()));
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::START);
    EXPECT_EQ(tf_state, tensorflow::serving::ModelVersionStatus_State_START);

    mvs.setLoading(ModelVersionStatusErrorCode::OK);
    tf_state = static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(mvs.getState()));
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::LOADING);
    EXPECT_EQ(tf_state, tensorflow::serving::ModelVersionStatus_State_LOADING);

    mvs.setAvailable(ModelVersionStatusErrorCode::OK);
    tf_state = static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(mvs.getState()));
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::AVAILABLE);
    EXPECT_EQ(tf_state, tensorflow::serving::ModelVersionStatus_State_AVAILABLE);

    mvs.setUnloading(ModelVersionStatusErrorCode::OK);
    tf_state = static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(mvs.getState()));
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::UNLOADING);
    EXPECT_EQ(tf_state, tensorflow::serving::ModelVersionStatus_State_UNLOADING);

    mvs.setEnd(ModelVersionStatusErrorCode::OK);
    tf_state = static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(mvs.getState()));
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::END);
    EXPECT_EQ(tf_state, tensorflow::serving::ModelVersionStatus_State_END);
}

TEST(ModelVersionStatus, simple_test_flow) {
    ModelVersionStatus mvs("SampleModelName", 15);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::START);
    EXPECT_EQ(mvs.getStateString(), "START");
    EXPECT_EQ(mvs.getErrorCode(), ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getErrorMsg(), "OK");

    mvs.setLoading();
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::LOADING);
    EXPECT_EQ(mvs.getStateString(), "LOADING");
    EXPECT_EQ(mvs.getErrorCode(), ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getErrorMsg(), "OK");

    mvs.setAvailable();
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::AVAILABLE);
    EXPECT_EQ(mvs.getStateString(), "AVAILABLE");
    EXPECT_EQ(mvs.getErrorCode(), ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getErrorMsg(), "OK");

    mvs.setUnloading();
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::UNLOADING);
    EXPECT_EQ(mvs.getStateString(), "UNLOADING");
    EXPECT_EQ(mvs.getErrorCode(), ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getErrorMsg(), "OK");

    mvs.setEnd();
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::END);
    EXPECT_EQ(mvs.getStateString(), "END");
    EXPECT_EQ(mvs.getErrorCode(), ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getErrorMsg(), "OK");
}

TEST(ModelVersionStatus, loading_fault_test_flow) {
    ModelVersionStatus mvs("SampleModelName", 15);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::START);
    EXPECT_EQ(mvs.getStateString(), "START");

    mvs.setLoading();
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::LOADING);
    EXPECT_EQ(mvs.getStateString(), "LOADING");
    EXPECT_EQ(mvs.getErrorCode(), ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getErrorMsg(), "OK");

    mvs.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::LOADING);
    EXPECT_EQ(mvs.getStateString(), "LOADING");
    EXPECT_EQ(mvs.getErrorCode(), ModelVersionStatusErrorCode::UNKNOWN);
    EXPECT_EQ(mvs.getErrorMsg(), "UNKNOWN");

    // others are not implemented in python version.
}
