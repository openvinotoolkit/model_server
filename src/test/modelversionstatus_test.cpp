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

#include "../modelversionstatus.hpp"
#include "gtest/gtest.h"

using namespace ovms;

TEST(ModelVersionStatus, state_transitions) {
    ModelVersionStatus mvs("SampleModelName", 15);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::START);

    mvs.setLoading(ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::LOADING);

    mvs.setAvailable(ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::AVAILABLE);

    mvs.setUnloading(ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::UNLOADING);

    mvs.setEnd(ModelVersionStatusErrorCode::OK);
    EXPECT_EQ(mvs.getState(), ovms::ModelVersionState::END);
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
