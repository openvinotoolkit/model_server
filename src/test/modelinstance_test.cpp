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
#include <iostream>
#include <fstream>
#include <filesystem>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stdlib.h>

#include "../modelinstance.hpp"

using testing::Return;

const std::vector<ovms::ModelVersionState> INFER_QUEUE_SUCCESS_FOR_STATES {
    ovms::ModelVersionState::AVAILABLE
};
const std::vector<ovms::ModelVersionState> INFER_QUEUE_FAILURE_FOR_STATES {
    ovms::ModelVersionState::START,
    ovms::ModelVersionState::LOADING,
    ovms::ModelVersionState::UNLOADING,
    ovms::ModelVersionState::END
};

const ovms::ModelConfig DUMMY_MODEL_CONFIG {
   "dummy",
   std::filesystem::current_path().u8string() + "/src/test/dummy",
   "CPU",  // backend
   1,  // batchsize
   1,  // NIREQ
};

namespace {
class MockModelInstanceInState : public ovms::ModelInstance {
    static const ovms::model_version_t UNUSED_VERSION = 987789;
public:
    MockModelInstanceInState(ovms::ModelVersionState state) {
        status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_VERSION, state);
   }
};

class MockModelInstance : public ovms::ModelInstance {
public:
   // MOCK_METHOD(void, loadInputTensors, (const ovms::ModelConfig& config), ());
   // MOCK_METHOD(void, loadOutputTensors, (const ovms::ModelConfig& config), ());
   // MOCK_METHOD(const std::string&, getName, (), ());
   // MOCK_METHOD(const std::string&, getPath, (), ());
   // MOCK_METHOD(const ovms::model_version_t&, getVersion, (), ());
   // MOCK_METHOD(ovms::ModelVersionStatus&, getStatus, (), (const));
   // MOCK_METHOD(const std::string&, getBackend, (), ());
   // MOCK_METHOD(size_t, getBatchSize, (), ());
   // MOCK_METHOD(const ovms::tensor_map_t&, getInputsInfo, (), ());
   // MOCK_METHOD(const ovms::tensor_map_t&, getOutputsInfo, (), ());
   MOCK_METHOD(bool, canUnloadInstance, (), (const));
   // MOCK_METHOD(ovms::OVInferRequestsQueue&, getInferRequestsQueue, (), ());
   // MOCK_METHOD(ovms::Status, loadModel, (const ovms::ModelConfig& config), ());
   // MOCK_METHOD(void, unloadModel, (), ());
   // MOCK_METHOD(const ovms::ValidationStatusCode, validate, (const tensorflow::serving::PredictRequest* request), ());
};

}  // namespace

class TestUnloadModel : public ::testing::Test {};

TEST_F(TestUnloadModel, CantUnloadModelWhilePredictPathAcquiredAndLockedInstance) {
   ovms::ModelInstance modelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   ovms::Status status = modelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
   ASSERT_TRUE(status.ok());
   modelInstance.increasePredictRequestsHandlesCount();
   EXPECT_FALSE(modelInstance.canUnloadInstance());
}

TEST_F(TestUnloadModel, CanUnloadModelNotHoldingModelInstanceAtPredictPath) {
   ovms::ModelInstance modelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   ovms::Status status = modelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_TRUE(status.ok());
   ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
   modelInstance.increasePredictRequestsHandlesCount();
   modelInstance.decreasePredictRequestsHandlesCount();
   EXPECT_TRUE(modelInstance.canUnloadInstance());
}

TEST_F(TestUnloadModel, CheckIfCanUnload) {
   MockModelInstance mockModelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, mockModelInstance.getStatus().getState());
   EXPECT_CALL(mockModelInstance, canUnloadInstance())
      .WillOnce(Return(false))
      .WillOnce(Return(true));
   mockModelInstance.unloadModel();
   EXPECT_EQ(ovms::ModelVersionState::END, mockModelInstance.getStatus().getState());
}

class MockModelInstanceCheckingUnloadingState : public ovms::ModelInstance {
public:
   virtual bool canUnloadInstance() const {
      EXPECT_EQ(ovms::ModelVersionState::UNLOADING, getStatus().getState());
      return true;
   }
};

TEST_F(TestUnloadModel, CheckIfStateIsUnloadingDuringUnloading) {
   std::filesystem::path dir = std::filesystem::current_path();
   std::string dummy_model = dir.u8string() + "/src/test/dummy";
   MockModelInstanceCheckingUnloadingState mockModelInstance;
   // TODO dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, mockModelInstance.getStatus().getState());
   mockModelInstance.unloadModel();
   EXPECT_EQ(ovms::ModelVersionState::END, mockModelInstance.getStatus().getState());
}
