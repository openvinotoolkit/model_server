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
   MOCK_METHOD(bool, canUnloadInferRequests, (), (const));
   // MOCK_METHOD(ovms::Status, getInferRequestsQueue, (std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueueIn), ());
   // MOCK_METHOD(ovms::Status, loadModel, (const ovms::ModelConfig& config), ());
   // MOCK_METHOD(void, unloadModel, (), ());
   // MOCK_METHOD(const ovms::ValidationStatusCode, validate, (const tensorflow::serving::PredictRequest* request), ());
};

}  // namespace
class GetInferRequestQueueForState : public ::testing::Test {};

TEST_F(GetInferRequestQueueForState, SuccessForAvailable) {
   MockModelInstanceInState mockModelInstance(ovms::ModelVersionState::AVAILABLE);
   std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueue;
   ovms::Status status = mockModelInstance.getInferRequestsQueue(inferRequestsQueue);
   EXPECT_EQ(status, ovms::Status::OK);
}

TEST_F(GetInferRequestQueueForState, NotReadyYetForStart) {
   MockModelInstanceInState mockModelInstance(ovms::ModelVersionState::START);
   std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueue;
   ovms::Status status = mockModelInstance.getInferRequestsQueue(inferRequestsQueue);
   EXPECT_EQ(status, ovms::Status::MODEL_LOADING);
}

TEST_F(GetInferRequestQueueForState, NotReadyYetForLoading) {
   MockModelInstanceInState mockModelInstance(ovms::ModelVersionState::LOADING);
   std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueue;
   ovms::Status status = mockModelInstance.getInferRequestsQueue(inferRequestsQueue);
   EXPECT_EQ(status, ovms::Status::MODEL_LOADING);
}

TEST_F(GetInferRequestQueueForState, ModelRetiredForUnloading) {
   MockModelInstanceInState mockModelInstance(ovms::ModelVersionState::UNLOADING);
   std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueue;
   ovms::Status status = mockModelInstance.getInferRequestsQueue(inferRequestsQueue);
   EXPECT_EQ(status, ovms::Status::MODEL_RETIRED);
}

TEST_F(GetInferRequestQueueForState, ModelRetiredForEnd) {
   MockModelInstanceInState mockModelInstance(ovms::ModelVersionState::END);
   std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueue;
   ovms::Status status = mockModelInstance.getInferRequestsQueue(inferRequestsQueue);
   EXPECT_EQ(status, ovms::Status::MODEL_RETIRED);
}

class TestUnloadModel : public ::testing::Test {};

TEST_F(TestUnloadModel, CantUnloadModelWhileHoldingInferRequestsQueueReference) {
   ovms::ModelInstance modelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   modelInstance.loadModel(DUMMY_MODEL_CONFIG);
   std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueue;
   ovms::Status status = modelInstance.getInferRequestsQueue(inferRequestsQueue);
   ASSERT_EQ(ovms::Status::OK, status);
   ASSERT_EQ(2, inferRequestsQueue.use_count()) << "We expect to have ownership here & in modelInstance";
   EXPECT_FALSE(modelInstance.canUnloadInferRequests());
}

TEST_F(TestUnloadModel, CanUnloadModelNotHoldingInferRequestsQueueReference) {
   ovms::ModelInstance modelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   modelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
   std::shared_ptr<ovms::OVInferRequestsQueue> inferRequestsQueue;
   ovms::Status status = modelInstance.getInferRequestsQueue(inferRequestsQueue);
   ASSERT_EQ(ovms::Status::OK, status);
   ASSERT_EQ(2, inferRequestsQueue.use_count()) << "We expect to have ownership here & in modelInstance";
   EXPECT_FALSE(modelInstance.canUnloadInferRequests());
   inferRequestsQueue.reset();
   EXPECT_TRUE(modelInstance.canUnloadInferRequests());
   modelInstance.unloadModel();
   EXPECT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
}

using testing::Return;

TEST_F(TestUnloadModel, CheckIfCanUnloadInferRequests) {
   MockModelInstance mockModelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, mockModelInstance.getStatus().getState());
   EXPECT_CALL(mockModelInstance, canUnloadInferRequests())
      .WillOnce(Return(false))
      .WillOnce(Return(true));
   mockModelInstance.unloadModel();
   EXPECT_EQ(ovms::ModelVersionState::END, mockModelInstance.getStatus().getState());
}

class MockModelInstanceCheckingUnloadingState : public ovms::ModelInstance {
public:
   virtual bool canUnloadInferRequests() const {
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
