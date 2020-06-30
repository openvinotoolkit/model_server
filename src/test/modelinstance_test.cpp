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
#include <cstdio>
#include <fstream>
#include <filesystem>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stdlib.h>

#include "../modelinstance.hpp"
#include "test_utils.hpp"

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
   MOCK_METHOD(bool, canUnloadInstance, (), (const));
};

}  // namespace

class TestUnloadModel : public ::testing::Test {};

TEST_F(TestUnloadModel, CantUnloadModelWhilePredictPathAcquiredAndLockedInstance) {
   ovms::ModelInstance modelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   ovms::Status status = modelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
   ASSERT_EQ(status, ovms::StatusCode::OK);
   modelInstance.increasePredictRequestsHandlesCount();
   EXPECT_FALSE(modelInstance.canUnloadInstance());
}

TEST_F(TestUnloadModel, CanUnloadModelNotHoldingModelInstanceAtPredictPath) {
   ovms::ModelInstance modelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   ovms::Status status = modelInstance.loadModel(DUMMY_MODEL_CONFIG);
   ASSERT_EQ(status, ovms::StatusCode::OK);
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

class TestLoadModel : public ::testing::Test {};

class MockModelInstanceThrowingFileNotFoundForLoadingCNN : public ovms::ModelInstance {
   protected:
   std::unique_ptr<InferenceEngine::CNNNetwork> loadOVCNNNetworkPtr(const std::string& modelFile) override {
      throw std::runtime_error("File was not found");
      return nullptr;
   }
};

TEST_F(TestLoadModel, CheckIfOVNonExistingXMLFileErrorIsCatched) {
   // Check if handling file removal after file existence was checked
   MockModelInstanceThrowingFileNotFoundForLoadingCNN mockModelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
   // TODO we do validation before passing file to OV need to test that too. Hence INTERNAL_ERROR not FILE_INVALID
   EXPECT_EQ(status, ovms::StatusCode::INTERNAL_ERROR) << status.string();
}

class MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork : public ovms::ModelInstance {
   protected:
   void loadExecutableNetworkPtr(const ovms::plugin_config_t& pluginConfig) override {
      throw std::runtime_error("File was not found");
   }
};

TEST_F(TestLoadModel, CheckIfOVNonExistingBinFileErrorIsCatched) {
   // Check if handling file removal after file existence was checked
   MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork mockModelInstance;
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
   EXPECT_EQ(status, ovms::StatusCode::INTERNAL_ERROR) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingXmlFileReturnsFileInvalid) {
   ovms::ModelInstance mockModelInstance;

   const std::string modelPath = "/tmp/test_load_model";
   std::filesystem::create_directories(modelPath);
   ovms::model_version_t version = 1;
   const std::string versionDirectoryPath = modelPath + "/" + std::to_string(version);
   if (!std::filesystem::exists(versionDirectoryPath)) {
      ASSERT_TRUE(std::filesystem::create_directories(versionDirectoryPath));
   }
   {
      std::ofstream binFile{versionDirectoryPath + "/NOT_USED_NAME.bin"};
      binFile << "NOT_NEEDED_CONTENT" << std::endl;
   }
   auto xmlFilename = (versionDirectoryPath + "/NOT_USED_NAME.xml");
   if (std::filesystem::exists(xmlFilename)) {
      ASSERT_EQ(0, std::remove(xmlFilename.c_str()));
   }
   const ovms::ModelConfig config {
      "NOT_USED_NAME",
      modelPath,
      "CPU",  // backend
      1,  // batchsize
      1,  // NIREQ
      version,  // version
   };
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   auto status = mockModelInstance.loadModel(config);
   EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingBinFileReturnsFileInvalid) {
   ovms::ModelInstance mockModelInstance;

   const std::string modelPath = "/tmp/test_load_model";
   std::filesystem::create_directories(modelPath);
   ovms::model_version_t version = 1;
   const std::string versionDirectoryPath = modelPath + "/" + std::to_string(version);
   if (!std::filesystem::exists(versionDirectoryPath)) {
      ASSERT_TRUE(std::filesystem::create_directories(versionDirectoryPath));
   }
   {
      std::ofstream xmlFile{versionDirectoryPath + "/NOT_USED_NAME.xml"};
      xmlFile << "NOT_NEEDED_CONTENT" << std::endl;
   }
   auto binFilename = (versionDirectoryPath + "/NOT_USED_NAME.bin");
   if (std::filesystem::exists(binFilename)) {
      ASSERT_EQ(0, std::remove(binFilename.c_str()));
   }
   const ovms::ModelConfig config {
      "NOT_USED_NAME",
      modelPath,
      "CPU",  // backend
      1,  // batchsize
      1,  // NIREQ
      version,  // version
   };
   // dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   auto status = mockModelInstance.loadModel(config);
   EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, SuccesfullLoad) {
   std::filesystem::path dir = std::filesystem::current_path();
   std::string dummy_model = dir.u8string() + "/src/test/dummy";
   ovms::ModelInstance modelInstance;
   // TODO dirty hack to avoid initializing config
   setenv("NIREQ", "1", 1);
   EXPECT_EQ(modelInstance.loadModel(DUMMY_MODEL_CONFIG), ovms::StatusCode::OK);
   EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}
