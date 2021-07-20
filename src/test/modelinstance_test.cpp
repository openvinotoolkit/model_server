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
#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "../get_model_metadata_impl.hpp"
#include "../modelinstance.hpp"
#include "test_utils.hpp"

using testing::Return;

const std::vector<ovms::ModelVersionState> INFER_QUEUE_SUCCESS_FOR_STATES{
    ovms::ModelVersionState::AVAILABLE};
const std::vector<ovms::ModelVersionState> INFER_QUEUE_FAILURE_FOR_STATES{
    ovms::ModelVersionState::START,
    ovms::ModelVersionState::LOADING,
    ovms::ModelVersionState::UNLOADING,
    ovms::ModelVersionState::END};

namespace {
class MockModelInstanceInState : public ovms::ModelInstance {
    static const ovms::model_version_t UNUSED_VERSION = 987789;

public:
    MockModelInstanceInState(ovms::ModelVersionState state) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {
        status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_VERSION, state);
    }
};

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance() :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {}
    MOCK_METHOD(bool, canUnloadInstance, (), (const));
};

}  // namespace

class TestUnloadModel : public ::testing::Test {};

TEST_F(TestUnloadModel, SuccessfulUnload) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ASSERT_EQ(modelInstance.loadModel(DUMMY_MODEL_CONFIG), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.retireModel();
    EXPECT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
}

TEST_F(TestUnloadModel, CantUnloadModelWhilePredictPathAcquiredAndLockedInstance) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ovms::Status status = modelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    ASSERT_EQ(status, ovms::StatusCode::OK);
    modelInstance.increasePredictRequestsHandlesCount();
    EXPECT_FALSE(modelInstance.canUnloadInstance());
}

TEST_F(TestUnloadModel, CanUnloadModelNotHoldingModelInstanceAtPredictPath) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ovms::Status status = modelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.increasePredictRequestsHandlesCount();
    modelInstance.decreasePredictRequestsHandlesCount();
    EXPECT_TRUE(modelInstance.canUnloadInstance());
}

TEST_F(TestUnloadModel, UnloadWaitsUntilMetadataResponseIsBuilt) {
    static std::thread thread;
    static std::shared_ptr<ovms::ModelInstance> instance;

    class MockModelInstanceTriggeringUnload : public ovms::ModelInstance {
    public:
        MockModelInstanceTriggeringUnload() :
            ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {}
        // This is to trigger model unloading in separate thread during GetModelMetadataImpl::buildResponse call.
        const ovms::tensor_map_t& getInputsInfo() const override {
            thread = std::thread([]() {
                instance->retireModel();
            });
            // We need to wait for thread to start and trigger model unloading
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return ovms::ModelInstance::getInputsInfo();
        }
    };
    instance = std::make_shared<MockModelInstanceTriggeringUnload>();
    ovms::Status status = instance->loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, instance->getStatus().getState());
    tensorflow::serving::GetModelMetadataResponse response;
    EXPECT_EQ(ovms::GetModelMetadataImpl::buildResponse(instance, &response), ovms::StatusCode::OK);
    thread.join();
    EXPECT_EQ(ovms::ModelVersionState::END, instance->getStatus().getState());

    // We expect unload to wait for response building by checking if packed data is correct.
    // If unloadModel didn't wait for building to complete we would have empty input/output map.
    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);
    const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();
    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(inputs.begin()->second.name(), DUMMY_MODEL_INPUT_NAME);
    EXPECT_EQ(outputs.begin()->second.name(), DUMMY_MODEL_OUTPUT_NAME);
}

TEST_F(TestUnloadModel, CheckIfCanUnload) {
    MockModelInstance mockModelInstance;
    mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, mockModelInstance.getStatus().getState());
    EXPECT_CALL(mockModelInstance, canUnloadInstance())
        .WillOnce(Return(false))
        .WillOnce(Return(true));
    mockModelInstance.retireModel();
    EXPECT_EQ(ovms::ModelVersionState::END, mockModelInstance.getStatus().getState());
}

class MockModelInstanceCheckingUnloadingState : public ovms::ModelInstance {
public:
    MockModelInstanceCheckingUnloadingState() :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {}
    virtual bool canUnloadInstance() const {
        EXPECT_EQ(ovms::ModelVersionState::UNLOADING, getStatus().getState());
        return true;
    }
};

TEST_F(TestUnloadModel, CheckIfStateIsUnloadingDuringUnloading) {
    MockModelInstanceCheckingUnloadingState mockModelInstance;
    mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, mockModelInstance.getStatus().getState());
    mockModelInstance.retireModel();
    EXPECT_EQ(ovms::ModelVersionState::END, mockModelInstance.getStatus().getState());
}

class TestLoadModel : public ::testing::Test {};

class MockModelInstanceThrowingFileNotFoundForLoadingCNN : public ovms::ModelInstance {
public:
    MockModelInstanceThrowingFileNotFoundForLoadingCNN() :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {}

protected:
    std::unique_ptr<InferenceEngine::CNNNetwork> loadOVCNNNetworkPtr(const std::string& modelFile) override {
        throw std::runtime_error("File was not found");
        return nullptr;
    }
};

TEST_F(TestLoadModel, CheckIfOVNonExistingXMLFileErrorIsCatched) {
    // Check if handling file removal after file existence was checked
    MockModelInstanceThrowingFileNotFoundForLoadingCNN mockModelInstance;
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    EXPECT_EQ(status, ovms::StatusCode::INTERNAL_ERROR) << status.string();
}

class MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork : public ovms::ModelInstance {
public:
    MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork() :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {}

protected:
    void loadExecutableNetworkPtr(const ovms::plugin_config_t& pluginConfig) override {
        throw std::runtime_error("File was not found");
    }
};

TEST_F(TestLoadModel, CheckIfOVNonExistingBinFileErrorIsCatched) {
    // Check if handling file removal after file existence was checked
    MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork mockModelInstance;
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    EXPECT_EQ(status, ovms::StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingXmlFileReturnsFileInvalid) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);

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
    const ovms::ModelConfig config{
        "NOT_USED_NAME",
        modelPath,  // base path
        "CPU",      // target device
        "1",        // batchsize
        1,          // NIREQ
        false,      // is stateful
        false,      // idle sequence cleanup enabled
        false,      // low latency transformation enabled
        500,        // steteful sequence max number
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingBinFileReturnsFileInvalid) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);

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
    const ovms::ModelConfig config{
        "NOT_USED_NAME",
        modelPath,  // base path
        "CPU",      // target device
        "1",        // batchsize
        1,          // NIREQ
        false,      // is stateful
        false,      // idle sequence cleanup enabled
        false,      // low latency transformation enabled
        500,        // steteful sequence max number
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, SuccessfulLoad) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    EXPECT_EQ(modelInstance.loadModel(DUMMY_MODEL_CONFIG), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestLoadModel, UnSuccessfulLoadWhenNireqTooHigh) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    auto config = DUMMY_MODEL_CONFIG;
    config.setNireq(100000 + 1);
    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::INVALID_NIREQ);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState()) << modelInstance.getStatus().getStateString();
}

class TestReloadModel : public ::testing::Test {};

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoaded) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ASSERT_TRUE(modelInstance.loadModel(DUMMY_MODEL_CONFIG).ok());
    EXPECT_TRUE(modelInstance.reloadModel(DUMMY_MODEL_CONFIG).ok());
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoadedWithChangedModelMapping) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_TRUE(modelInstance.loadModel(config).ok());
    ovms::mapping_config_t mappingOutputs{{"a", "output"}};
    ovms::mapping_config_t mappingInputs{{"b", "input"}};
    config.setMappingInputs(mappingInputs);
    config.setMappingOutputs(mappingOutputs);
    EXPECT_TRUE(modelInstance.reloadModel(config).ok());
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    auto inputsMap = modelInstance.getInputsInfo();
    auto outputsMap = modelInstance.getOutputsInfo();
    EXPECT_EQ(1, inputsMap.size());
    EXPECT_EQ(1, outputsMap.size());
    EXPECT_NE(inputsMap.find("input"), inputsMap.end());
    EXPECT_NE(outputsMap.find("output"), outputsMap.end());
    for (auto& [key, value] : inputsMap) {
        std::cout << key << std::endl;
    }
    for (auto& [key, value] : outputsMap) {
        std::cout << key << std::endl;
    }
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyUnloaded) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ASSERT_TRUE(modelInstance.loadModel(DUMMY_MODEL_CONFIG).ok());
    modelInstance.retireModel();
    ASSERT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
    EXPECT_TRUE(modelInstance.reloadModel(DUMMY_MODEL_CONFIG).ok());
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoadedWithNewBatchSize) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    auto newBatchSize = config.getBatchSize() + 1;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(newBatchSize, {}, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoadedWithNewShape) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.parseShapeParameter("{\"b\": \"auto\"}");
    std::map<std::string, ovms::shape_t> requestShapes = {{"b", {2, 10}}};
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(0, requestShapes, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyUnloadedWithNewBatchSize) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.retireModel();
    ASSERT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
    auto newBatchSize = config.getBatchSize() + 1;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(newBatchSize, {}, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyUnloadedWithNewShape) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.parseShapeParameter("auto");
    std::map<std::string, ovms::shape_t> requestShapes = {{"b", {2, 10}}};
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.retireModel();
    ASSERT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(0, requestShapes, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST(CpuThroughputStreamsNotSpecified, DefaultIsSetForCPU) {
    ovms::ModelConfig config;
    config.setTargetDevice("CPU");
    config.setPluginConfig({});
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 1);
}

TEST(CpuThroughputStreamsNotSpecified, DefaultIsSetForHeteroCPU) {
    ovms::ModelConfig config;
    config.setTargetDevice("HETERO:MYRIAD,CPU");
    config.setPluginConfig({});
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 1);
}

TEST(CpuThroughputStreamsNotSpecified, NotSetForNonCpuDevices) {
    ovms::ModelConfig config;
    config.setPluginConfig({});
    config.setTargetDevice("MYRIAD");
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 0);
    config.setTargetDevice("HDDL");
    pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 0);
    config.setTargetDevice("GPU");
    pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 0);
}
