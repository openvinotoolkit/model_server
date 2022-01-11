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
    MockModelInstanceInState(ov::runtime::Core& ieCore, ovms::ModelVersionState state) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {
        status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_VERSION, state);
    }
};

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::runtime::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}
    MOCK_METHOD(bool, canUnloadInstance, (), (const));
};

}  // namespace

class TestUnloadModel : public ::testing::Test {
protected:
    std::unique_ptr<ov::runtime::Core> ieCore;
    void SetUp() {
        ieCore = std::make_unique<ov::runtime::Core>();
    }
};

TEST_F(TestUnloadModel, SuccessfulUnload) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ASSERT_EQ(modelInstance.loadModel(DUMMY_MODEL_CONFIG), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.retireModel();
    EXPECT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
}

TEST_F(TestUnloadModel, CantUnloadModelWhilePredictPathAcquiredAndLockedInstance) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::Status status = modelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    ASSERT_EQ(status, ovms::StatusCode::OK);
    modelInstance.increasePredictRequestsHandlesCount();
    EXPECT_FALSE(modelInstance.canUnloadInstance());
}

TEST_F(TestUnloadModel, CanUnloadModelNotHoldingModelInstanceAtPredictPath) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
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
        MockModelInstanceTriggeringUnload(ov::runtime::Core& ieCore) :
            ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}
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
    instance = std::make_shared<MockModelInstanceTriggeringUnload>(*ieCore);
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
    MockModelInstance mockModelInstance(*ieCore);
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
    MockModelInstanceCheckingUnloadingState(ov::runtime::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}
    virtual bool canUnloadInstance() const {
        EXPECT_EQ(ovms::ModelVersionState::UNLOADING, getStatus().getState());
        return true;
    }
};

TEST_F(TestUnloadModel, CheckIfStateIsUnloadingDuringUnloading) {
    MockModelInstanceCheckingUnloadingState mockModelInstance(*ieCore);
    mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, mockModelInstance.getStatus().getState());
    mockModelInstance.retireModel();
    EXPECT_EQ(ovms::ModelVersionState::END, mockModelInstance.getStatus().getState());
}

class TestLoadModel : public ::testing::Test {
protected:
    std::unique_ptr<ov::runtime::Core> ieCore;
    void SetUp() {
        ieCore = std::make_unique<ov::runtime::Core>();
    }
};

class MockModelInstanceThrowingFileNotFoundForLoadingCNN : public ovms::ModelInstance {
public:
    MockModelInstanceThrowingFileNotFoundForLoadingCNN(ov::runtime::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}

protected:
    std::shared_ptr<ov::Model> loadOVCNNNetworkPtr(const std::string& modelFile) override {
        throw std::runtime_error("File was not found");
        return nullptr;
    }
};

TEST_F(TestLoadModel, CheckIfOVNonExistingXMLFileErrorIsCatched) {
    // Check if handling file removal after file existence was checked
    MockModelInstanceThrowingFileNotFoundForLoadingCNN mockModelInstance(*ieCore);
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    EXPECT_EQ(status, ovms::StatusCode::INTERNAL_ERROR) << status.string();
}

class MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork : public ovms::ModelInstance {
public:
    MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork(ov::runtime::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}

protected:
    void loadExecutableNetworkPtr(const ovms::plugin_config_t& pluginConfig) override {
        throw std::runtime_error("File was not found");
    }
};

TEST_F(TestLoadModel, CheckIfOVNonExistingBinFileErrorIsCatched) {
    // Check if handling file removal after file existence was checked
    MockModelInstanceThrowingFileNotFoundForLoadingExecutableNetwork mockModelInstance(*ieCore);
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    EXPECT_EQ(status, ovms::StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingXmlFileReturnsFileInvalid) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

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
        500,        // steteful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingBinFileReturnsFileInvalid) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

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
        500,        // steteful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, SuccessfulLoad) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    EXPECT_EQ(modelInstance.loadModel(DUMMY_MODEL_CONFIG), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestLoadModel, UnSuccessfulLoadWhenNireqTooHigh) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    auto config = DUMMY_MODEL_CONFIG;
    config.setNireq(100000 + 1);
    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::INVALID_NIREQ);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState()) << modelInstance.getStatus().getStateString();
}

TEST_F(TestLoadModel, UnSuccessfulLoadWhenLayoutIncorrect) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    auto config = DUMMY_MODEL_CONFIG;
    config.parseLayoutParameter("nchw:nc");
    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::NETWORK_NOT_LOADED);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState()) << modelInstance.getStatus().getStateString();
}

class TestLoadModelWithMapping : public TestLoadModel {
protected:
    void SetUp() override {
        TestLoadModel::SetUp();
        config = DUMMY_MODEL_CONFIG;
        ovms::mapping_config_t mappingOutputs{{"a", "output"}};
        ovms::mapping_config_t mappingInputs{{"b", "input"}};
        config.setMappingInputs(mappingInputs);
        config.setMappingOutputs(mappingOutputs);

        ovms::mapping_config_t realMappingOutputs{{"output", "a"}};
        ovms::mapping_config_t realMappingInputs{{"input", "b"}};
        config.setRealMappingInputs(realMappingInputs);
        config.setRealMappingOutputs(realMappingOutputs);
    }
    ovms::ModelConfig config;
    ovms::shapes_info_map_t shapeMap;
    ovms::layouts_map_t layouts;
};

TEST_F(TestLoadModelWithMapping, SuccessfulLoad) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = ovms::LayoutConfiguration{"LAYOUT_INPUT"};
    layouts["output"] = ovms::LayoutConfiguration{"LAYOUT_OUTPUT"};
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestLoadModelWithMapping, UnSuccessfulLoadOldInputShapeName) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["b"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = "LAYOUT_INPUT";
    layouts["output"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::CONFIG_SHAPE_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());
}

TEST_F(TestLoadModelWithMapping, UnSuccessfulLoadOldInputLayoutName) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["b"] = "LAYOUT_INPUT";
    layouts["output"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());
}

TEST_F(TestLoadModelWithMapping, UnSuccessfulLoadOldOutputLayoutName) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = "LAYOUT_INPUT";
    layouts["a"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());
}

class TestReloadModel : public ::testing::Test {
protected:
    std::unique_ptr<ov::runtime::Core> ieCore;
    void SetUp() {
        ieCore = std::make_unique<ov::runtime::Core>();
    }
};

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoaded) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ASSERT_TRUE(modelInstance.loadModel(DUMMY_MODEL_CONFIG).ok());
    EXPECT_TRUE(modelInstance.reloadModel(DUMMY_MODEL_CONFIG).ok());
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoadedWithChangedModelMapping) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
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
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ASSERT_TRUE(modelInstance.loadModel(DUMMY_MODEL_CONFIG).ok());
    modelInstance.retireModel();
    ASSERT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
    EXPECT_TRUE(modelInstance.reloadModel(DUMMY_MODEL_CONFIG).ok());
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoadedWithNewBatchSize) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    auto newBatchSize = ovms::Dimension(config.getBatchSize().value().getStaticValue() + 1);
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(newBatchSize, {}, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, ReloadWithIncorrectLayoutAndThenFix) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(config.parseLayoutParameter("nchw:nc"), ovms::StatusCode::OK);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::NETWORK_NOT_LOADED);
    ASSERT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState()) << modelInstance.getStatus().getStateString();
    ASSERT_EQ(config.parseLayoutParameter("cn:nc"), ovms::StatusCode::OK);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyLoadedWithNewShape) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.parseShapeParameter("{\"b\": \"auto\"}");
    std::map<std::string, ovms::shape_t> requestShapes = {{"b", {2, 10}}};
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(std::nullopt, requestShapes, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyUnloadedWithNewBatchSize) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchSize(1);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.retireModel();
    ASSERT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
    auto newBatchSize = ovms::Dimension(config.getBatchSize().value().getStaticValue() + 1);
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(newBatchSize, {}, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModel, SuccessfulReloadFromAlreadyUnloadedWithNewShape) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.parseShapeParameter("auto");
    std::map<std::string, ovms::shape_t> requestShapes = {{"b", {2, 10}}};
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.retireModel();
    ASSERT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unloadGuard;
    EXPECT_EQ(modelInstance.reloadModel(std::nullopt, requestShapes, unloadGuard), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

class TestReloadModelWithMapping : public TestReloadModel {
protected:
    void SetUp() override {
        TestReloadModel::SetUp();
        config = DUMMY_MODEL_CONFIG;
        ovms::mapping_config_t mappingOutputs{{"a", "output"}};
        ovms::mapping_config_t mappingInputs{{"b", "input"}};
        config.setMappingInputs(mappingInputs);
        config.setMappingOutputs(mappingOutputs);

        ovms::mapping_config_t realMappingOutputs{{"output", "a"}};
        ovms::mapping_config_t realMappingInputs{{"input", "b"}};
        config.setRealMappingInputs(realMappingInputs);
        config.setRealMappingOutputs(realMappingOutputs);
    }
    ovms::ModelConfig config;
    ovms::shapes_info_map_t shapeMap;
    ovms::layouts_map_t layouts;
};

TEST_F(TestReloadModelWithMapping, SuccessfulReload) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = "LAYOUT_INPUT";
    layouts["output"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModelWithMapping, UnSuccessfulReloadOldInputShapeName) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["b"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = "LAYOUT_INPUT";
    layouts["output"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_SHAPE_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModelWithMapping, UnSuccessfulReloadOldInputLayoutName) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["b"] = "LAYOUT_INPUT";
    layouts["output"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModelWithMapping, UnSuccessfulReloadOldOutputLayoutName) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = "LAYOUT_INPUT";
    layouts["a"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());
}

TEST_F(TestReloadModelWithMapping, ReloadMultipleTimes) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    // initial load
    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    // load with mapping
    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = "LAYOUT_INPUT";
    layouts["output"] = "LAYOUT_OUTPUT";
    config.setLayouts(layouts);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    // load with invalid shape and layouts
    ovms::shapes_info_map_t shapeMapInvalid;
    shapeMapInvalid["b"] = inputShape;
    config.setShapes(shapeMapInvalid);

    ovms::layouts_map_t layoutsInvalid;
    layoutsInvalid["b"] = "LAYOUT_INPUT";
    layoutsInvalid["a"] = "LAYOUT_OUTPUT";
    config.setLayouts(layoutsInvalid);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_SHAPE_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());

    // load with valid config
    config.setShapes(shapeMap);
    config.setLayouts(layouts);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    // load with invalid layouts
    config.setLayouts(layoutsInvalid);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());

    // load with valid config
    config.setShapes(shapeMap);
    config.setLayouts(layouts);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    // load with unknown shape
    ovms::shapes_info_map_t shapeMapUnknown;
    shapeMapUnknown["unknown"] = inputShape;
    config.setShapes(shapeMapUnknown);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_SHAPE_IS_NOT_IN_NETWORK);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());

    // load with valid config
    config.setShapes(shapeMap);
    config.setLayouts(layouts);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    // load with invalid layout
    ovms::layouts_map_t layoutsUnknown;
    layoutsUnknown["input"] = "LAYOUT_INPUT";
    layoutsUnknown["unknown"] = "LAYOUT_OUTPUT";
    config.setLayouts(layoutsUnknown);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_LAYOUT_IS_NOT_IN_NETWORK);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());

    // load with valid config
    config.setShapes(shapeMap);
    config.setLayouts(layouts);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
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
