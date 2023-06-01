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
#include "../modelinstanceunloadguard.hpp"
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
    MockModelInstanceInState(ov::Core& ieCore, ovms::ModelVersionState state) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {
        status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_VERSION, state);
    }
};

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}
    MOCK_METHOD(bool, canUnloadInstance, (), (const));
};

}  // namespace

class TestUnloadModel : public ::testing::Test {
protected:
    std::unique_ptr<ov::Core> ieCore;
    void SetUp() {
        ieCore = std::make_unique<ov::Core>();
    }
};

TEST_F(TestUnloadModel, SuccessfulUnload) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ASSERT_EQ(modelInstance.loadModel(DUMMY_MODEL_CONFIG), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    modelInstance.retireModel();
    EXPECT_EQ(ovms::ModelVersionState::END, modelInstance.getStatus().getState());
}

TEST_F(TestUnloadModel, SuccessfulUnloadSaved_Model) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ASSERT_EQ(modelInstance.loadModel(DUMMY_SAVED_MODEL_CONFIG), ovms::StatusCode::OK);
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
        MockModelInstanceTriggeringUnload(ov::Core& ieCore) :
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
    MockModelInstanceCheckingUnloadingState(ov::Core& ieCore) :
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

class TestLoadModel : public TestWithTempDir {
protected:
    std::unique_ptr<ov::Core> ieCore;
    void SetUp() {
        TestWithTempDir::SetUp();
        ieCore = std::make_unique<ov::Core>();
    }
};

class MockModelInstanceWithRTMap : public ovms::ModelInstance {
private:
    ov::RTMap inputRtMap;
    ov::RTMap outputRtMap;

protected:
    std::shared_ptr<ov::Model> loadOVModelPtr(const std::string& modelFile) override {
        auto model = ovms::ModelInstance::loadOVModelPtr(modelFile);
        model->input(DUMMY_MODEL_INPUT_NAME).get_rt_info() = this->inputRtMap;
        model->output(DUMMY_MODEL_OUTPUT_NAME).get_rt_info() = this->outputRtMap;
        return model;
    }

public:
    MockModelInstanceWithRTMap(ov::Core& ieCore, const ov::RTMap& inputRtMap, const ov::RTMap& outputRtMap) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore),
        inputRtMap(inputRtMap),
        outputRtMap(outputRtMap) {
    }
};

// The RTMap is not populated with layout info by Model Optimizer.
// This test ensures that default layout is picked by OVMS and it is "N...".
TEST_F(TestLoadModel, LoadModelWithEmptyRTMapLayoutSetsDefaultLayout) {
    auto inputRtMap = ov::RTMap();
    auto outputRtMap = ov::RTMap();
    MockModelInstanceWithRTMap mockModelInstance(*ieCore, inputRtMap, outputRtMap);
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(mockModelInstance.getInputsInfo().size(), 1);
    ASSERT_EQ(mockModelInstance.getOutputsInfo().size(), 1);
    EXPECT_EQ(mockModelInstance.getInputsInfo().begin()->second->getLayout(), ovms::Layout::getDefaultLayout());
    EXPECT_EQ(mockModelInstance.getOutputsInfo().begin()->second->getLayout(), ovms::Layout::getDefaultLayout());
}

// The RTMap is populated with layout info by Model Optimizer.
// This test ensures the data is read as default when nothing is specified otherwise in ModelConfig.
TEST_F(TestLoadModel, LoadModelWithRTMapLayout) {
    auto inputRtMap = ov::RTMap({{"param", ov::LayoutAttribute(ov::Layout("NC"))}});
    auto outputRtMap = ov::RTMap({{"param", ov::LayoutAttribute(ov::Layout("CN"))}});
    MockModelInstanceWithRTMap mockModelInstance(*ieCore, inputRtMap, outputRtMap);
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(mockModelInstance.getInputsInfo().size(), 1);
    ASSERT_EQ(mockModelInstance.getOutputsInfo().size(), 1);
    EXPECT_EQ(mockModelInstance.getInputsInfo().begin()->second->getLayout(), ovms::Layout("NC"));
    EXPECT_EQ(mockModelInstance.getOutputsInfo().begin()->second->getLayout(), ovms::Layout("CN"));
}

// The RTMap is populated with layout info by Model Optimizer.
// This test ensures the data is not read from .xml file but rather overwritten by --layout parameter.
TEST_F(TestLoadModel, LoadModelWithRTMapLayoutOverwriteByParameter) {
    auto inputRtMap = ov::RTMap({{"param", ov::LayoutAttribute(ov::Layout("NC"))}});
    auto outputRtMap = ov::RTMap({{"param", ov::LayoutAttribute(ov::Layout("CN"))}});
    MockModelInstanceWithRTMap mockModelInstance(*ieCore, inputRtMap, outputRtMap);
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(config.parseLayoutParameter("{\"b\":\"CN\",\"a\":\"NC\"}"), ovms::StatusCode::OK);
    auto status = mockModelInstance.loadModel(config);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    ASSERT_EQ(mockModelInstance.getInputsInfo().size(), 1);
    ASSERT_EQ(mockModelInstance.getOutputsInfo().size(), 1);
    EXPECT_EQ(mockModelInstance.getInputsInfo().begin()->second->getLayout(), ovms::Layout("CN"));
    EXPECT_EQ(mockModelInstance.getOutputsInfo().begin()->second->getLayout(), ovms::Layout("NC"));
}

// The RTMap is populated with layout info by Model Optimizer.
// This test ensures that OVMS refuses to load model with Model Optimizer layout set to invalid layout.
// Invalid layout, meaning the number of dimensions in shape does not match number of dimensions in layout.
TEST_F(TestLoadModel, LoadModelWithRTMapParameterInputLayoutIncompatible) {
    auto inputRtMap = ov::RTMap({{"param", ov::LayoutAttribute(ov::Layout("NCHW"))}});
    auto outputRtMap = ov::RTMap();
    MockModelInstanceWithRTMap mockModelInstance(*ieCore, inputRtMap, outputRtMap);
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NOT_LOADED) << status.string();
}

TEST_F(TestLoadModel, LoadModelWithRTMapParameterOutputLayoutIncompatible) {
    auto inputRtMap = ov::RTMap();
    auto outputRtMap = ov::RTMap({{"param", ov::LayoutAttribute(ov::Layout("NCHW"))}});
    MockModelInstanceWithRTMap mockModelInstance(*ieCore, inputRtMap, outputRtMap);
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NOT_LOADED) << status.string();
}

class MockModelInstanceThrowingFileNotFoundForLoadingCNN : public ovms::ModelInstance {
public:
    MockModelInstanceThrowingFileNotFoundForLoadingCNN(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}

protected:
    std::shared_ptr<ov::Model> loadOVModelPtr(const std::string& modelFile) override {
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

class MockModelInstanceThrowingFileNotFoundForLoadingCompiledModel : public ovms::ModelInstance {
public:
    MockModelInstanceThrowingFileNotFoundForLoadingCompiledModel(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {}

protected:
    void loadCompiledModelPtr(const ovms::plugin_config_t& pluginConfig) override {
        throw std::runtime_error("File was not found");
    }
};

TEST_F(TestLoadModel, CheckIfOVNonExistingBinFileErrorIsCatched) {
    // Check if handling file removal after file existence was checked
    MockModelInstanceThrowingFileNotFoundForLoadingCompiledModel mockModelInstance(*ieCore);
    auto status = mockModelInstance.loadModel(DUMMY_MODEL_CONFIG);
    EXPECT_EQ(status, ovms::StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingXmlFileReturnsFileInvalid) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    const std::string modelPath = directoryPath + "/test_load_model";
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
        500,        // stateful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, CheckIfNonExistingBinFileReturnsFileInvalid) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    const std::string modelPath = directoryPath + "/test_load_model";
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
        500,        // stateful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID) << status.string();
}

TEST_F(TestLoadModel, CheckMultipleFormatsHandling) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    const std::string modelPath = directoryPath + "/test_multiple_models";
    std::filesystem::create_directories(modelPath);
    ovms::model_version_t version = 1;
    const std::string versionDirectoryPath = modelPath + "/" + std::to_string(version);
    if (!std::filesystem::exists(versionDirectoryPath)) {
        ASSERT_TRUE(std::filesystem::create_directories(versionDirectoryPath));
    }
    {
        std::ofstream xmlFile{versionDirectoryPath + "/model.xml"};
        xmlFile << "NOT_NEEDED_CONTENT" << std::endl;
    }
    {
        std::ofstream binFile{versionDirectoryPath + "/model.bin"};
        binFile << "NOT_NEEDED_CONTENT" << std::endl;
    }
    {
        std::ofstream onnxFile{versionDirectoryPath + "/model.onnx"};
        onnxFile << "NOT_NEEDED_CONTENT" << std::endl;
    }
    {
        std::ofstream savedModelFile{versionDirectoryPath + "/saved_model.pb"};
        savedModelFile << "NOT_NEEDED_CONTENT" << std::endl;
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
        500,        // stateful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    auto model_files = modelInstance.getModelFiles();
    EXPECT_EQ(model_files.front(), directoryPath + "/test_multiple_models/1/model.xml");
}

TEST_F(TestLoadModel, CheckSavedModelHandling) {
    ovms::ModelInstance modelInstance("saved-model", UNUSED_MODEL_VERSION, *ieCore);

    const std::string modelPath = directoryPath + "/test_saved_model";
    std::filesystem::create_directories(modelPath);
    ovms::model_version_t version = 1;
    const std::string versionDirectoryPath = modelPath + "/" + std::to_string(version);
    if (!std::filesystem::exists(versionDirectoryPath)) {
        ASSERT_TRUE(std::filesystem::create_directories(versionDirectoryPath));
    }
    {
        std::ofstream savedModelFile{versionDirectoryPath + "/saved_model.pb"};
        savedModelFile << "NOT_NEEDED_CONTENT" << std::endl;
    }
    const ovms::ModelConfig config{
        "saved-model",
        modelPath,  // base path
        "CPU",      // target device
        "1",        // batchsize
        1,          // NIREQ
        false,      // is stateful
        false,      // idle sequence cleanup enabled
        false,      // low latency transformation enabled
        500,        // stateful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    auto model_files = modelInstance.getModelFiles();

    EXPECT_EQ(model_files.front(), directoryPath + "/test_saved_model/1/");
}

TEST_F(TestLoadModel, CheckTFModelHandling) {
    ovms::ModelInstance modelInstance("tf", UNUSED_MODEL_VERSION, *ieCore);

    const std::string modelPath = directoryPath + "/test_tf";
    std::filesystem::create_directories(modelPath);
    ovms::model_version_t version = 1;
    const std::string versionDirectoryPath = modelPath + "/" + std::to_string(version);
    if (!std::filesystem::exists(versionDirectoryPath)) {
        ASSERT_TRUE(std::filesystem::create_directories(versionDirectoryPath));
    }
    {
        std::ofstream savedModelFile{versionDirectoryPath + "/model.pb"};
        savedModelFile << "NOT_NEEDED_CONTENT" << std::endl;
    }
    const ovms::ModelConfig config{
        "saved-model",
        modelPath,  // base path
        "CPU",      // target device
        "1",        // batchsize
        1,          // NIREQ
        false,      // is stateful
        false,      // idle sequence cleanup enabled
        false,      // low latency transformation enabled
        500,        // stateful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    auto model_files = modelInstance.getModelFiles();
    ASSERT_NE(model_files.size(), 0);
    EXPECT_EQ(model_files.front(), directoryPath + "/test_tf/1/model.pb");
}

TEST_F(TestLoadModel, CheckONNXModelHandling) {
    ovms::ModelInstance modelInstance("tf", UNUSED_MODEL_VERSION, *ieCore);

    const std::string modelPath = directoryPath + "/test_onnx";
    std::filesystem::create_directories(modelPath);
    ovms::model_version_t version = 1;
    const std::string versionDirectoryPath = modelPath + "/" + std::to_string(version);
    if (!std::filesystem::exists(versionDirectoryPath)) {
        ASSERT_TRUE(std::filesystem::create_directories(versionDirectoryPath));
    }
    {
        std::ofstream savedModelFile{versionDirectoryPath + "/my-model.onnx"};
        savedModelFile << "NOT_NEEDED_CONTENT" << std::endl;
    }
    const ovms::ModelConfig config{
        "saved-model",
        modelPath,  // base path
        "CPU",      // target device
        "1",        // batchsize
        1,          // NIREQ
        false,      // is stateful
        false,      // idle sequence cleanup enabled
        false,      // low latency transformation enabled
        500,        // stateful sequence max number,
        "",         // cache dir
        version,    // version
        modelPath,  // local path
    };
    auto status = modelInstance.loadModel(config);
    auto model_files = modelInstance.getModelFiles();

    EXPECT_EQ(model_files.front(), directoryPath + "/test_onnx/1/my-model.onnx");
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
    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::MODEL_NOT_LOADED);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState()) << modelInstance.getStatus().getStateString();
}

TEST_F(TestLoadModel, UnSuccessfulLoadWhenInputLayoutIncompatible) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    auto config = DUMMY_MODEL_CONFIG;
    config.parseLayoutParameter("{\"b\":\"nchw\"}");
    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::MODEL_NOT_LOADED);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState()) << modelInstance.getStatus().getStateString();
}

TEST_F(TestLoadModel, UnSuccessfulLoadWhenOutputLayoutIncompatible) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    auto config = DUMMY_MODEL_CONFIG;
    config.parseLayoutParameter("{\"a\":\"nchw\"}");
    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::MODEL_NOT_LOADED);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState()) << modelInstance.getStatus().getStateString();
}

TEST_F(TestLoadModel, SuccessfulLoadDummyAllDimensionsAny) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(-1,-1)"), ovms::StatusCode::OK);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    ASSERT_NE(modelInstance.getInputsInfo().begin(), modelInstance.getInputsInfo().end());
    ASSERT_EQ(modelInstance.getInputsInfo().begin()->second->getShape(), ovms::Shape(
                                                                             {ovms::Dimension::any(), ovms::Dimension::any()}));
}

TEST_F(TestLoadModel, SuccessfulLoadDummyDimensionRanges) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setBatchingParams("0");
    ASSERT_EQ(config.parseShapeParameter("(20:30,40:50)"), ovms::StatusCode::OK);
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    ASSERT_NE(modelInstance.getInputsInfo().begin(), modelInstance.getInputsInfo().end());
    ASSERT_EQ(modelInstance.getInputsInfo().begin()->second->getShape(), ovms::Shape(
                                                                             {{20, 30}, {40, 50}}));
}

TEST_F(TestLoadModel, CorrectNumberOfStreamsSet) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    config.setPluginConfig({{"CPU_THROUGHPUT_STREAMS", "6"}});
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    ASSERT_EQ(modelInstance.getNumOfStreams(), 6);
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
    ovms::layout_configurations_map_t layouts;
};

TEST_F(TestLoadModelWithMapping, SuccessfulLoad) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = ovms::LayoutConfiguration{"NC"};
    layouts["output"] = ovms::LayoutConfiguration{"NC"};
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    EXPECT_EQ(modelInstance.getInputsInfo().begin()->second->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::NO_PROCESSING);
    EXPECT_EQ(modelInstance.getInputsInfo().begin()->second->getShape(), ovms::Shape({1, 10}));
    EXPECT_EQ(modelInstance.getOutputsInfo().begin()->second->getShape(), ovms::Shape({1, 10}));
}

TEST_F(TestLoadModelWithMapping, SuccessfulLoadBytesEncoded) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = ovms::LayoutConfiguration{"NC"};
    layouts["output"] = ovms::LayoutConfiguration{"NC"};
    config.setLayouts(layouts);
    config.parseShapeParameter("(1,10,10)");

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    EXPECT_EQ(modelInstance.getInputsInfo().begin()->second->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::NO_PROCESSING);
    EXPECT_EQ(modelInstance.getInputsInfo().begin()->second->getShape(), ovms::Shape({1, 10}));
    EXPECT_EQ(modelInstance.getOutputsInfo().begin()->second->getShape(), ovms::Shape({1, 10}));
}

TEST_F(TestLoadModelWithMapping, SuccessfulLoadChangingModelLayout) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    layouts["input"] = ovms::LayoutConfiguration{"CN", "NC"};
    layouts["output"] = ovms::LayoutConfiguration{"CN", "NC"};
    config.setLayouts(layouts);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());
    EXPECT_EQ(modelInstance.getInputsInfo().begin()->second->getShape(), ovms::Shape({10, 1}));
    EXPECT_EQ(modelInstance.getOutputsInfo().begin()->second->getShape(), ovms::Shape({10, 1}));
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
    std::unique_ptr<ov::Core> ieCore;
    void SetUp() {
        ieCore = std::make_unique<ov::Core>();
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
    ASSERT_EQ(modelInstance.loadModel(config), ovms::StatusCode::MODEL_NOT_LOADED);
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
    ovms::layout_configurations_map_t layouts;
};

TEST_F(TestReloadModelWithMapping, SuccessfulReload) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);

    EXPECT_EQ(modelInstance.loadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    ovms::ShapeInfo inputShape{ovms::FIXED, {1, 20}};
    shapeMap["input"] = inputShape;
    config.setShapes(shapeMap);

    layouts["input"] = "NC";
    layouts["output"] = "NC";
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

    layouts["input"] = "NC";
    layouts["output"] = "NC";
    config.setLayouts(layouts);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    // load with invalid shape and layouts
    ovms::shapes_info_map_t shapeMapInvalid;
    shapeMapInvalid["b"] = inputShape;
    config.setShapes(shapeMapInvalid);

    ovms::layout_configurations_map_t layoutsInvalid;
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
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_SHAPE_IS_NOT_IN_MODEL);
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance.getStatus().getState());

    // load with valid config
    config.setShapes(shapeMap);
    config.setLayouts(layouts);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance.getStatus().getState());

    // load with invalid layout
    ovms::layout_configurations_map_t layoutsUnknown;
    layoutsUnknown["input"] = "LAYOUT_INPUT";
    layoutsUnknown["unknown"] = "LAYOUT_OUTPUT";
    config.setLayouts(layoutsUnknown);
    EXPECT_EQ(modelInstance.reloadModel(config), ovms::StatusCode::CONFIG_LAYOUT_IS_NOT_IN_MODEL);
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
    EXPECT_EQ(pluginConfig.count("PERFORMANCE_HINT"), 1);
}

TEST(CpuThroughputStreamsNotSpecified, NotSetForHeteroCPU) {
    ovms::ModelConfig config;
    config.setTargetDevice("HETERO:MYRIAD,CPU");
    config.setPluginConfig({});
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 0);
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

TEST(CpuThroughputStreamsNotSpecified, NotSetWhenPerfHintSpecified) {
    ovms::ModelConfig config;
    config.setPluginConfig({{"PERFORMANCE_HINT", "LATENCY"}});
    config.setTargetDevice("CPU");
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 0);
    config.setPluginConfig({{"PERFORMANCE_HINT", "THROUGHTPUT"}});
    pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("CPU_THROUGHPUT_STREAMS"), 0);
}

TEST(CpuThroughputNotSpecified, AffinityWithoutHint) {
    ovms::ModelConfig config;
    config.setPluginConfig({{"AFFINITY", "NUMA"}});
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("PERFORMANCE_HINT"), 1);
    EXPECT_EQ(pluginConfig.count("AFFINITY"), 1);
}

TEST(CpuThroughputNotSpecified, AffinityWithNumStreams) {
    ovms::ModelConfig config;
    config.setPluginConfig({{"NUM_STREAMS", "4"}, {"AFFINITY", "NUMA"}});
    ovms::plugin_config_t pluginConfig = ovms::ModelInstance::prepareDefaultPluginConfig(config);
    EXPECT_EQ(pluginConfig.count("PERFORMANCE_HINT"), 0);
    EXPECT_EQ(pluginConfig.count("AFFINITY"), 1);
    EXPECT_EQ(pluginConfig.count("NUM_STREAMS"), 1);
}

TEST(TensorMap, TestProcessingHintFromShape) {
    auto servableInputs = ovms::tensor_map_t({
        {"Input_FP32_1_224_224_3_NHWC",
            std::make_shared<ovms::TensorInfo>("Input_FP32_1_224_224_3_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3})},
        {"Input_U8_1_3_NCHW",
            std::make_shared<ovms::TensorInfo>("Input_U8_1_3_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3})},
        {"Input_U8_3_N",
            std::make_shared<ovms::TensorInfo>("Input_U8_3_N", ovms::Precision::U8, ovms::shape_t{3})},
        {"Input_U8_-1_N",
            std::make_shared<ovms::TensorInfo>("Input_U8_-1_N", ovms::Precision::U8, ovms::Shape{ovms::Dimension::any()})},
    });
    auto servableOutputs = ovms::tensor_map_t({{"Output_U8_-1_-1_N?",
                                                   std::make_shared<ovms::TensorInfo>("Output_U8_-1_-1_N?", ovms::Precision::U8, ovms::Shape{ovms::Dimension::any(), ovms::Dimension::any()})},
        {"Output_U8_-1_-1_N?_string",
            std::make_shared<ovms::TensorInfo>("Output_U8_-1_-1_N?_string", ovms::Precision::U8, ovms::Shape{ovms::Dimension::any(), ovms::Dimension::any()})},
        {"Output_FP32_-1_-1_N?_string",
            std::make_shared<ovms::TensorInfo>("Output_FP32_-1_-1_N?_string", ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), ovms::Dimension::any()})}});

    EXPECT_EQ(servableInputs["Input_FP32_1_224_224_3_NHWC"]->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::IMAGE);
    EXPECT_EQ(servableInputs["Input_U8_1_3_NCHW"]->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::STRING_2D_U8);
    EXPECT_EQ(servableInputs["Input_U8_3_N"]->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::NO_PROCESSING);  // due to static dimension
    EXPECT_EQ(servableInputs["Input_U8_-1_N"]->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::STRING_1D_U8);
    EXPECT_EQ(servableOutputs["Output_U8_-1_-1_N?"]->getPostProcessingHint(), ovms::TensorInfo::ProcessingHint::NO_PROCESSING);           // due to no suffix
    EXPECT_EQ(servableOutputs["Output_U8_-1_-1_N?_string"]->getPostProcessingHint(), ovms::TensorInfo::ProcessingHint::STRING_2D_U8);     // due to suffix
    EXPECT_EQ(servableOutputs["Output_FP32_-1_-1_N?_string"]->getPostProcessingHint(), ovms::TensorInfo::ProcessingHint::NO_PROCESSING);  // no processing due to not being U8
}

TEST(TensorMap, TestProcessingHintFromShape_Demultiplexer) {
    auto servableInputs = ovms::tensor_map_t({
        {"Input_FP32_1_1_224_224_3_NHWC",
            std::make_shared<ovms::TensorInfo>("Input_FP32_1_1_224_224_3_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3})->createCopyWithDemultiplexerDimensionPrefix(1)},
        {"Input_U8_1_1_3_NCHW",
            std::make_shared<ovms::TensorInfo>("Input_U8_1_1_3_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3})->createCopyWithDemultiplexerDimensionPrefix(1)},
        {"Input_U8_1_3_N",
            std::make_shared<ovms::TensorInfo>("Input_U8_1_3_N", ovms::Precision::U8, ovms::shape_t{3})->createCopyWithDemultiplexerDimensionPrefix(1)},
    });

    EXPECT_EQ(servableInputs["Input_FP32_1_1_224_224_3_NHWC"]->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::IMAGE);
    EXPECT_EQ(servableInputs["Input_U8_1_1_3_NCHW"]->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::NO_PROCESSING);  // due to demultiplexer
    EXPECT_EQ(servableInputs["Input_U8_1_3_N"]->getPreProcessingHint(), ovms::TensorInfo::ProcessingHint::NO_PROCESSING);       // due to demultiplexer
}
