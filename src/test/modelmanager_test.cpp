//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../model.hpp"
#include "../modelmanager.hpp"
#include "../prediction_service_utils.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"

using testing::_;
using testing::ContainerEq;
using testing::Return;
using testing::ReturnRef;
using testing::UnorderedElementsAre;

namespace {
const char* config_1_model = R"({
   "model_config_list": [
    {
      "config": {
        "name": "resnet",
        "base_path": "/tmp/models/dummy1",
        "target_device": "CPU",
        "model_version_policy": {"all": {}}
      }
   }]
})";

const char* config_2_models = R"({
   "model_config_list": [
    {
      "config": {
        "name": "resnet",
        "base_path": "/tmp/models/dummy1",
        "target_device": "CPU",
        "model_version_policy": {"all": {}}
      }
    },
    {
      "config": {
        "name": "alpha",
        "base_path": "/tmp/models/dummy2",
        "target_device": "CPU",
        "model_version_policy": {"all": {}}
      }
    }]
})";

const std::string FIRST_MODEL_NAME = "resnet";
const std::string SECOND_MODEL_NAME = "alpha";

const std::string model_1_path = "/tmp/models/dummy1/1";
const std::string model_2_path = "/tmp/models/dummy2/2";

class MockModel : public ovms::Model {
public:
    MockModel() :
        Model("MOCK_NAME", false, nullptr) {}
    MOCK_METHOD(ovms::Status, addVersion, (const ovms::ModelConfig&, InferenceEngine::Core&, ov::runtime::Core&), (override));
};

std::shared_ptr<MockModel> modelMock;

}  // namespace

class MockModelManager : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return modelMock;
    }
};

TEST(ModelManager, ConfigParseNoModels) {
    std::string configFile = createConfigFileWithContent("{ \"model_config_list\": [ ] }\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST(ModelManager, WrongConfigFile) {
    std::string configFile = "123/tmp/not_a_valid_file_name";
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::CONFIG_FILE_INVALID);
}

TEST(ModelManager, ConfigParseEmpty) {
    std::string configFile = createConfigFileWithContent("\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST(ModelManager, ConfigNotAJson) {
    std::string configFile = createConfigFileWithContent("abcdfgh");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST(ModelManager, ConfigParseEmptyJson) {
    std::string configFile = createConfigFileWithContent("{}\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST(ModelManager, ConfigParseNodeConfigWithoutNameKey) {
    const char* configWithoutNameKey = R"({
       "model_config_list": [
       {
          "config": {
            "base_path": "/tmp/models/dummy2"
          }
       }]
    })";

    std::string configFile = createConfigFileWithContent(configWithoutNameKey);
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST(ModelManager, ConfigParseNodeConfigWihoutBasePathKey) {
    const char* configWithoutBasePathKey = R"({
       "model_config_list": [
       {
          "config": {
            "name": "alpha"
          }
       }]
    })";

    std::string configFile = createConfigFileWithContent(configWithoutBasePathKey);
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST(ModelManager, parseConfigWhenPipelineDefinitionMatchSchema) {
    const char* configWithPipelineDefinitionMatchSchema = R"({
        "model_config_list": [
            {
                "config": {
                    "name": "alpha",
                    "base_path": "/tmp/models/dummy1"
                }
            },
            {
                "config": {
                    "name": "beta",
                    "base_path": "/tmp/models/dummy2"
                }
            }
        ],
        "pipeline_config_list": 
        [
            {
                "name": "ensemble_name1", 
                "inputs": ["in"], 
                "outputs": [{"a":{"node_name": "beta","data_item": "text"}}], 
                "nodes": [  
                    { 
                        "name": "alpha", 
                        "model_name": "dummy",
                        "type": "DL model", 
                        "inputs": [{"a":{"node_name": "input","data_item": "in"}}], 
                        "outputs": [{"data_item": "prob","alias": "prob"}] 
                    }, 
                    { 
                        "name": "beta", 
                        "model_name": "dummy",
                        "type": "DL model",
                        "inputs": [{"a":{"node_name": "alpha","data_item": "prob"}}],
                        "outputs": [{"data_item": "text","alias": "text"}] 
                    }
                ]
            }
        ]
    })";

    std::string configFile = "/tmp/ovms_config_file.json";
    createConfigFileWithContent(configWithPipelineDefinitionMatchSchema, configFile);
    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
    modelMock.reset();
}

void setupModelsDirs() {
    std::filesystem::create_directory("/tmp/models");
    std::filesystem::create_directory("/tmp/models/dummy1");
    std::filesystem::create_directory("/tmp/models/dummy2");
}

TEST(ModelManager, configRelodNotNeededManyThreads) {
    std::string configFile = "/tmp/config.json";

    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;
    setupModelsDirs();
    createConfigFileWithContent(config_2_models, configFile);
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    int numberOfThreads = 10;
    std::vector<std::thread> threads;
    std::function<void()> func = [&manager]() {
        bool isNeeded = false;
        manager.configFileReloadNeeded(isNeeded);
        EXPECT_EQ(isNeeded, false);
    };

    for (int i = 0; i < numberOfThreads; i++) {
        threads.push_back(std::thread(func));
    }

    for (auto& thread : threads) {
        thread.join();
    }
    manager.join();
    modelMock.reset();
}

TEST(ModelManager, configRelodNeededManyThreads) {
    std::string configFile = "/tmp/config.json";

    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;
    setupModelsDirs();
    createConfigFileWithContent(config_2_models, configFile);
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    int numberOfThreads = 10;
    std::vector<std::thread> threads;

    bool isNeeded = false;
    std::function<void()> func = [&manager, &isNeeded]() {
        manager.configFileReloadNeeded(isNeeded);
        EXPECT_EQ(isNeeded, true);
    };

    manager.configFileReloadNeeded(isNeeded);
    EXPECT_EQ(isNeeded, false);
    createConfigFileWithContent(config_2_models, configFile);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    for (int i = 0; i < numberOfThreads; i++) {
        threads.push_back(std::thread(func));
    }

    for (auto& thread : threads) {
        thread.join();
    }
    manager.join();
    modelMock.reset();
}

TEST(ModelManager, configReloadNeededChange) {
    std::string configFile = "/tmp/config.json";

    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    createConfigFileWithContent(config_2_models, configFile);
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    bool isNeeded = false;
    manager.configFileReloadNeeded(isNeeded);
    EXPECT_EQ(isNeeded, false);

    createConfigFileWithContent(config_2_models, configFile);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    manager.configFileReloadNeeded(isNeeded);
    EXPECT_EQ(isNeeded, true);

    manager.join();
    modelMock.reset();
}

TEST(ModelManager, loadConfigManyThreads) {
    std::string configFile = "/tmp/config.json";

    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;
    setupModelsDirs();
    createConfigFileWithContent(config_2_models, configFile);

    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    int numberOfThreads = 10;
    std::vector<std::thread> threads;
    std::function<void()> func = [&manager, configFile]() {
        EXPECT_EQ(manager.loadConfig(configFile), ovms::StatusCode::OK);
    };

    for (int i = 0; i < numberOfThreads; i++) {
        threads.push_back(std::thread(func));
    }

    for (auto& thread : threads) {
        thread.join();
    }
    manager.join();
    modelMock.reset();
}

TEST(ModelManager, configReloadNeededBeforeConfigLoad) {
    std::string configFile = "/tmp/config.json";

    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    createConfigFileWithContent(config_2_models, configFile);
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    bool isNeeded = false;
    manager.configFileReloadNeeded(isNeeded);
    EXPECT_EQ(isNeeded, false);

    createConfigFileWithContent(config_2_models, configFile);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    manager.configFileReloadNeeded(isNeeded);
    EXPECT_EQ(isNeeded, true);

    manager.loadConfig(configFile);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    manager.configFileReloadNeeded(isNeeded);
    EXPECT_EQ(isNeeded, false);

    manager.join();
    modelMock.reset();
}

TEST(ModelManager, parseConfigWhenOnlyPipelineDefinitionProvided) {
    const char* configWithOnlyPipelineDefinitionProvided = R"({
    "pipeline_config_list": 
    {
    "name": "ensemble_name1", 
    "inputs": ["in"], 
    "outputs": [{"out1": {"node_name": "beta","data_item": "text"}}], 
    "nodes": [  
    { 
    "name": "alpha", 
    "type": "DL Model", 
    "inputs": [{"data": {"node_name": "input","data_item": "in"}}], 
    "outputs": [{"data_item": "prob","alias": "prob"}] 
    }, 
    { 
    "name": "beta", 
    "type": "DL Model",
    "inputs": [{"data": {"node_name": "alpha","data_item": "prob"}}],
    "outputs": [{"data_item": "text","alias": "text"}] 
    }]}})";

    std::string configFile = "/tmp/ovms_config_file.json";
    createConfigFileWithContent(configWithOnlyPipelineDefinitionProvided, configFile);
    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
    manager.join();
    modelMock.reset();
}

TEST(ModelManager, ReadsVersionsFromDisk) {
    const std::string path = "/tmp/test_model/";

    for (auto i : {1, 5, 8, 10}) {
        std::filesystem::create_directories(path + std::to_string(i));
    }

    std::filesystem::create_directories(path + "unknown_dir11");  // invalid version directory
    ovms::model_versions_t versions;
    std::shared_ptr<ovms::FileSystem> fs = std::make_shared<ovms::LocalFileSystem>();

    auto status = ovms::ModelManager::getInstance().readAvailableVersions(fs, path, versions);

    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_THAT(versions, ::testing::UnorderedElementsAre(1, 5, 8, 10));
}

TEST(ModelManager, PathEscapeError1) {
    const std::string path = "/tmp/../test_model/";

    ovms::model_versions_t versions;
    std::shared_ptr<ovms::FileSystem> fs = std::make_shared<ovms::LocalFileSystem>();

    auto status = ovms::ModelManager::getInstance().readAvailableVersions(fs, path, versions);

    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST(ModelManager, PathEscapeError2) {
    const std::string path = "../tmp/test_model/";

    ovms::model_versions_t versions;
    std::shared_ptr<ovms::FileSystem> fs = std::make_shared<ovms::LocalFileSystem>();

    auto status = ovms::ModelManager::getInstance().readAvailableVersions(fs, path, versions);

    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST(ModelManager, ReadVersionsInvalidPath) {
    const std::string path = "/tmp/inexisting_path/8bt4kv";

    try {
        std::filesystem::remove(path);
    } catch (const std::filesystem::filesystem_error&) {
    }

    std::vector<ovms::model_version_t> versions;
    std::shared_ptr<ovms::FileSystem> fs = std::make_shared<ovms::LocalFileSystem>();
    auto status = ovms::ModelManager::getInstance().readAvailableVersions(fs, path, versions);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST(ModelManager, StartFromFile) {
    std::filesystem::create_directories(model_1_path);
    std::filesystem::create_directories(model_2_path);
    std::string fileToReload = "/tmp/ovms_config_file1.json";
    createConfigFileWithContent(config_1_model, fileToReload);
    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    EXPECT_CALL(*modelMock, addVersion(_, _, _))
        .Times(1)
        .WillRepeatedly(Return(ovms::Status(ovms::StatusCode::OK)));
    auto status = manager.startFromFile(fileToReload);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
    modelMock.reset();
}

TEST(ModelManager, StartFromFileWhenModelFilesMissing) {
    std::filesystem::create_directories(model_1_path);
    std::string fileToReload = "/tmp/ovms_config_file1.json";
    createConfigFileWithContent(config_1_model, fileToReload);
    ConstructorEnabledModelManager manager;
    ASSERT_TRUE(std::filesystem::is_empty(model_1_path));
    auto status = manager.startFromFile(fileToReload);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
}

TEST(ModelManager, ConfigReloadingShouldAddNewModel) {
    std::filesystem::create_directories(model_1_path);
    std::filesystem::create_directories(model_2_path);
    std::string fileToReload = "/tmp/ovms_config_file2.json";
    createConfigFileWithContent(config_1_model, fileToReload);
    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;
    EXPECT_CALL(*modelMock, addVersion(_, _, _))
        .WillRepeatedly(Return(ovms::Status(ovms::StatusCode::OK)));

    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    auto models = manager.getModels().size();
    EXPECT_EQ(models, 1);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    std::thread t([&manager]() {
        waitForOVMSConfigReload(manager);
    });
    t.join();
    createConfigFileWithContent(config_2_models, fileToReload);
    bool isNeeded = false;
    manager.configFileReloadNeeded(isNeeded);
    ASSERT_EQ(isNeeded, true);
    std::thread s([&manager]() {
        waitForOVMSConfigReload(manager);
    });
    s.join();
    models = manager.getModels().size();
    EXPECT_EQ(models, 2);
    manager.join();
    modelMock.reset();
}

TEST(ModelManager, ConfigReloadingWithWrongInputName) {
    ConstructorEnabledModelManager manager;
    ovms::ModelConfig config;
    config.parseShapeParameter("{\"wrong_input_name\": \"(1,3,224,224)\"}");
    config.setBasePath("/ovms/src/test/dummy");
    auto status = manager.reloadModelWithVersions(config);
    ASSERT_EQ(status, ovms::StatusCode::CONFIG_SHAPE_IS_NOT_IN_NETWORK);
}

TEST(ModelManager, ConfigReloadingStatefulDynamic) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    config.setStateful(true);
    config.setBatchingMode(ovms::Mode::AUTO);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_STATEFUL_MODEL);

    config.setBatchingMode(ovms::Mode::FIXED);
    config.setShapes({{"A", {ovms::Mode::AUTO, {1, 3, 224, 224}}}});
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_STATEFUL_MODEL);
}

TEST(ModelManager, ConfigReloadingNonStateful) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    config.setStateful(false);
    config.setLowLatencyTransformation(true);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER);
}

TEST(ModelManager, ConfigReloadingStatelessToStateful) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    config.setStateful(true);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::REQUESTED_MODEL_TYPE_CHANGE);
}

TEST(ModelManager, ConfigReloadingStatefulToStateless) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    config.setStateful(true);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);

    config.setStateful(false);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::REQUESTED_MODEL_TYPE_CHANGE);
}

class DummyModelDirectoryStructure {
private:
    std::string modelSourcePath;

public:
    DummyModelDirectoryStructure(std::string modelName) :
        modelSourcePath("/ovms/src/test/dummy/1/") {
        name = modelName;
        std::string modelPath = "/tmp/" + name;
        std::filesystem::remove_all(modelPath);
    }
    ~DummyModelDirectoryStructure() {
        std::string modelPath = "/tmp/" + name;
        std::filesystem::remove_all(modelPath);
    }

    std::string name;

    void addVersion(int number, bool valid) {
        std::string versionPath = "/tmp/" + name + "/" + std::to_string(number);
        std::filesystem::create_directories(versionPath);
        std::filesystem::copy(modelSourcePath, versionPath, std::filesystem::copy_options::recursive);
        if (!valid) {
            std::filesystem::copy(versionPath + "/dummy.bin", versionPath + "/dummy.xml", std::filesystem::copy_options::overwrite_existing);
        }
    }
    void removeVersion(int number) {
        std::string versionPath = "/tmp/" + name + "/" + std::to_string(number);
        std::filesystem::remove_all(versionPath);
    }
};

TEST(ModelManager, HandlingInvalidLastVersion) {
    DummyModelDirectoryStructure modelDirectory("HandlingInvalidLastVersion");
    bool validVersion = true;
    // valid version 1 and 2, invalid 3
    // expected version 2 as available
    modelDirectory.addVersion(1, validVersion);
    modelDirectory.addVersion(2, validVersion);
    modelDirectory.addVersion(3, !validVersion);
    ovms::ModelConfig config;
    config.setBasePath("/tmp/" + modelDirectory.name);
    config.setName(modelDirectory.name);
    config.setNireq(1);
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);
    std::shared_ptr<ovms::ModelInstance> modelInstance1;
    std::shared_ptr<ovms::ModelInstance> modelInstance2;
    std::shared_ptr<ovms::ModelInstance> modelInstance3;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = manager.getModelInstance(modelDirectory.name, 2, modelInstance2, modelInstanceUnloadGuard);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    ASSERT_EQ(modelInstance2->getStatus().getState(), ovms::ModelVersionState::AVAILABLE);
    modelInstanceUnloadGuard.reset();
    status = manager.getModelInstance(modelDirectory.name, 3, modelInstance3, modelInstanceUnloadGuard);
    modelInstanceUnloadGuard.reset();
    ASSERT_EQ(status, ovms::StatusCode::MODEL_VERSION_NOT_LOADED_YET);

    // dropped versions 2 and 3
    // expected version 1 as available, 2 as ended
    modelDirectory.removeVersion(3);
    modelDirectory.removeVersion(2);
    manager.reloadModelWithVersions(config);
    ASSERT_EQ(modelInstance2->getStatus().getState(), ovms::ModelVersionState::END);
    status = manager.getModelInstance(modelDirectory.name, 1, modelInstance1, modelInstanceUnloadGuard);
    modelInstanceUnloadGuard.reset();
    ASSERT_EQ(status, ovms::StatusCode::OK);
    ASSERT_EQ(modelInstance1->getStatus().getState(), ovms::ModelVersionState::AVAILABLE);

    // added invalid version 2
    // expected no changes: 1 as available, 2 as loading with error
    modelDirectory.addVersion(2, !validVersion);
    manager.reloadModelWithVersions(config);
    ASSERT_EQ(modelInstance1->getStatus().getState(), ovms::ModelVersionState::AVAILABLE);
    ASSERT_EQ(modelInstance2->getStatus().getState(), ovms::ModelVersionState::LOADING);
    ASSERT_EQ(modelInstance2->getStatus().getErrorCode(), ovms::ModelVersionStatusErrorCode::UNKNOWN);

    // fixed version 2
    // expected 2 as available and 1 as ended
    modelDirectory.removeVersion(2);
    modelDirectory.addVersion(2, validVersion);
    manager.reloadModelWithVersions(config);
    ASSERT_EQ(modelInstance1->getStatus().getState(), ovms::ModelVersionState::END);
    ASSERT_EQ(modelInstance2->getStatus().getState(), ovms::ModelVersionState::AVAILABLE);
    ASSERT_EQ(modelInstance2->getStatus().getErrorCode(), ovms::ModelVersionStatusErrorCode::OK);
}

TEST(ModelManager, InitialFailedLoadingVersionSavesModelVersionWithProperStatus) {
    DummyModelDirectoryStructure modelDirectory("InitialFailedLoadingVersionSavesModelVersionWithProperStatus");
    bool validVersion = true;
    modelDirectory.addVersion(1, !validVersion);
    ovms::ModelConfig config;
    config.setBasePath("/tmp/" + modelDirectory.name);
    config.setName(modelDirectory.name);
    config.setNireq(1);
    ConstructorEnabledModelManager manager;
    manager.reloadModelWithVersions(config);
    auto model = manager.findModelByName(modelDirectory.name);
    ASSERT_NE(model, nullptr);
    auto versions = model->getModelVersionsMapCopy();
    EXPECT_EQ(versions.size(), 1);
    auto versionIt = versions.find(1);
    ASSERT_NE(versionIt, versions.end());
    ASSERT_EQ(versionIt->second.getStatus().getState(), ovms::ModelVersionState::LOADING);
}

TEST(ModelManager, ModelVersionFailedReloadReportsFailedStatus) {
    DummyModelDirectoryStructure modelDirectory("ModelVersionFailedReloadReportsFailedStatus");
    bool validVersion = true;
    modelDirectory.addVersion(1, validVersion);
    ovms::ModelConfig config;
    config.setBasePath("/tmp/" + modelDirectory.name);
    config.setName(modelDirectory.name);
    config.setNireq(1);
    ConstructorEnabledModelManager manager;
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    // force reload with wrong shape
    config.setTargetDevice("invalid");
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE);
}

TEST(ModelManager, ConfigReloadingWithTwoModelsWithTheSameName) {
    const char* configWithTwoSameNames = R"({
   "model_config_list": [
    {
      "config": {
        "name": "same_name",
        "base_path": "/tmp/models/dummy1"
      }
    },
    {
      "config": {
        "name": "same_name",
        "base_path": "/tmp/models/dummy2"
      }
    }]})";
    std::filesystem::create_directories(model_1_path);
    std::filesystem::create_directories(model_2_path);
    std::string fileToReload = "/tmp/ovms_config_file2.json";
    createConfigFileWithContent(configWithTwoSameNames, fileToReload);
    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    EXPECT_CALL(*modelMock, addVersion(_, _, _))
        .Times(1)
        .WillRepeatedly(Return(ovms::Status(ovms::StatusCode::OK)));
    auto status = manager.startFromFile(fileToReload);
    auto models = manager.getModels().size();
    EXPECT_EQ(models, 1);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
    modelMock.reset();
}

class MockModelManagerWithModelInstancesJustChangingStates : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return std::make_shared<MockModelWithInstancesJustChangingStates>(name);
    }
    ovms::Status readAvailableVersions(
        std::shared_ptr<ovms::FileSystem>& fs,
        const std::string& base,
        ovms::model_versions_t& versions) override {
        versions.resize(toRegister.size());
        std::copy(toRegister.begin(), toRegister.end(), versions.begin());
        return ovms::StatusCode::OK;
    };
    void registerVersionToLoad(ovms::model_version_t version) {
        toRegister.emplace_back(version);
    }

private:
    std::vector<ovms::model_version_t> toRegister;
};

TEST(ModelManager, ConfigReloadingShouldRetireModelInstancesOfModelRemovedFromJson) {
    std::filesystem::create_directories(model_1_path);
    std::filesystem::create_directories(model_2_path);
    std::string fileToReload = "/tmp/ovms_config_file2.json";
    createConfigFileWithContent(config_2_models, fileToReload);
    modelMock = std::make_shared<MockModel>();
    MockModelManagerWithModelInstancesJustChangingStates manager;
    manager.registerVersionToLoad(1);
    manager.registerVersionToLoad(2);
    auto status = manager.startFromFile(fileToReload);
    manager.startWatcher();
    auto models = manager.getModels();
    ASSERT_EQ(models.size(), 2);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    waitForOVMSConfigReload(manager);
    models = manager.getModels();
    ASSERT_EQ(models.size(), 2);
    for (auto& nameModel : models) {
        for (auto& versionModelInstance : nameModel.second->getModelVersions()) {
            ASSERT_EQ(ovms::ModelVersionState::AVAILABLE, versionModelInstance.second->getStatus().getState());
        }
    }
    // we remove SECOND_MODEL from config file and expect to have all versions of it retired
    createConfigFileWithContent(config_1_model, fileToReload);
    waitForOVMSConfigReload(manager);
    models = manager.getModels();
    ASSERT_EQ(models.size(), 2);
    for (auto& versionModelInstance : manager.getModels().at(FIRST_MODEL_NAME)->getModelVersions()) {
        EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, versionModelInstance.second->getStatus().getState());
    }
    for (auto& versionModelInstance : manager.getModels().at(SECOND_MODEL_NAME)->getModelVersions()) {
        EXPECT_EQ(ovms::ModelVersionState::END, versionModelInstance.second->getStatus().getState());
    }
    manager.join();
}

class MockModelInstanceInStateWithConfig : public ovms::ModelInstance {
    static const ovms::model_version_t UNUSED_VERSION = 987789;

public:
    MockModelInstanceInStateWithConfig(ovms::ModelVersionState state, const ovms::ModelConfig& modelConfig, InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore, ieCore_2) {
        status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_VERSION, state);
        config = modelConfig;
    }
};

std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> getMockedModelVersionInstances(
    std::map<ovms::ModelVersionState, ovms::model_versions_t> initialVersionStates,
    InferenceEngine::Core& ieCore,
    ov::runtime::Core& ieCore_2,
    const ovms::ModelConfig& modelConfig = ovms::ModelConfig{}) {
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> modelVersions;
    std::array<ovms::ModelVersionState, 5> states{
        ovms::ModelVersionState::START,
        ovms::ModelVersionState::LOADING,
        ovms::ModelVersionState::AVAILABLE,
        ovms::ModelVersionState::UNLOADING,
        ovms::ModelVersionState::END};
    for (auto& state : states) {
        for (auto& version : initialVersionStates[state]) {
            modelVersions[version] = std::make_shared<MockModelInstanceInStateWithConfig>(state, modelConfig, ieCore, ieCore_2);
        }
    }
    return modelVersions;
}

class ModelManagerVersionsReload : public ::testing::Test {
protected:
    std::unique_ptr<InferenceEngine::Core> ieCore;
    std::unique_ptr<ov::runtime::Core> ieCore_2;

public:
    ModelManagerVersionsReload() {
        versionsToRetire = std::make_shared<ovms::model_versions_t>();
        versionsToReload = std::make_shared<ovms::model_versions_t>();
        versionsToStart = std::make_shared<ovms::model_versions_t>();
    }
    void SetUp() {
        ieCore = std::make_unique<InferenceEngine::Core>();
        ieCore_2 = std::make_unique<ov::runtime::Core>();
        initialVersions.clear();
        versionsToRetire->clear();
        versionsToReload->clear();
        versionsToStart->clear();
    }
    std::map<ovms::ModelVersionState, ovms::model_versions_t> initialVersions;
    std::shared_ptr<ovms::model_versions_t> versionsToRetire;
    std::shared_ptr<ovms::model_versions_t> versionsToReload;
    std::shared_ptr<ovms::model_versions_t> versionsToStart;
    ovms::ModelConfig oldConfig;
};

TEST_F(ModelManagerVersionsReload, RetireOldAddNew) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {1};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{2};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{2};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{1};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, NoVersionsChange) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {2, 3};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {1};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{2, 3};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, KeepOldAddNewNoRetired) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {1, 2};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{1, 2, 3};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{3};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, KeepOldAddNewWithRetiredVersions) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {2, 3};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {1};
    ovms::model_versions_t requestedVersions{2, 3, 4};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{4};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, JustAddNewVersions) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{1, 2};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{1, 2};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, JustRetireVersion) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {1, 2, 3};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{2, 3};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{1};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, ResurrectRetiredVersion) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {2};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {1};
    ovms::model_versions_t requestedVersions{1, 2};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{1};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, RessurectRetireAddAtTheSameTime) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {2};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {1};
    ovms::model_versions_t requestedVersions{1, 3};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{3};
    ovms::model_versions_t expectedVersionsToReload{1};
    ovms::model_versions_t expectedVersionsToRetire{2};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, DontStartAlreadyStartingVersion) {
    initialVersions[ovms::ModelVersionState::START] = {1};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{1};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, DontStartAlreadyLoadingVersion) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {1};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{1};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, DontRetireAlreadyUnloadingVersion) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {1};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}
TEST_F(ModelManagerVersionsReload, RetireLoadingVersion) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {1};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{1};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, RetireStartingVersion) {
    initialVersions[ovms::ModelVersionState::START] = {1};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{};
    ovms::model_versions_t expectedVersionsToRetire{1};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

TEST_F(ModelManagerVersionsReload, ReloadUnloadingVersion) {
    initialVersions[ovms::ModelVersionState::START] = {};
    initialVersions[ovms::ModelVersionState::LOADING] = {};
    initialVersions[ovms::ModelVersionState::AVAILABLE] = {};
    initialVersions[ovms::ModelVersionState::UNLOADING] = {1};
    initialVersions[ovms::ModelVersionState::END] = {};
    ovms::model_versions_t requestedVersions{1};
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances =
        getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2);
    ovms::ModelManager::getVersionsToChange(oldConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    ovms::model_versions_t expectedVersionsToStart{};
    ovms::model_versions_t expectedVersionsToReload{1};
    ovms::model_versions_t expectedVersionsToRetire{};
    EXPECT_THAT(expectedVersionsToStart, ContainerEq(*versionsToStart));
    EXPECT_THAT(expectedVersionsToReload, ContainerEq(*versionsToReload));
    EXPECT_THAT(expectedVersionsToRetire, ContainerEq(*versionsToRetire));
}

class ReloadAvailableModelDueToConfigChange : public ::testing::Test {
protected:
    std::unique_ptr<InferenceEngine::Core> ieCore;
    std::unique_ptr<ov::runtime::Core> ieCore_2;

public:
    ReloadAvailableModelDueToConfigChange() {
        versionsToRetire = std::make_shared<ovms::model_versions_t>();
        versionsToReload = std::make_shared<ovms::model_versions_t>();
        versionsToStart = std::make_shared<ovms::model_versions_t>();
    }
    void SetUp() {
        ieCore = std::make_unique<InferenceEngine::Core>();
        initialVersions.clear();
        versionsToRetire->clear();
        versionsToReload->clear();
        versionsToStart->clear();
        initialVersions[ovms::ModelVersionState::START] = {1};
        initialVersions[ovms::ModelVersionState::LOADING] = {2};
        initialVersions[ovms::ModelVersionState::AVAILABLE] = {3};
        initialVersions[ovms::ModelVersionState::UNLOADING] = {4};
        initialVersions[ovms::ModelVersionState::END] = {5};
    }

    std::map<ovms::ModelVersionState, ovms::model_versions_t> initialVersions;
    std::shared_ptr<ovms::model_versions_t> versionsToRetire;
    std::shared_ptr<ovms::model_versions_t> versionsToReload;
    std::shared_ptr<ovms::model_versions_t> versionsToStart;
    std::map<ovms::model_version_t, std::shared_ptr<ovms::ModelInstance>> mockModelVersionInstances;
    ovms::model_versions_t requestedVersions{3};
    ovms::ModelConfig config;
};

TEST_F(ReloadAvailableModelDueToConfigChange, SameConfig_ExpectNoReloads) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre());
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToBasePathChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setBasePath("new/custom/path");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToTargetDeviceChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setTargetDevice("GPU");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToBatchingModeChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setBatchingParams("auto");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToBatchSizeChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setBatchingParams("20");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToNireqChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setNireq(50);
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToPluginConfigChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setPluginConfig({{"A", "B"}});
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToLayoutChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setLayout("NEW_LAYOUT");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToNamedLayoutChange) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setLayouts({{"A", "B"}});
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToShapeConfigurationChange_Auto) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.parseShapeParameter("auto");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectNoReloadWhenShapeConfigurationStill_Auto) {
    config.parseShapeParameter("auto");
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.parseShapeParameter("auto");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre());
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectNoReloadWhenShapeConfigurationStill_Fixed) {
    config.parseShapeParameter("(1,3,224,224)");
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.parseShapeParameter("(1,3,224,224)");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre());
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToShapeConfigurationChange_AnonymousToNamed) {
    config.parseShapeParameter("auto");
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.parseShapeParameter("{\"a\": \"auto\"");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToShapeConfigurationChange_NoNamed) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.parseShapeParameter("(1,3,224,224)");
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToMultipleInputShapeChange) {
    ovms::ModelConfig previouslyLoadedConfig = config;
    previouslyLoadedConfig.setShapes({{"A", {ovms::Mode::FIXED, {1, 3, 224, 224}}},
        {"B", {ovms::Mode::FIXED, {1, 100}}}});
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, previouslyLoadedConfig);
    ovms::ModelConfig newConfig = config;
    newConfig.setShapes({{"A", {ovms::Mode::FIXED, {3, 3, 224, 224}}},
        {"B", {ovms::Mode::FIXED, {1, 100}}}});
    ovms::ModelManager::getVersionsToChange(newConfig, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

TEST_F(ReloadAvailableModelDueToConfigChange, ExpectReloadDueToShapeConfigurationChange_Named) {
    mockModelVersionInstances = getMockedModelVersionInstances(initialVersions, *ieCore, *ieCore_2, config);
    config.setShapes({{"A", {ovms::Mode::FIXED, {1, 3, 224, 224}}}});
    ovms::ModelManager::getVersionsToChange(config, mockModelVersionInstances, requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    EXPECT_THAT(*versionsToReload, UnorderedElementsAre(3));
}

// TODO: Enable tests once the issue https://jira.devtools.intel.com/browse/CVS-71746 is resolved.
class DISABLED_GetModelInstanceTest : public ::testing::Test {};

class MockModel : public ovms::Model {};

std::shared_ptr<ovms::Model> model;

class MockModelManagerWith1Model : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return model;
    }
};

TEST_F(DISABLED_GetModelInstanceTest, WithRequestedNameShouldReturnModelNameMissing) {
    MockModelManagerWith1Model manager;
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = manager.getModelInstance("SOME", 0, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_NAME_MISSING) << "Should fail with no model with such name registered";
}

TEST_F(DISABLED_GetModelInstanceTest, WithRequestedUnexistingVersionShouldReturnModelVersionMissing) {
    MockModelManagerWith1Model manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    model = std::make_unique<ovms::Model>(config.getName(), false, nullptr);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = manager.getModelInstance(config.getName(), 2, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_MISSING) << "Should fail with no model with such name registered";
}

class MockModelInstanceFakeLoad : public ovms::ModelInstance {
public:
    MockModelInstanceFakeLoad(InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore, ieCore_2) {}

protected:
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        status = ovms::ModelVersionStatus(name, version);
        status.setAvailable();
        return ovms::StatusCode::OK;
    }
};

class ModelWithModelInstanceFakeLoad : public ovms::Model {
public:
    ModelWithModelInstanceFakeLoad(const std::string& name) :
        Model(name, false, nullptr) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t, InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) override {
        return std::make_shared<MockModelInstanceFakeLoad>(ieCore, ieCore_2);
    }
};

std::shared_ptr<ModelWithModelInstanceFakeLoad> modelWithModelInstanceFakeLoad;

TEST_F(DISABLED_GetModelInstanceTest, WithRequestedDefaultVersionUnloadedShouldReturnModelVersionMissing) {
    MockModelManagerWith1Model manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    model = std::make_unique<ovms::Model>(config.getName(), false, nullptr);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::model_versions_t> versionsToRetire = std::make_shared<ovms::model_versions_t>();
    versionsToRetire->emplace_back(1);
    model->retireVersions(versionsToRetire);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = manager.getModelInstance(config.getName(), 0, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_MISSING);
}

TEST_F(DISABLED_GetModelInstanceTest, WithRequestedVersion1ShouldReturnModelVersionNotLoadedAnymore) {
    MockModelManagerWith1Model manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    model = std::make_unique<ovms::Model>(config.getName(), false, nullptr);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::model_versions_t> versionsToRetire = std::make_shared<ovms::model_versions_t>();
    versionsToRetire->emplace_back(1);
    model->retireVersions(versionsToRetire);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = manager.getModelInstance(config.getName(), 1, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
}

class ModelInstanceLoadedStuckInLoadingState : public ovms::ModelInstance {
public:
    ModelInstanceLoadedStuckInLoadingState(InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore, ieCore_2) {}

protected:
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        status = ovms::ModelVersionStatus(name, version);
        status.setLoading();
        return ovms::StatusCode::OK;
    }
};

class ModelWithModelInstanceLoadedStuckInLoadingState : public ovms::Model {
public:
    ModelWithModelInstanceLoadedStuckInLoadingState(const std::string& name) :
        Model(name, false, nullptr) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t, InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) override {
        return std::make_shared<ModelInstanceLoadedStuckInLoadingState>(ieCore, ieCore_2);
    }
};

std::shared_ptr<ModelWithModelInstanceLoadedStuckInLoadingState> modelWithModelInstanceLoadedStuckInLoadingState;

class ModelManagerWithModelInstanceLoadedStuckInLoadingState : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return modelWithModelInstanceLoadedStuckInLoadingState;
    }
};

const int AVAILABLE_STATE_DELAY_MILLISECONDS = 5;

class ModelInstanceLoadedWaitInLoadingState : public ovms::ModelInstance {
public:
    ModelInstanceLoadedWaitInLoadingState(InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2, const uint modelInstanceLoadDelayInMilliseconds) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore, ieCore_2),
        modelInstanceLoadDelayInMilliseconds(modelInstanceLoadDelayInMilliseconds) {}

protected:
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        this->status = ovms::ModelVersionStatus(name, version);
        this->status.setLoading();
        std::thread([this]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(modelInstanceLoadDelayInMilliseconds));
            status.setAvailable();
            modelLoadedNotify.notify_all();
        })
            .detach();
        return ovms::StatusCode::OK;
    }

private:
    const uint modelInstanceLoadDelayInMilliseconds;
};

class ModelWithModelInstanceLoadedWaitInLoadingState : public ovms::Model {
public:
    ModelWithModelInstanceLoadedWaitInLoadingState(const std::string& name, const uint modelInstanceLoadDelayInMilliseconds) :
        Model(name, false, nullptr),
        modelInstanceLoadDelayInMilliseconds(modelInstanceLoadDelayInMilliseconds) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string&, const ovms::model_version_t, InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2) override {
        return std::make_shared<ModelInstanceLoadedWaitInLoadingState>(ieCore, ieCore_2, modelInstanceLoadDelayInMilliseconds);
    }

private:
    const uint modelInstanceLoadDelayInMilliseconds;
};

std::shared_ptr<ModelWithModelInstanceLoadedWaitInLoadingState> modelWithModelInstanceLoadedWaitInLoadingState;

class ModelManagerWithModelInstanceLoadedWaitInLoadingState : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return modelWithModelInstanceLoadedWaitInLoadingState;
    }

    void setWaitForModelLoadedTimeoutMs(uint32_t newTimeout) {
        waitForModelLoadedTimeoutMs = newTimeout;
    }
};

class ModelInstanceModelLoadedNotify : public ::testing::Test {};

TEST_F(ModelInstanceModelLoadedNotify, WhenChangedStateFromLoadingToAvailableInNotReachingTimeoutShouldSuceed) {
    // Need unit tests for modelInstance load first
    ModelManagerWithModelInstanceLoadedWaitInLoadingState manager;
    const uint32_t modelLoadingTimeoutMs = 100;
    manager.setWaitForModelLoadedTimeoutMs(modelLoadingTimeoutMs);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    modelWithModelInstanceLoadedWaitInLoadingState = std::make_shared<ModelWithModelInstanceLoadedWaitInLoadingState>(
        config.getName(), modelLoadingTimeoutMs / 1.5);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = manager.getModelInstance(config.getName(), 1, modelInstance, modelInstanceUnloadGuardPtr);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance->getStatus().getState());
}

TEST_F(ModelInstanceModelLoadedNotify, WhenChangedStateFromLoadingToAvailableInReachingTimeoutShouldReturnModelNotLoadedYet) {
    // Need unit tests for modelInstance load first
    ModelManagerWithModelInstanceLoadedWaitInLoadingState manager;
    const uint32_t modelLoadingTimeoutMs = 100;
    manager.setWaitForModelLoadedTimeoutMs(modelLoadingTimeoutMs);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;

    const auto MODEL_LOADING_LONGER_THAN_WAIT_FOR_LOADED_TIMEOUT_MS = 1.5 * modelLoadingTimeoutMs;
    modelWithModelInstanceLoadedWaitInLoadingState = std::make_shared<ModelWithModelInstanceLoadedWaitInLoadingState>(
        config.getName(), MODEL_LOADING_LONGER_THAN_WAIT_FOR_LOADED_TIMEOUT_MS);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK_RELOADED);
    ASSERT_EQ(ovms::ModelVersionState::LOADING, modelWithModelInstanceLoadedWaitInLoadingState->getModelInstanceByVersion(1)->getStatus().getState());
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = manager.getModelInstance(config.getName(), 1, modelInstance, modelInstanceUnloadGuardPtr);
    spdlog::error("State: {}", (int)modelInstance->getStatus().getState());
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance->getStatus().getState()) << "State:" << (int)modelInstance->getStatus().getState();
    spdlog::error("State: {}", (int)modelInstance->getStatus().getState());
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_NOT_LOADED_YET);
}
