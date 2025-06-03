//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "test_utils.hpp"
#include "../filesystem.hpp"
#include "../module_names.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "../capi_frontend/server_settings.hpp"
#include "../config_export_module/config_export.hpp"

const std::string expectedConfigContents = R"({
    "model_config_list": [
        { 
            "config": {
                "name": "model1",
                "base_path": "/model1/Path"
            }
        }
    ]
}
)";

const std::string expectedConfigContentsTwoModels = R"({
    "model_config_list": [
        {
            "config": {
                "name": "model1",
                "base_path": "/model1/Path"
            }
        },
        {
            "config": {
                "name": "model2",
                "base_path": "/model2/Path"
            }
        }
    ]
}
)";

const std::string expectedEmptyConfigContents = R"({
    "model_config_list": []
}
)";

const std::string expected2ModelsConfigContents = R"({
    "model_config_list": [
        {
            "config": {
                "name": "model1",
                "base_path": "/model1/Path"
            }
        },
        {
            "config": {
                "name": "model3",
                "base_path": "/model3/Path"
            }
        }
    ]
}
)";

const std::string expected2ModelsConfigContentsFirst = R"({
    "model_config_list": [
        {
            "config": {
                "name": "model2",
                "base_path": "/model2/Path"
            }
        },
        {
            "config": {
                "name": "model3",
                "base_path": "/model3/Path"
            }
        }
    ]
}
)";

const std::string expected2ModelsConfigContentsLast = R"({
    "model_config_list": [
        {
            "config": {
                "name": "model1",
                "base_path": "/model1/Path"
            }
        },
        {
            "config": {
                "name": "model2",
                "base_path": "/model2/Path"
            }
        }
    ]
}
)";

class ConfigCreationTest : public TestWithTempDir {
protected:
    ovms::ModelsSettingsImpl modelsSettings;
    void SetUp() {
        TestWithTempDir::SetUp();
        this->modelsSettings.modelName = "model1";
        this->modelsSettings.modelPath = "/model1/Path";
        this->modelsSettings.configPath = this->directoryPath;
    }
    void TearDown() {
        TestWithTempDir::TearDown();
    }
};

TEST_F(ConfigCreationTest, positiveAddModel) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToEmptyConfig) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedEmptyConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToExistingConfig) {
    std::string configContents = R"(
{
  "model_config_list": [
    {
      "config": {
        "name": "model1",
        "base_path": "/models/resnet-50-tf",
        "batch_size": 1,
        "target_device": "CPU"
      }
    }
  ],
  "pipeline_config_list": [],
  "custom_loader_config_list": []
})";

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    createConfigFileWithContent(configContents, configFile);

    auto status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    std::string expectedConfigContents = R"({
    "model_config_list": [],
    "pipeline_config_list": [],
    "custom_loader_config_list": []
}
)";

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveAddTwoModelsToNonEmptyConfig) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    this->modelsSettings.modelName = "model2";
    this->modelsSettings.modelPath = "/model2/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContentsTwoModels, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToNonEmptyConfig) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    this->modelsSettings.modelName = "model2";
    this->modelsSettings.modelPath = "/model2/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    // Add third model
    this->modelsSettings.modelName = "model3";
    this->modelsSettings.modelPath = "/model3/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    this->modelsSettings.modelName = "model2";
    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expected2ModelsConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToNonEmptyConfigFirstModel) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    this->modelsSettings.modelName = "model2";
    this->modelsSettings.modelPath = "/model2/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    // Add third model
    this->modelsSettings.modelName = "model3";
    this->modelsSettings.modelPath = "/model3/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    this->modelsSettings.modelName = "model1";
    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expected2ModelsConfigContentsFirst, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToNonEmptyConfigLast) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    this->modelsSettings.modelName = "model2";
    this->modelsSettings.modelPath = "/model2/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    // Add third model
    this->modelsSettings.modelName = "model3";
    this->modelsSettings.modelPath = "/model3/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    this->modelsSettings.modelName = "model3";
    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expected2ModelsConfigContentsLast, configContents) << configContents;
}

TEST_F(ConfigCreationTest, negativeWrongPathsEnable) {
    this->modelsSettings.configPath = "";
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);

    this->modelsSettings.configPath = ovms::FileSystem::appendSlash(this->directoryPath) + "some.file";
    createConfigFileWithContent(expectedConfigContents, this->modelsSettings.configPath);
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST_F(ConfigCreationTest, negativeWrongPathsDisable) {
    this->modelsSettings.configPath = "";
    auto status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);

    this->modelsSettings.configPath = ovms::FileSystem::appendSlash(this->directoryPath) + "some.file";
    createConfigFileWithContent(expectedConfigContents, this->modelsSettings.configPath);
    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST_F(ConfigCreationTest, negativeInternalErrorForType) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::DELETE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::NOT_IMPLEMENTED);
    status = ovms::updateConfig(this->modelsSettings, ovms::UNKNOWN_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::NOT_IMPLEMENTED);
}

TEST_F(ConfigCreationTest, negativeAddTheSameModelName) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NAME_OCCUPIED);
}

TEST_F(ConfigCreationTest, negativeRemoveNotExistingName) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add third model
    this->modelsSettings.modelName = "model3";
    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NAME_MISSING);

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, negativeInvalidJson) {
    // Create config file with an empty config & reload
    const std::string configStr = R"({
    "model_confdffig_list":[]
    })";
    createConfigFileWithContent(configStr, ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json");
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);

    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST_F(ConfigCreationTest, positiveWithStart) {
    this->modelsSettings.modelPath = getGenericFullPathForSrcTest("/ovms/src/test/dummy");
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    ConstructorEnabledModelManager manager;
    status = manager.startFromFile(ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
}

TEST_F(ConfigCreationTest, positiveEndToEndEnableDisable) {
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;
    server.setShutdownRequest(0);
    char* argv[] = {
        (char*)"ovms",
        (char*)"--add_to_config",
        (char*)this->modelsSettings.configPath.c_str(),
        (char*)"--model_name",
        (char*)this->modelsSettings.modelName.c_str(),
        (char*)"--model_path",
        (char*)this->modelsSettings.modelPath.c_str(),
    };

    int argc = 7;
    t.reset(new std::thread([&argc, &argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(argc, argv));
    }));

    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME) != ovms::ModuleState::NOT_INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 3)) {
    }

    ASSERT_EQ(server.getModuleState(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME), ovms::ModuleState::NOT_INITIALIZED);
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::NOT_INITIALIZED);

    server.setShutdownRequest(1);
    t->join();
    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    server.setShutdownRequest(0);
    char* argv2[] = {
        (char*)"ovms",
        (char*)"--remove_from_config",
        (char*)this->modelsSettings.configPath.c_str(),
        (char*)"--model_name",
        (char*)this->modelsSettings.modelName.c_str(),
    };

    argc = 5;
    t.reset(new std::thread([&argc, &argv2, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(argc, argv2));
    }));

    start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME) != ovms::ModuleState::SHUTDOWN) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 3)) {
    }

    ASSERT_EQ(server.getModuleState(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME), ovms::ModuleState::SHUTDOWN);
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::NOT_INITIALIZED);
    server.setShutdownRequest(1);
    t->join();
    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedEmptyConfigContents, configContents) << configContents;
}
