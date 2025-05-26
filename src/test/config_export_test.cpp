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

const std::string expectedEmptyConfigContents = R"({}
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
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToEmptyConfig) {
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    status = ovms::createConfig(this->modelsSettings, ovms::disable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedEmptyConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveAddTwoModelsToNonEmptyConfig) {
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    this->modelsSettings.modelName = "model2";
    this->modelsSettings.modelPath = "/model2/Path";
    status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContentsTwoModels, configContents) << configContents;
}


TEST_F(ConfigCreationTest, positiveRemoveOneModelToNonEmptyConfig) {
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    this->modelsSettings.modelName = "model2";
    this->modelsSettings.modelPath = "/model2/Path";
    status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    // Add third model
    this->modelsSettings.modelName = "model3";
    this->modelsSettings.modelPath = "/model3/Path";
    status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    this->modelsSettings.modelName = "model2";
    status = ovms::createConfig(this->modelsSettings, ovms::disable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedEmptyConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, negativeWrongPath) {
    this->modelsSettings.configPath = "";
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST_F(ConfigCreationTest, negativeInternalErrorForType) {
    auto status = ovms::createConfig(this->modelsSettings, ovms::delete_model);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);
    status = ovms::createConfig(this->modelsSettings, ovms::unknown_model);
    ASSERT_EQ(status, ovms::StatusCode::INTERNAL_ERROR);
}

TEST_F(ConfigCreationTest, negativeAddTheSameModelName) {
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NAME_OCCUPIED);
}

TEST_F(ConfigCreationTest, negativeRemoveNotExistingName) {
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add third model
    this->modelsSettings.modelName = "model3";
    status = ovms::createConfig(this->modelsSettings, ovms::disable_model);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NAME_MISSING);
}

TEST_F(ConfigCreationTest, negativeInvalidJson) {
    // Create config file with an empty config & reload
    const std::string configStr = R"({
    "model_confdffig_list":[]
    })";
    createConfigFileWithContent(configStr, ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json");
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);

    status = ovms::createConfig(this->modelsSettings, ovms::disable_model);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST_F(ConfigCreationTest, positiveWithStart) {
    this->modelsSettings.modelPath = getGenericFullPathForSrcTest("/ovms/src/test/dummy");
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    ConstructorEnabledModelManager manager;
    status = manager.startFromFile(ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

