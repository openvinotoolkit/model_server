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

#include "src/filesystem.hpp"
#include "src/status.hpp"
#include "src/stringutils.hpp"
#include "src/capi_frontend/server_settings.hpp"
#include "src/config_export_module/config_export.hpp"

#include "light_test_utils.hpp"
#include "platform_utils.hpp"
#include "test_with_temp_dir.hpp"

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

const std::string expectedConfigContentsWindows = R"({
    "model_config_list": [
        { 
            "config": {
                "name": "model1",
                "base_path": "model1\\Path"
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
        this->modelsSettings.configPath = ovms::FileSystem::appendSlash(this->directoryPath) + "config.json";
    }
    void TearDown() {
        TestWithTempDir::TearDown();
    }
};

TEST_F(ConfigCreationTest, positiveAddWithDirectConfigFilePathNotExisting) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();

    std::string configFile = this->modelsSettings.configPath;
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}
TEST_F(ConfigCreationTest, positiveAddWithDirectConfigFilePathExisting) {
    std::string configContents = expectedEmptyConfigContents;
    createConfigFileWithContent(configContents, this->modelsSettings.configPath);
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();

    std::string configFile = this->modelsSettings.configPath;
    std::string configContentsRead = GetFileContents(configFile);
    std::string expectedContent = expectedConfigContents;
    ovms::erase_spaces(expectedContent);
    ovms::erase_spaces(configContentsRead);
    ASSERT_EQ(expectedContent, configContentsRead) << configContents;
}
TEST_F(ConfigCreationTest, positiveRemoveModelWithDirectConfigFilePathExisting) {
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

    createConfigFileWithContent(configContents, this->modelsSettings.configPath);

    auto status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    std::string expectedConfigContents = R"({
    "model_config_list": [],
    "pipeline_config_list": [],
    "custom_loader_config_list": []
}
)";

    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}
TEST_F(ConfigCreationTest, negativeRemoveModelWithDirectConfigFilePathNotExisting) {
    this->modelsSettings.configPath = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "SOME_NONEXISTING_FILE.json";
    auto status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID) << status.string();
}

TEST_F(ConfigCreationTest, positiveAddModel) {
#ifdef _WIN32
    this->modelsSettings.modelPath = "model1\\Path";
#endif
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configContents = GetFileContents(this->modelsSettings.configPath);

#ifdef _WIN32
    const std::string* expectedConfig = &expectedConfigContentsWindows;
#elif __linux__
    const std::string* expectedConfig = &expectedConfigContents;
#endif

    ASSERT_EQ(*expectedConfig, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToEmptyConfig) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    configContents = GetFileContents(this->modelsSettings.configPath);
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

    createConfigFileWithContent(configContents, this->modelsSettings.configPath);

    auto status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK) << configContents;

    std::string expectedConfigContents = R"({
    "model_config_list": [],
    "pipeline_config_list": [],
    "custom_loader_config_list": []
}
)";

    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveAddTwoModelsToNonEmptyConfig) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    this->modelsSettings.modelName = "model2";
    this->modelsSettings.modelPath = "/model2/Path";
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContentsTwoModels, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToNonEmptyConfig) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configContents = GetFileContents(this->modelsSettings.configPath);
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

    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expected2ModelsConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToNonEmptyConfigFirstModel) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configContents = GetFileContents(this->modelsSettings.configPath);
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

    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expected2ModelsConfigContentsFirst, configContents) << configContents;
}

TEST_F(ConfigCreationTest, positiveRemoveOneModelToNonEmptyConfigLast) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configContents = GetFileContents(this->modelsSettings.configPath);
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

    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expected2ModelsConfigContentsLast, configContents) << configContents;
}

TEST_F(ConfigCreationTest, negativeWrongPathsEnable) {
    this->modelsSettings.configPath = "";
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST_F(ConfigCreationTest, negativeWrongPathsDisable) {
    this->modelsSettings.configPath = "";
    auto status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::PATH_INVALID);

    this->modelsSettings.configPath = ovms::FileSystem::appendSlash(this->directoryPath) + "some.file";
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

    std::string configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add second model
    status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NAME_OCCUPIED);
}

TEST_F(ConfigCreationTest, negativeRemoveNotExistingName) {
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    // Add third model
    this->modelsSettings.modelName = "model3";
    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::MODEL_NAME_MISSING);

    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;
}

TEST_F(ConfigCreationTest, negativeInvalidJson) {
    // Create config file with an empty config & reload
    const std::string configStr = R"({
    "model_confdffig_list":[]
    })";
    createConfigFileWithContent(configStr, this->modelsSettings.configPath);
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);

    status = ovms::updateConfig(this->modelsSettings, ovms::DISABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
}
