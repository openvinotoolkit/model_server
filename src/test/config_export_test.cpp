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

const std::string expectedConfigContents = R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": "model1",
                    "base_path": "/model1/Path"
                }
            }
        ]
    }
)";

const std::string expectedEmptyConfigContents = R"(
    {
        "model_config_list": [
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

TEST_F(ConfigCreationTest, positiveRemoveModelToEmptyConfig) {
    auto status = ovms::createConfig(this->modelsSettings, ovms::enable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    std::string configFile = ovms::FileSystem::appendSlash(this->modelsSettings.configPath) + "config.json";
    std::string configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedConfigContents, configContents) << configContents;

    status = ovms::createConfig(this->modelsSettings, ovms::disable_model);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    configContents = GetFileContents(configFile);
    ASSERT_EQ(expectedEmptyConfigContents, configContents) << configContents;
}

