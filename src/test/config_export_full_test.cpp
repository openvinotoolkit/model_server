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

#include "constructor_enabled_model_manager.hpp"
#include "test_utils.hpp"
#include "light_test_utils.hpp"
#include "platform_utils.hpp"
#include "test_with_temp_dir.hpp"
#include "../filesystem.hpp"
#include "../module_names.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../capi_frontend/server_settings.hpp"
#include "../config_export_module/config_export.hpp"
namespace {
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
const std::string expectedEmptyConfigContents = R"({
    "model_config_list": []
}
)";
}  // namespace

class ConfigCreationFullTest : public TestWithTempDir {
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
TEST_F(ConfigCreationFullTest, positiveWithStart) {
    this->modelsSettings.modelPath = getGenericFullPathForSrcTest("/ovms/src/test/dummy");
    auto status = ovms::updateConfig(this->modelsSettings, ovms::ENABLE_MODEL);
    ASSERT_EQ(status, ovms::StatusCode::OK);

    ConstructorEnabledModelManager manager;
    status = manager.startFromFile(this->modelsSettings.configPath);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
}

TEST_F(ConfigCreationFullTest, positiveEndToEndEnableDisable) {
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
    std::string configContents = GetFileContents(this->modelsSettings.configPath);
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
    while ((server.getModuleState(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME) != ovms::ModuleState::NOT_INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 3)) {
    }

    ASSERT_EQ(server.getModuleState(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME), ovms::ModuleState::NOT_INITIALIZED);
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::NOT_INITIALIZED);
    server.setShutdownRequest(1);
    t->join();
    configContents = GetFileContents(this->modelsSettings.configPath);
    ASSERT_EQ(expectedEmptyConfigContents, configContents) << configContents;
}
