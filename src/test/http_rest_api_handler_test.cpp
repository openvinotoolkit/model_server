//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include <gtest/gtest.h>

#include "../modelmanager.hpp"
#include "../http_rest_api_handler.hpp"
#include "../config.hpp"
#include "../logging.hpp"

#include "test_utils.hpp"

#pragma GCC diagnostic ignored "-Wwrite-strings"

static const char* config_1 = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
            }
        }
    ]
})";

class ModelControlApi : public TestWithTempDir {};

TEST(ModelControlApi, nonExistingConfigFile) {
    std::filesystem::remove("/tmp/ovms_config_file.json");
    auto configFile = createConfigFileWithContent(config_1);
    char* n_argv[] = {"ovms", "--config_path", "/tmp/nonExistingConfig.json", "--file_system_poll_wait_seconds", "0"};
    int arg_count = 5;
    ovms::Config::instance().parse(arg_count, n_argv);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;
    auto status = handler.processModelControlApiRequest(response);

    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID);
}

TEST(ModelControlApi, positive) {
    std::filesystem::remove("/tmp/ovms_config_file.json");
    auto configFile = createConfigFileWithContent(config_1);
    char* n_argv[] = {"ovms", "--config_path", "/tmp/ovms_config_file.json", "--file_system_poll_wait_seconds", "0"};
    int arg_count = 5;
    ovms::Config::instance().parse(arg_count, n_argv);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;
    auto status = handler.processModelControlApiRequest(response);

    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST(ModelControlApi, reloadNotNeeded) {
    std::filesystem::remove("/tmp/ovms_config_file.json");
    auto configFile = createConfigFileWithContent(config_1);
    char* n_argv[] = {"ovms", "--config_path", "/tmp/ovms_config_file.json", "--file_system_poll_wait_seconds", "0"};
    int arg_count = 5;
    ovms::Config::instance().parse(arg_count, n_argv);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    manager.loadConfig(configFile);
    auto status = handler.processModelControlApiRequest(response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOAD_NOT_NEEDED);
}
