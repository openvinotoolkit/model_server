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

#include "../config.hpp"
#include "../http_rest_api_handler.hpp"
#include "../logging.hpp"
#include "../modelmanager.hpp"
#include "test_utils.hpp"

#pragma GCC diagnostic ignored "-Wwrite-strings"

static const char* configWith1Dummy = R"(
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

class ModelManagerTest : public ovms::ModelManager {};

class ConfigApi : public TestWithTempDir {
    std::string configFilePath;

public:
    void SetUpConfig(const std::string& configContent) {
        configFilePath = directoryPath + "/ovms_config.json";
        createConfigFileWithContent(configContent, configFilePath);
        char* n_argv[] = {"ovms", "--config_path", &configFilePath[0], "--file_system_poll_wait_seconds", "0"};
        int arg_count = 5;
        ovms::Config::instance().parse(arg_count, n_argv);
    }

    void LoadConfig(ModelManagerTest& manager) {
        manager.loadConfig(configFilePath);
    }

    void RemoveConfig() {
        std::filesystem::remove(configFilePath);
    }
};

class ConfigReload : public ConfigApi {};

TEST_F(ConfigReload, nonExistingConfigFile) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);
    LoadConfig(manager);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    RemoveConfig();
    auto status = handler.processConfigReloadRequest(response, manager);
    const char* expectedJson = "{\n\t\"error\": \"Config file not found or cannot open.\"\n}";
    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID);
}

TEST_F(ConfigReload, removeConfigFileThenRestore) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);
    LoadConfig(manager);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    RemoveConfig();
    auto status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID);

    SetUpConfig(configWith1Dummy);
    status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);
}

TEST_F(ConfigReload, startWith1DummyThenReload) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWith1Dummy);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    auto status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);
}

static const char* configWith1DummyInTmp = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/tmp/dummy"
            }
        }
    ]
})";

TEST_F(ConfigReload, startWith1DummyThenAddVersion) {
    ModelManagerTest manager;
    SetUpConfig(configWith1DummyInTmp);
    std::filesystem::remove_all("/tmp/dummy");
    std::filesystem::copy("/ovms/src/test/dummy", "/tmp/dummy", std::filesystem::copy_options::recursive);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson1 = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    auto status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson1, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_NOT_RELOADED);

    std::filesystem::copy("/ovms/src/test/dummy/1", "/tmp/dummy/2", std::filesystem::copy_options::recursive);

    const char* expectedJson2 = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "END",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  },
  {
   "version": "2",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson2, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);
    std::filesystem::remove_all("/tmp/dummy");
}

static const char* emptyConfig = R"(
{
    "model_config_list": []
})";

TEST_F(ConfigReload, StartWith1DummyThenReloadToRetireDummy) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);
    LoadConfig(manager);
    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson_1 = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";

    auto status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(expectedJson_1, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_NOT_RELOADED);

    RemoveConfig();
    SetUpConfig(emptyConfig);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson_2 = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "END",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";

    status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(expectedJson_2, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);
}

TEST_F(ConfigReload, reloadNotNeeded) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    LoadConfig(manager);
    auto status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(status, ovms::StatusCode::OK_NOT_RELOADED);
}

TEST_F(ConfigReload, reloadNotNeededManyThreads) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);

    int numberOfThreads = 10;
    std::vector<std::thread> threads;
    std::function<void()> func = [&handler, &manager]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::string response;
        EXPECT_EQ(handler.processConfigReloadRequest(response, manager), ovms::StatusCode::OK_NOT_RELOADED);
    };

    for (int i = 0; i < numberOfThreads; i++) {
        threads.push_back(std::thread(func));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

static const char* configWith1DummyPipeline = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                            "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                        "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                        "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(ConfigReload, StartWith1DummyThenReloadToAddPipeline) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWith1DummyPipeline);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
},
"pipeline1Dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    auto status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);
}

static const char* configWith2DummyPipelines = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                            "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                        "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                        "data_item": "new_dummy_output"}
                }
            ]
        },
        {
            "name": "pipeline2Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                            "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                        "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                        "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TEST_F(ConfigReload, StartWith1DummyPipelineThenReloadToAddPipeline) {
    ModelManagerTest manager;
    SetUpConfig(configWith1DummyPipeline);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWith1DummyPipeline);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson_1 = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
},
"pipeline1Dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";

    auto status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(expectedJson_1, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);

    RemoveConfig();
    SetUpConfig(configWith2DummyPipelines);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson_2 = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
},
"pipeline1Dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
},
"pipeline2Dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";

    status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(expectedJson_2, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);
}

TEST_F(ConfigReload, addPipelineWithNonexistingNode) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWith1Dummy);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson_1 = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";

    auto status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(expectedJson_1, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);

static const char* configWith1PipelineWithNonExistingNode = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "nonExisting",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                            "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                        "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                        "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

    RemoveConfig();
    SetUpConfig(configWith1PipelineWithNonExistingNode);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson_2 = "{\n\t\"error\": \"Reloading config file failed. Check server logs for more info.\"\n}";

    status = handler.processConfigReloadRequest(response, manager);
    EXPECT_EQ(expectedJson_2, response);
    EXPECT_EQ(status, ovms::StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL);
}

class ConfigStatus : public ConfigApi {};

TEST_F(ConfigStatus, configWithPipelines) {
    ModelManagerTest manager;
    SetUpConfig(configWith2DummyPipelines);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;
    LoadConfig(manager);

    const char* expectedJson = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
},
"pipeline1Dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
},
"pipeline2Dummy" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    auto status = handler.processConfigStatusRequest(response, manager);
    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}
