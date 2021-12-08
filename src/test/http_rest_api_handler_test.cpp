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

static const char* configWith1DummyNew = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
		"batch_size": "16"
            }
        }
    ]
})";

class ModelManagerTest : public ovms::ModelManager {};

class ConfigApi : public TestWithTempDir {
    std::string configFilePath;
    std::string modelPath;
    std::string modelName;

public:
    void SetUpConfig(const std::string& configContent) {
        configFilePath = directoryPath + "/ovms_config.json";
        createConfigFileWithContent(configContent, configFilePath);
        char* n_argv[] = {"ovms", "--config_path", configFilePath.data(), "--file_system_poll_wait_seconds", "0"};
        int arg_count = 5;
        ovms::Config::instance().parse(arg_count, n_argv);
    }

    void LoadConfig(ModelManagerTest& manager) {
        manager.loadConfig(configFilePath);
    }

    void RemoveConfig() {
        std::filesystem::remove(configFilePath);
    }

    void SetUpSingleModel(std::string modelPath, std::string modelName) {
        this->modelPath = modelPath;
        this->modelName = modelName;
        char* n_argv[] = {"ovms", "--model_path", this->modelPath.data(), "--model_name", this->modelName.data(), "--file_system_poll_wait_seconds", "0"};
        int arg_count = 7;
        ovms::Config::instance().parse(arg_count, n_argv);
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
    EXPECT_EQ(status, ovms::StatusCode::CONFIG_FILE_TIMESTAMP_READING_FAILED);
}

static const char* configWithModelNonExistingPath = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/non/exisiting"
            }
        }
    ]
})";

TEST_F(ConfigReload, nonExistingModelPathInConfig) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);
    LoadConfig(manager);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;
    RemoveConfig();
    SetUpConfig(configWithModelNonExistingPath);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto status = handler.processConfigReloadRequest(response, manager);
    const char* expectedJson = "{\n\t\"error\": \"Reloading config file failed. Check server logs for more info.\"\n}";
    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

static const char* configWithDuplicatedModelName = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
            }
        },
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/add_two_inputs_model"
            }
        }
    ]
})";

TEST_F(ConfigReload, duplicatedModelNameInConfig) {
    ModelManagerTest manager;
    SetUpConfig(configWithDuplicatedModelName);
    LoadConfig(manager);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto status = handler.processConfigReloadRequest(response, manager);
    const char* expectedJson = "{\n\t\"error\": \"Reloading config file failed. Check server logs for more info.\"\n}";
    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_NAME_OCCUPIED);
}

TEST_F(ConfigReload, startWith1DummyThenReload) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWith1DummyNew);
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

TEST_F(ConfigReload, singleModel) {
    ModelManagerTest manager;
    SetUpSingleModel("/ovms/src/test/dummy", "dummy");
    manager.startFromConfig();

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    auto status = handler.processConfigReloadRequest(response, manager);

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

    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_NOT_RELOADED);
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

TEST_F(ConfigReload, startWithMissingXmlThenAddAndReload) {
    ModelManagerTest manager;
    SetUpConfig(configWith1DummyInTmp);
    std::filesystem::remove_all("/tmp/dummy");
    std::filesystem::create_directory("/tmp/dummy");
    std::filesystem::create_directory("/tmp/dummy/1");
    std::filesystem::copy("/ovms/src/test/dummy/1/dummy.bin", "/tmp/dummy/1/dummy.bin", std::filesystem::copy_options::recursive);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson1 = "{\n\t\"error\": \"Reloading config file failed. Check server logs for more info.\"\n}";
    auto status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson1, response);
    EXPECT_EQ(status, ovms::StatusCode::FILE_INVALID);

    std::filesystem::copy("/ovms/src/test/dummy/1/dummy.xml", "/tmp/dummy/1/dummy.xml", std::filesystem::copy_options::recursive);

    const char* expectedJson2 = R"({
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
    status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson2, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_RELOADED);
    std::filesystem::remove_all("/tmp/dummy");
}

TEST_F(ConfigReload, startWithEmptyModelDir) {
    ModelManagerTest manager;
    SetUpConfig(configWith1DummyInTmp);
    std::filesystem::remove_all("/tmp/dummy");
    std::filesystem::create_directory("/tmp/dummy");

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson = R"({
"dummy" : 
{
 "model_version_status": []
}
})";
    auto status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::OK_NOT_RELOADED);

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

static const char* configWith1DummyPipelineNew = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "batch_size": "16"

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

static const char* configWithPipelineWithInvalidOutputs = R"(
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
                                        "data_item": "non_existing_output"}
                }
            ]
        }
    ]
})";

TEST_F(ConfigReload, StartWith1DummyThenReloadToAddPipelineWithInvalidOutputs) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWithPipelineWithInvalidOutputs);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson = "{\n\t\"error\": \"Reloading config file failed. Check server logs for more info.\"\n}";
    auto status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_DATA_SOURCE);
}

TEST_F(ConfigReload, reloadWithInvalidPipelineConfigManyThreads) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWithPipelineWithInvalidOutputs);
    int numberOfThreads = 2;
    std::vector<std::thread> threads;
    std::function<void()> func = [&handler, &manager]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::string response;
        EXPECT_EQ(handler.processConfigReloadRequest(response, manager), ovms::StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_DATA_SOURCE);
    };

    for (int i = 0; i < numberOfThreads; i++) {
        threads.push_back(std::thread(func));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

TEST_F(ConfigReload, reloadWithInvalidModelConfigManyThreads) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWithDuplicatedModelName);
    int numberOfThreads = 2;
    std::vector<std::thread> threads;
    std::function<void()> func = [&handler, &manager]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::string response;
        EXPECT_EQ(handler.processConfigReloadRequest(response, manager), ovms::StatusCode::MODEL_NAME_OCCUPIED);
    };

    for (int i = 0; i < numberOfThreads; i++) {
        threads.push_back(std::thread(func));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
static const char* configWithPipelineContainsNonExistingModel = R"(
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
                    "model_name": "non-existing",
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

TEST_F(ConfigReload, StartWith1DummyThenReloadToAddPipelineWithNonExistingModel) {
    ModelManagerTest manager;
    SetUpConfig(configWith1Dummy);

    auto handler = ovms::HttpRestApiHandler(10);
    std::string response;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LoadConfig(manager);
    RemoveConfig();
    SetUpConfig(configWithPipelineContainsNonExistingModel);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const char* expectedJson = "{\n\t\"error\": \"Reloading config file failed. Check server logs for more info.\"\n}";
    auto status = handler.processConfigReloadRequest(response, manager);

    EXPECT_EQ(expectedJson, response);
    EXPECT_EQ(status, ovms::StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL);
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
    SetUpConfig(configWith1DummyPipelineNew);
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
