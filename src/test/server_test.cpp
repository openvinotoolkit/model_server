//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <chrono>
#include <thread>

#include <gmock/gmock.h>
#include <grpcpp/create_channel.h>
#include <gtest/gtest.h>

#include "../cleaner_utils.hpp"
#include "../kfs_grpc_inference_service.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../model.hpp"
#include "../modelmanager.hpp"
#include "../node_library.hpp"
#include "../prediction_service_utils.hpp"
#include "../server.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"

using testing::_;
using testing::ContainerEq;
using testing::Return;
using testing::ReturnRef;
using testing::UnorderedElementsAre;

using grpc::Channel;
using grpc::ClientContext;

template <typename T>
class ResourceGuard {
    T* ptr;

public:
    ResourceGuard(T* ptr) :
        ptr(ptr) {}
    ~ResourceGuard() { delete ptr; }
};

struct Configuration {
    std::string address = "localhost";
    std::string port = "9178";
};

class ServingClient {
    std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;

public:
    ServingClient(std::shared_ptr<Channel> channel, const Configuration& config) :
        stub_(inference::GRPCInferenceService::NewStub(channel)) {
        this->config = config;
    }

    // Pre-processing function for synthetic data.
    // gRPC request proto is generated with synthetic data with shape/precision matching endpoint metadata.
    void verifyLive(grpc::StatusCode expectedStatus = grpc::StatusCode::OK, bool alive = true) {
        ClientContext context;
        ::inference::ServerLiveRequest request;
        ::inference::ServerLiveResponse response;

        auto status = stub_->ServerLive(&context, request, &response);
        ASSERT_EQ(status.error_code(), expectedStatus);
        EXPECT_EQ(response.live(), alive);
    }

private:
    Configuration config;
    ::inference::ServerLiveRequest request;
};

void requestServerAlive(grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    Configuration config;
    config.address = "localhost";
    config.port = "9178";
    std::string address = config.address + ":" + config.port;
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args), config);
    client.verifyLive(status, expectedStatus);
}

TEST(Server, ServerNotAliveBeforeStart) {
    // here we should fail to connect before starting server
    requestServerAlive(grpc::StatusCode::UNAVAILABLE, false);
}

TEST(Server, ServerAliveBeforeLoadingModels) {
    // purpose of this test is to ensure that the server responds with alive=true before loading any models.
    // this is to make sure that eg. k8s won't restart container until all models are loaded because of not being alive
    requestServerAlive(grpc::StatusCode::UNAVAILABLE, false);
    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)"/ovms/src/test/dummy",
        nullptr};
    SPDLOG_ERROR("ER");
    ovms::Server& server = ovms::Server::instance();
    std::thread t([&argv, &server]() {
        server.start(5, argv);
    });
    auto start = std::chrono::high_resolution_clock::now();
    while (ovms::Server::instance().getModuleState("GRPCModule") != ovms::ModuleState::INITIALIZED &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 1)) {
    }
    requestServerAlive(grpc::StatusCode::OK, true);
    ovms::Server::instance().setShutdownRequest(1);
    t.join();
    requestServerAlive(grpc::StatusCode::UNAVAILABLE, false);
}
class MockModel2 : public ovms::Model {
public:
    MockModel2() :
        Model("MOCK_NAME", false, nullptr) {}
    MOCK_METHOD(ovms::Status, addVersion, (const ovms::ModelConfig&, ov::Core&), (override));
};

std::shared_ptr<MockModel2> modelMock;

class MockModelManagerR : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return modelMock;
    }
};

TEST(ModelManager, ConfigParseNoModelsR) {
    std::string configFile = createConfigFileWithContent("{ \"model_config_list\": [ ] }\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST(ModelManager, ConfigParseEmptyJsonR) {
    std::string configFile = createConfigFileWithContent("{}\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
}

TEST(ModelManager, ConfigParseNodeConfigWithoutNameKeyR) {
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

TEST(ModelManager, ConfigParseNodeConfigWihoutBasePathKeyR) {
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

TEST(ModelManager, parseConfigWhenPipelineDefinitionMatchSchemaR) {
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
    modelMock = std::make_shared<MockModel2>();
    MockModelManagerR manager;

    auto status = manager.startFromFile(configFile);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
    modelMock.reset();
}

void setupModelsDirsR() {
    std::filesystem::create_directory("/tmp/models");
    std::filesystem::create_directory("/tmp/models/dummy1");
    std::filesystem::create_directory("/tmp/models/dummy2");
}

const char* config_2_modelsR = R"({
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
TEST(ModelManager, configRelodNotNeededManyThreadsR) {
    std::string configFile = "/tmp/config.json";

    modelMock = std::make_shared<MockModel2>();
    MockModelManagerR manager;
    setupModelsDirsR();
    createConfigFileWithContent(config_2_modelsR, configFile);
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
