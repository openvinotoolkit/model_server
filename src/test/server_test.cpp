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
#include <random>
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
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"

using ovms::ModelManager;
using ovms::Module;
using ovms::ModuleState;
using ovms::Server;

using testing::_;
using testing::ContainerEq;
using testing::Return;
using testing::ReturnRef;
using testing::UnorderedElementsAre;

using grpc::Channel;
using grpc::ClientContext;

struct Configuration {
    std::string address = "localhost";
    std::string port = "9178";
};

class ServingClient {
    std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;

public:
    ServingClient(std::shared_ptr<Channel> channel) :
        stub_(inference::GRPCInferenceService::NewStub(channel)) {
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
    void verifyReady(grpc::StatusCode expectedStatus = grpc::StatusCode::OK, bool ready = true) {
        ClientContext context;
        ::inference::ServerReadyRequest request;
        ::inference::ServerReadyResponse response;

        auto status = stub_->ServerReady(&context, request, &response);
        ASSERT_EQ(status.error_code(), expectedStatus);
        EXPECT_EQ(response.ready(), ready);
    }
    void verifyModelReady(const std::string& modelName, grpc::StatusCode expectedStatus = grpc::StatusCode::OK, bool ready = true) {
        ClientContext context;
        ::inference::ModelReadyRequest request;
        ::inference::ModelReadyResponse response;
        request.set_name(modelName);

        auto status = stub_->ModelReady(&context, request, &response);
        ASSERT_EQ(status.error_code(), expectedStatus);
        EXPECT_EQ(response.ready(), ready);
    }
};

void requestServerAlive(const char* grpcPort, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyLive(status, expectedStatus);
}
void requestServerReady(const char* grpcPort, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    SPDLOG_DEBUG("Verying if server is ready on address: {}", address);
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyReady(status, expectedStatus);
}

void requestModelReady(const char* grpcPort, const std::string& modelName, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    SPDLOG_DEBUG("Verying if server is ready on address: {}", address);
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyModelReady(modelName, status, expectedStatus);
}

TEST(Server, ServerNotAliveBeforeStart) {
    // here we should fail to connect before starting server
    requestServerAlive("9178", grpc::StatusCode::UNAVAILABLE, false);
}

class MockedServableManagerModule : public ovms::ServableManagerModule {
public:
    bool waitWithStart = true;
    bool waitWithChangingState = true;
    MockedServableManagerModule() = default;
    int start(const ovms::Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        while (waitWithStart)  // TODO add 3s failsafe
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto status = this->servableManager->start(config);
        if (status.ok()) {
            while (waitWithChangingState)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

            state = ModuleState::INITIALIZED;
            return EXIT_SUCCESS;
        }
        SPDLOG_ERROR("ovms::ModelManager::Start() Error: {}", status.string());
        return EXIT_FAILURE;
    }
};

class MockedServer : public Server {
protected:
    MockedServer() = default;

public:
    static MockedServer& instance() {
        static MockedServer global;
        return global;
    }
    std::unique_ptr<Module> createModule(const std::string& name) const override {
        if (name != ovms::SERVABLE_MANAGER_MODULE_NAME)
            return Server::createModule(name);
        return std::make_unique<MockedServableManagerModule>();
    };
    Module* getModule(const std::string& name) {
        return const_cast<Module*>(Server::getModule(name));
    }
};

using ovms::SERVABLE_MANAGER_MODULE_NAME;

TEST(Server, ServerAliveBeforeLoadingModels) {
    // purpose of this test is to ensure that the server responds with alive=true before loading any models.
    // this is to make sure that eg. k8s won't restart container until all models are loaded because of not being alivea
    std::string port = "9000";
    const char* testPort = port.c_str();
    std::mt19937_64 eng{std::random_device{}()};
    std::uniform_int_distribution<> dist{0, 9};
    std::this_thread::sleep_for(std::chrono::milliseconds{dist(eng)});
    for (auto j : {1, 2, 3}) {
        char* digitToRandomize = (char*)testPort + j;
        *digitToRandomize += dist(eng);
    }

    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)"/ovms/src/test/dummy",
        (char*)"--log_level",
        (char*)"DEBUG",
        (char*)"--port",
        (char*)port.c_str(),
        nullptr};

    // start testing
    requestServerAlive(argv[8], grpc::StatusCode::UNAVAILABLE, false);
    MockedServer& server = MockedServer::instance();
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(9, argv));
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState("GRPCServerModule") != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 1)) {
    }

    SPDLOG_INFO("here ensure that server is already live but not ready yet");
    requestServerAlive(argv[8], grpc::StatusCode::OK, true);
    requestServerReady(argv[8], grpc::StatusCode::OK, false);
    requestModelReady(argv[8], argv[2], grpc::StatusCode::NOT_FOUND, false);

    SPDLOG_INFO("here check that model & server still is not ready since servable manager module only started loading \n\
    we have to wait for module to start loading");
    while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) == ovms::ModuleState::NOT_INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 1)) {
    }
    requestModelReady(argv[8], argv[2], grpc::StatusCode::NOT_FOUND, false);
    auto mockedServableManagerModule = dynamic_cast<MockedServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME));
    ASSERT_NE(nullptr, mockedServableManagerModule);

    SPDLOG_INFO("here we start loading model \n\
    however modelmanager adds instance of the model only after it was properly loaded \n\
     this could be potentially changed");
    mockedServableManagerModule->waitWithStart = false;
    requestServerReady(argv[8], grpc::StatusCode::OK, false);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // average:32ms on CLX3 to load model
    requestModelReady(argv[8], argv[2], grpc::StatusCode::NOT_FOUND, false);

    SPDLOG_INFO("/ here check that server eventually is still not ready beceause module is not initialized \n\
    // sleep potentially to improve with signaling");
    std::this_thread::sleep_for(std::chrono::milliseconds(70));  // average:32ms on CLX3
    requestModelReady(argv[8], argv[2], grpc::StatusCode::OK, true);
    requestServerReady(argv[8], grpc::StatusCode::OK, false);

    SPDLOG_INFO("here check that server is finally ready");
    mockedServableManagerModule->waitWithChangingState = false;
    requestServerReady(argv[8], grpc::StatusCode::OK, false);
    server.setShutdownRequest(1);
    t.join();
    SPDLOG_INFO("here check end statuses");
    requestModelReady(argv[8], argv[2], grpc::StatusCode::UNAVAILABLE, false);
    requestServerReady(argv[8], grpc::StatusCode::UNAVAILABLE, false);
    requestServerAlive(argv[8], grpc::StatusCode::UNAVAILABLE, false);
}
