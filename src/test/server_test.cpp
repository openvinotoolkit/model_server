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
#include <httplib.h>

#include "../cleaner_utils.hpp"
#include "../dags/node_library.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../model.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../modelmanager.hpp"
#include "../module_names.hpp"
#include "../prediction_service_utils.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../version.hpp"
#include "c_api_test_utils.hpp"
#include "mockmodelinstancechangingstates.hpp"

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

const std::string portOldDefault{"9178"};
const std::string typicalRestDefault{"9179"};

struct Configuration {
    std::string address = "localhost";
    std::string port = portOldDefault;
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

        ASSERT_NE(nullptr, stub_);
        auto status = stub_->ServerLive(&context, request, &response);
        // if we failed to connect it is ok return here
        ASSERT_EQ(status.error_code(), expectedStatus);
        EXPECT_EQ(response.live(), alive);
    }
    void verifyReady(grpc::StatusCode expectedStatus = grpc::StatusCode::OK, bool ready = true) {
        ClientContext context;
        ::inference::ServerReadyRequest request;
        ::inference::ServerReadyResponse response;

        ASSERT_NE(nullptr, stub_);
        auto status = stub_->ServerReady(&context, request, &response);
        ASSERT_EQ(status.error_code(), expectedStatus);
        EXPECT_EQ(response.ready(), ready);
    }
    void verifyModelReady(const std::string& modelName, grpc::StatusCode expectedStatus = grpc::StatusCode::OK, bool ready = true) {
        ClientContext context;
        ::KFSGetModelStatusRequest request;
        ::KFSGetModelStatusResponse response;
        request.set_name(modelName);

        ASSERT_NE(nullptr, stub_);
        auto status = stub_->ModelReady(&context, request, &response);
        ASSERT_EQ(status.error_code(), expectedStatus);
        EXPECT_EQ(response.ready(), ready);
    }

    void verifyServerMetadata(grpc::StatusCode expectedStatus = grpc::StatusCode::OK) {
        ClientContext context;
        ::KFSServerMetadataRequest request;
        ::KFSServerMetadataResponse response;
        ASSERT_NE(nullptr, stub_);
        auto status = stub_->ServerMetadata(&context, request, &response);
        ASSERT_EQ(status.error_code(), expectedStatus);
        EXPECT_EQ(response.name(), PROJECT_NAME);
        EXPECT_EQ(response.version(), PROJECT_VERSION);
        EXPECT_EQ(response.extensions().size(), 0);
    }
};

static void requestServerAlive(const char* grpcPort, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    SPDLOG_INFO("Verifying if server is live on address: {}", address);
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyLive(status, expectedStatus);
}

static void requestServerReady(const char* grpcPort, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    SPDLOG_INFO("Verifying if server is ready on address: {}", address);
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyReady(status, expectedStatus);
}

static void requestModelReady(const char* grpcPort, const std::string& modelName, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    SPDLOG_INFO("Verifying if model is ready on address: {}", address);
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyModelReady(modelName, status, expectedStatus);
}

static void checkServerMetadata(const char* grpcPort, grpc::StatusCode status = grpc::StatusCode::OK) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    SPDLOG_INFO("Verifying if server responds with correct metadata on address: {}", address);
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyServerMetadata(status);
}

static void requestRestServerAlive(const char* httpPort, httplib::StatusCode status = httplib::StatusCode::OK_200, bool expectedStatus = true) {
    std::unique_ptr<httplib::Client> cli{nullptr};
    try {
        cli = std::make_unique<httplib::Client>(std::string("http://localhost:") + httpPort);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Exception caught during rest request:{}", e.what());
        EXPECT_TRUE(false);
        return;
    } catch (...) {
        SPDLOG_ERROR("Exception caught during rest request");
        EXPECT_TRUE(false);
        return;
    }
    try {
        auto res = cli->Get("/v2/health/live");
        if (!res) {
            SPDLOG_ERROR("Got error:{}", httplib::to_string(res.error()));
            EXPECT_TRUE(!expectedStatus);
            return;
        } else {
            EXPECT_TRUE(expectedStatus);
        }
        if (res->status != httplib::StatusCode::OK_200) {
            SPDLOG_ERROR("Failed to get liveness status code: {}, status: {}", res->status, httplib::status_message(res->status));
            EXPECT_TRUE(false);
            return;
        }
    } catch (std::exception& e) {
        SPDLOG_ERROR("Exception caught during rest request:{}", e.what());
        EXPECT_TRUE(false);
        return;
    } catch (...) {
        SPDLOG_ERROR("Exception caught during rest request");
        EXPECT_TRUE(false);
        return;
    }
    return;
}

TEST(Server, ServerNotAliveBeforeStart) {
    // here we should fail to connect before starting server
    requestServerAlive(portOldDefault.c_str(), grpc::StatusCode::UNAVAILABLE, false);
}

using ovms::Config;
using ovms::SERVABLE_MANAGER_MODULE_NAME;

namespace {
class MockedServableManagerModule : public ovms::ServableManagerModule {
public:
    bool waitWithStart = true;
    bool waitWithChangingState = true;
    MockedServableManagerModule(ovms::Server& ovmsServer) :
        ovms::ServableManagerModule(ovmsServer) {}
    ovms::Status start(const Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        SPDLOG_INFO("Mocked {} starting", SERVABLE_MANAGER_MODULE_NAME);
        auto start = std::chrono::high_resolution_clock::now();
        while (waitWithStart &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto status = this->servableManager->start(config);
        if (status.ok()) {
            start = std::chrono::high_resolution_clock::now();
            while (waitWithChangingState &&
                   (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5))
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

            state = ModuleState::INITIALIZED;
            SPDLOG_INFO("Mocked {} started", SERVABLE_MANAGER_MODULE_NAME);
            return status;
        }
        SPDLOG_ERROR("ovms::ModelManager::Start() Error: {}", status.string());
        return status;
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
    std::unique_ptr<Module> createModule(const std::string& name) override {
        if (name != ovms::SERVABLE_MANAGER_MODULE_NAME)
            return Server::createModule(name);
        return std::make_unique<MockedServableManagerModule>(*this);
    };
    Module* getModule(const std::string& name) {
        return const_cast<Module*>(Server::getModule(name));
    }
};
}  // namespace
using ovms::SERVABLE_MANAGER_MODULE_NAME;

TEST(Server, ServerAliveBeforeLoadingModels) {
    // purpose of this test is to ensure that the server responds with alive=true before loading any models.
    // this is to make sure that eg. k8s won't restart container until all models are loaded because of not being alivea
    std::string port = "9000";
    randomizePort(port);

    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest("/ovms/src/test/dummy").c_str(),
        (char*)"--log_level",
        (char*)"DEBUG",
        (char*)"--port",
        (char*)port.c_str(),
        nullptr};

    SPDLOG_INFO("server should not respond with live when not started");
    requestServerAlive(argv[8], grpc::StatusCode::UNAVAILABLE, false);
    MockedServer& server = MockedServer::instance();
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(9, argv));
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::GRPC_SERVER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }

    SPDLOG_INFO("here ensure that server is already live but not ready yet");
    requestServerAlive(argv[8], grpc::StatusCode::OK, true);
    requestServerReady(argv[8], grpc::StatusCode::OK, false);
    requestModelReady(argv[8], argv[2], grpc::StatusCode::NOT_FOUND, false);

    SPDLOG_INFO(R"(here check that model & server still is not ready since servable manager module only started loading
    we have to wait for module to start loading)");
    while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) == ovms::ModuleState::NOT_INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }
    requestModelReady(argv[8], argv[2], grpc::StatusCode::NOT_FOUND, false);
    auto mockedServableManagerModule = dynamic_cast<MockedServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME));
    ASSERT_NE(nullptr, mockedServableManagerModule);

    SPDLOG_INFO(R"(here we start loading model
    however modelmanager adds instance of the model only after it was properly loaded
     this could be potentially changed)");
    mockedServableManagerModule->waitWithStart = false;
    requestServerReady(argv[8], grpc::StatusCode::OK, false);

    SPDLOG_INFO(R"(here check that server eventually is still not ready beceause module is not initialized
    sleep potentially to improve with signaling)");
    auto& manager = mockedServableManagerModule->getServableManager();
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 8) {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));  // average:32ms on CLX3 to load model
        std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
        auto status = manager.getModelInstance("dummy", 1, modelInstance, modelInstanceUnloadGuard);
        if (!status.ok())
            continue;
        if (modelInstance->getStatus().getState() == ovms::ModelVersionState::AVAILABLE)
            break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(15));  // average:32ms on CLX3 to load model

    requestModelReady(argv[8], argv[2], grpc::StatusCode::OK, true);
    requestServerReady(argv[8], grpc::StatusCode::OK, false);

    SPDLOG_INFO("here check that server is finally ready");
    start = std::chrono::high_resolution_clock::now();
    mockedServableManagerModule->waitWithChangingState = false;
    while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }
    requestServerReady(argv[8], grpc::StatusCode::OK, true);
    server.setShutdownRequest(1);
    t.join();
    server.setShutdownRequest(0);
    SPDLOG_INFO("here check end statuses");
    requestModelReady(argv[8], argv[2], grpc::StatusCode::UNAVAILABLE, false);
    requestServerReady(argv[8], grpc::StatusCode::UNAVAILABLE, false);
    requestServerAlive(argv[8], grpc::StatusCode::UNAVAILABLE, false);
}

TEST(Server, ServerMetadata) {
    std::string port = "9000";
    randomizePort(port);
    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest("/ovms/src/test/dummy").c_str(),
        (char*)"--port",
        (char*)port.c_str(),
        nullptr};

    ovms::Server& server = ovms::Server::instance();
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(7, argv));
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((ovms::Server::instance().getModuleState(ovms::GRPC_SERVER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }

    grpc::ChannelArguments args;
    std::string address = std::string("localhost:") + port;
    requestServerAlive(port.c_str(), grpc::StatusCode::OK, true);
    checkServerMetadata(port.c_str(), grpc::StatusCode::OK);
    server.setShutdownRequest(1);
    t.join();
    server.setShutdownRequest(0);
}

TEST(Server, GrpcWorkers2) {
    std::string port = "9000";
    randomizePort(port);
    std::string workers = "2";
    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest("/ovms/src/test/dummy").c_str(),
        (char*)"--port",
        (char*)port.c_str(),
        (char*)"--grpc_workers",
        (char*)workers.c_str(),
        (char*)"--log_level",
        (char*)"DEBUG",
        nullptr};

    ovms::Server& server = ovms::Server::instance();

#ifdef __linux__
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(11, argv));
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((ovms::Server::instance().getModuleState(ovms::GRPC_SERVER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }

    ASSERT_EQ(ovms::Server::instance().getModuleState(ovms::GRPC_SERVER_MODULE_NAME), ovms::ModuleState::INITIALIZED) << "Server not started error.";

    grpc::ChannelArguments args;
    std::string address = std::string("localhost:") + port;
    requestServerAlive(port.c_str(), grpc::StatusCode::OK, true);
    checkServerMetadata(port.c_str(), grpc::StatusCode::OK);
    server.setShutdownRequest(1);
    t.join();
    server.setShutdownRequest(0);
#elif _WIN32
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_FAILURE, server.start(11, argv));
    });
    t.join();
#endif
}

TEST(Server, ProperShutdownInCaseOfStartError) {
    std::string port = "9000";
    std::string restPort = "9000";
    randomizePort(port);
    randomizePort(restPort);
    while (port == restPort)
        randomizePort(restPort);
    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)"NON_EXISTING_PATH",
        (char*)"--port",
        (char*)port.c_str(),
        (char*)"--rest_port",
        (char*)restPort.c_str(),
        (char*)"--log_level",
        (char*)"DEBUG",
        nullptr};
    ovms::Server& server = ovms::Server::instance();
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_FAILURE, server.start(11, argv));
    });
    t.join();
    // this test should not hang
}

TEST(Server, grpcArguments) {
    std::string port = "9000";
    std::string channel_arguments_str = "grpc.max_connection_age_ms=2000,grpc.max_concurrent_streams=10";
    std::string grpc_max_threads = "8";
    std::string grpc_memory_quota = "100000";
    randomizePort(port);
    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest("/ovms/src/test/dummy").c_str(),
        (char*)"--port",
        (char*)port.c_str(),
        (char*)"--grpc_channel_arguments",
        (char*)channel_arguments_str.c_str(),
        (char*)"--grpc_max_threads",
        (char*)grpc_max_threads.c_str(),
        (char*)"--grpc_memory_quota",
        (char*)grpc_memory_quota.c_str(),
        nullptr};

    ovms::Server& server = ovms::Server::instance();
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(13, argv));
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((ovms::Server::instance().getModuleState(ovms::GRPC_SERVER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }

    std::string address = std::string("localhost:") + port;
    requestServerAlive(port.c_str(), grpc::StatusCode::OK, true);
    checkServerMetadata(port.c_str(), grpc::StatusCode::OK);
    server.setShutdownRequest(1);
    t.join();
    server.setShutdownRequest(0);
}
TEST(Server, CAPIAliveGrpcNotHttpNot) {
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_dummy.json").c_str());
    OVMS_Server* cserver = serverGuard.server;
    bool isLive = false;
    OVMS_ServerLive(cserver, &isLive);
    ASSERT_TRUE(isLive);
    // GRPC is initialized before Servable ManagerModule
    requestServerAlive(portOldDefault.c_str(), grpc::StatusCode::UNAVAILABLE, false);
    requestRestServerAlive(typicalRestDefault.c_str(), httplib::StatusCode::NotFound_404, false);
}
TEST(Server, CAPIAliveGrpcNotHttpYes) {
    GTEST_SKIP() << "Until we have a way to launch all tests restarting drogon";  // TODO @dkalinow to enable drogon tests
    std::string port = "9000";
    randomizePort(port);
    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--rest_port",
        (char*)port.c_str(),
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest("/ovms/src/test/dummy").c_str(),
        nullptr};

    ovms::Server& server = ovms::Server::instance();
    bool isLive = true;
    auto* cserver = reinterpret_cast<OVMS_Server*>(&server);
    OVMS_ServerLive(cserver, &isLive);
    ASSERT_TRUE(!isLive);
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(7, argv));
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((ovms::Server::instance().getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }
    isLive = false;
    OVMS_ServerLive(cserver, &isLive);
    ASSERT_TRUE(isLive);
    // GrPC is initialized before Servable ManagerModule
    requestServerAlive(portOldDefault.c_str(), grpc::StatusCode::UNAVAILABLE, false);
    requestRestServerAlive(port.c_str());
    server.setShutdownRequest(1);
    t.join();
    server.setShutdownRequest(0);
}
