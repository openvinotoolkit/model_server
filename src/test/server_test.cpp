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

void requestServerAlive(const char* grpcPort, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    Configuration config;
    config.address = "localhost"; //adaw
    config.port = grpcPort;
    std::string address = config.address + ":" + config.port;
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args), config);
    client.verifyLive(status, expectedStatus);
}

TEST(Server, ServerNotAliveBeforeStart) {
    // here we should fail to connect before starting server
    requestServerAlive("9178", grpc::StatusCode::UNAVAILABLE, false);
}

static int i = 0;
TEST(Server, ServerAliveBeforeLoadingModels) {
    // purpose of this test is to ensure that the server responds with alive=true before loading any models.
    // this is to make sure that eg. k8s won't restart container until all models are loaded because of not being alivea
    const char* testPort = "9170";
    char* last = (char*) testPort + 3;
    last += (i % 10);
    SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX             :i :{}", ++i);

    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)"/ovms/src/test/dummy",
        (char*)"--port",
        (char*)testPort,
        nullptr};
    SPDLOG_ERROR("ER");
    requestServerAlive(argv[6], grpc::StatusCode::UNAVAILABLE, false);
    SPDLOG_ERROR("ER");
    ovms::Server& server = ovms::Server::instance();
    std::thread t([&argv, &server]() {
        server.start(7, argv);
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((ovms::Server::instance().getModuleState("GRPCServerModule") != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 1)) {
    }
    SPDLOG_ERROR("ER");
    requestServerAlive(argv[6], grpc::StatusCode::OK, true);
    SPDLOG_ERROR("ER");
    ovms::Server::instance().setShutdownRequest(1);
    t.join();
    SPDLOG_ERROR("ER");
    requestServerAlive(argv[6], grpc::StatusCode::UNAVAILABLE, false);
    SPDLOG_ERROR("ER");
}
