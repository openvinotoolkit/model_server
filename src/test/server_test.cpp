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

private:
    ::inference::ServerLiveRequest request;
};

void requestServerAlive(const char* grpcPort, grpc::StatusCode status = grpc::StatusCode::OK, bool expectedStatus = true) {
    grpc::ChannelArguments args;
    std::string address = std::string("localhost") + ":" + grpcPort;
    ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    client.verifyLive(status, expectedStatus);
}

TEST(Server, ServerNotAliveBeforeStart) {
    // here we should fail to connect before starting server
    requestServerAlive("9178", grpc::StatusCode::UNAVAILABLE, false);
}

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
        (char*)"--port",
        (char*)port.c_str(),
        nullptr};
    requestServerAlive(argv[6], grpc::StatusCode::UNAVAILABLE, false);
    ovms::Server& server = ovms::Server::instance();
    std::thread t([&argv, &server]() {
        server.start(7, argv);
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((ovms::Server::instance().getModuleState("GRPCServerModule") != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 1)) {
    }
    requestServerAlive(argv[6], grpc::StatusCode::OK, true);
    ovms::Server::instance().setShutdownRequest(1);
    t.join();
    requestServerAlive(argv[6], grpc::StatusCode::UNAVAILABLE, false);
}
