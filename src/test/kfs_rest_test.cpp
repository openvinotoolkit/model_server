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
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "../config.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"

using ovms::Config;
using ovms::HttpRestApiHandler;
using ovms::KFS_GetModelMetadata;
using ovms::KFS_GetModelReady;
using ovms::Module;
using ovms::ModuleState;
using ovms::SERVABLE_MANAGER_MODULE_NAME;
using ovms::Server;

namespace {
class MockedServableManagerModule : public ovms::ServableManagerModule {
public:
    MockedServableManagerModule() = default;
    int start(const Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        SPDLOG_INFO("Mocked {} starting", SERVABLE_MANAGER_MODULE_NAME);
        // we don't start ModelManager since we don't use it here
        state = ModuleState::INITIALIZED;
        return EXIT_SUCCESS;
    }
};
class MockedGRPCServerModule : public ovms::GRPCServerModule {
public:
    MockedGRPCServerModule(ovms::Server& server) :
        GRPCServerModule(server) {}
    int start(const Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        SPDLOG_INFO("Mocked {} starting", SERVABLE_MANAGER_MODULE_NAME);
        // we don't start ModelManager since we don't use it here
        state = ModuleState::INITIALIZED;
        return EXIT_SUCCESS;
    }
};
class MockedServer : public Server {
public:
    MockedServer() = default;

    std::unique_ptr<Module> createModule(const std::string& name) override {
        if (name == ovms::SERVABLE_MANAGER_MODULE_NAME)
            return std::make_unique<MockedServableManagerModule>();
        if (name == ovms::GRPC_SERVER_MODULE_NAME)
            return std::make_unique<MockedGRPCServerModule>(*this);
        return Server::createModule(name);
    };
    Module* getModule(const std::string& name) {
        return const_cast<Module*>(Server::getModule(name));
    }
};
}  // namespace

class HttpRestApiHandlerTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        HttpRestApiHandlerTest::server = std::make_unique<MockedServer>();
        std::string port = "9000";
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
        thread = std::make_unique<std::thread>(
            [&argv]() {
                ASSERT_EQ(EXIT_SUCCESS, server->start(9, argv));
            });
        auto start = std::chrono::high_resolution_clock::now();
        while ((server->getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }
    }
    void SetUp() override {
        handler = std::make_unique<HttpRestApiHandler>(*server, 5);
    }
    void TearDown() override {
        handler.reset();
    }
    static void TearDownTestSuite() {
        server->setShutdownRequest(1);
        thread->join();
        server->setShutdownRequest(0);
    }
    static std::unique_ptr<MockedServer> server;
    static std::unique_ptr<std::thread> thread;
    std::unique_ptr<HttpRestApiHandler> handler;
};

std::unique_ptr<MockedServer> HttpRestApiHandlerTest::server = nullptr;
std::unique_ptr<std::thread> HttpRestApiHandlerTest::thread = nullptr;

TEST_F(HttpRestApiHandlerTest, RegexParseReady) {
    std::string request = "/v2/models/dummy/versions/1/ready";
    ovms::HttpRequestComponents comp;

    handler->parseRequestComponents(comp, "GET", request);

    ASSERT_EQ(comp.type, KFS_GetModelReady);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseMetadata) {
    std::string request = "/v2/models/dummy/versions/1/";
    ovms::HttpRequestComponents comp;

    handler->parseRequestComponents(comp, "GET", request);

    ASSERT_EQ(comp.type, KFS_GetModelMetadata);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseInfer) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    ovms::HttpRequestComponents comp;

    handler->parseRequestComponents(comp, "POST", request);

    ASSERT_EQ(comp.type, ovms::KFS_Infer);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, dispatchMetadata) {
    std::string request = "/v2/models/dummy/versions/1/";
    ovms::HttpRequestComponents comp;
    int c = 0;

    handler->registerHandler(KFS_GetModelMetadata, [&](const ovms::HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
        c++;
        return ovms::StatusCode::OK;
    });
    comp.type = ovms::KFS_GetModelMetadata;
    std::string discard;
    handler->dispatchToProcessor(std::string(), &discard, comp);

    ASSERT_EQ(c, 1);
}

TEST_F(HttpRestApiHandlerTest, dispatchReady) {
    std::string request = "/v2/models/dummy/versions/1/ready";
    ovms::HttpRequestComponents comp;
    int c = 0;

    handler->registerHandler(KFS_GetModelReady, [&](const ovms::HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
        c++;
        return ovms::StatusCode::OK;
    });
    comp.type = ovms::KFS_GetModelReady;
    std::string discard;
    handler->dispatchToProcessor(std::string(), &discard, comp);

    ASSERT_EQ(c, 1);
}

TEST_F(HttpRestApiHandlerTest, inferRequest) {
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,10],\"datatype\":\"FP32\",\"data\":[0,1,2,3,4,5,6,7,8,9]}],\"parameters\":{\"binary_data_output\":1}}";

    std::string modelName("dummy");
    std::string modelVersion("1");

    ::inference::ModelInferRequest grpc_request(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body));

    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), modelVersion);
    auto params = grpc_request.parameters();
    auto bdo = params["binary_data_output"];
    ASSERT_EQ(bdo.int64_param(), 1);
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "FP32");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 10);
    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().fp32_contents()) {
        ASSERT_EQ(content, i++);
    }


    std::string output(HttpRestApiHandler::preprocessInferRequest(request_body));
    ASSERT_EQ(output,exp);
}