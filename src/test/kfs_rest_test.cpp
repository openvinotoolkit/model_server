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
#include <rapidjson/document.h>

#include "../config.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../version.hpp"

using ovms::Config;
using ovms::HttpRestApiHandler;
using ovms::KFS_GetModelMetadata;
using ovms::KFS_GetModelReady;
using ovms::Module;
using ovms::ModuleState;
using ovms::SERVABLE_MANAGER_MODULE_NAME;
using ovms::Server;
using ovms::StatusCode;

namespace {
class MockedServer : public Server {
public:
    MockedServer() = default;
};
}  // namespace

class HttpRestApiHandlerTest : public ::testing::Test {
protected:
    const std::string modelName{"dummy"};
    const std::optional<uint64_t> modelVersion{1};

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
            (char*)"--batch_size",
            (char*)"auto",
            (char*)"--port",
            (char*)port.c_str(),
            nullptr};
        thread = std::make_unique<std::thread>(
            [&argv]() {
                ASSERT_EQ(EXIT_SUCCESS, server->start(11, argv));
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

TEST_F(HttpRestApiHandlerTest, RegexParseReadyWithImplicitVersion) {
    std::string request = "/v2/models/dummy/ready";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);

    ASSERT_EQ(comp.type, KFS_GetModelReady);
    ASSERT_EQ(comp.model_version, std::nullopt);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseReady) {
    std::string request = "/v2/models/dummy/versions/1/ready";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);

    ASSERT_EQ(comp.type, KFS_GetModelReady);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseMetadataWithImplicitVersion) {
    std::string request = "/v2/models/dummy";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);

    ASSERT_EQ(comp.type, KFS_GetModelMetadata);
    ASSERT_EQ(comp.model_version, std::nullopt);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseMetadata) {
    std::string request = "/v2/models/dummy/versions/1";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);

    ASSERT_EQ(comp.type, KFS_GetModelMetadata);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseInferWithImplicitVersion) {
    std::string request = "/v2/models/dummy/infer";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), StatusCode::OK);

    ASSERT_EQ(comp.type, ovms::KFS_Infer);
    ASSERT_EQ(comp.model_version, std::nullopt);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseInfer) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), StatusCode::OK);

    ASSERT_EQ(comp.type, ovms::KFS_Infer);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST_F(HttpRestApiHandlerTest, RegexParseServerMetadata) {
    std::string request = "/v2";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);

    ASSERT_EQ(comp.type, ovms::KFS_GetServerMetadata);
}

TEST_F(HttpRestApiHandlerTest, RegexParseServerReady) {
    std::string request = "/v2/health/ready";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);

    ASSERT_EQ(comp.type, ovms::KFS_GetServerReady);
}

TEST_F(HttpRestApiHandlerTest, RegexParseServerLive) {
    std::string request = "/v2/health/live";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);

    ASSERT_EQ(comp.type, ovms::KFS_GetServerLive);
}

TEST_F(HttpRestApiHandlerTest, RegexParseInferWithBinaryInputs) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    ovms::HttpRequestComponents comp;
    std::vector<std::pair<std::string, std::string>> headers;
    std::pair<std::string, std::string> binaryInputsHeader{"Inference-Header-Content-Length", "15"};
    headers.emplace_back(binaryInputsHeader);
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request, headers), StatusCode::OK);
}

TEST_F(HttpRestApiHandlerTest, RegexParseInferWithBinaryInputsSizeNegative) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    ovms::HttpRequestComponents comp;
    std::vector<std::pair<std::string, std::string>> headers;
    std::pair<std::string, std::string> binaryInputsHeader{"Inference-Header-Content-Length", "-15"};
    headers.emplace_back(binaryInputsHeader);
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request, headers), StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_INVALID);
}

TEST_F(HttpRestApiHandlerTest, RegexParseInferWithBinaryInputsSizeNotInt) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    ovms::HttpRequestComponents comp;
    std::vector<std::pair<std::string, std::string>> headers;
    std::pair<std::string, std::string> binaryInputsHeader{"Inference-Header-Content-Length", "value"};
    headers.emplace_back(binaryInputsHeader);
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request, headers), StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_INVALID);
}

TEST_F(HttpRestApiHandlerTest, dispatchMetadata) {
    std::string request = "/v2/models/dummy/versions/1";
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

TEST_F(HttpRestApiHandlerTest, modelMetadataRequest) {
    std::string request = "/v2/models/dummy/versions/1";
    ovms::HttpRequestComponents comp;

    handler->parseRequestComponents(comp, "GET", request);
    std::string response;
    handler->dispatchToProcessor(std::string(), &response, comp);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(std::string(doc["name"].GetString()), "dummy");
    ASSERT_EQ(std::string(doc["versions"].GetArray()[0].GetString()), "1");
    ASSERT_EQ(std::string(doc["platform"].GetString()), "OpenVINO");

    ASSERT_EQ(std::string(doc["inputs"].GetArray()[0].GetObject()["name"].GetString()), "b");
    ASSERT_EQ(std::string(doc["inputs"].GetArray()[0].GetObject()["datatype"].GetString()), "FP32");
    ASSERT_EQ(doc["inputs"].GetArray()[0].GetObject()["shape"].GetArray()[0].GetInt(), 1);
    ASSERT_EQ(doc["inputs"].GetArray()[0].GetObject()["shape"].GetArray()[1].GetInt(), 10);

    ASSERT_EQ(std::string(doc["outputs"].GetArray()[0].GetObject()["name"].GetString()), "a");
    ASSERT_EQ(std::string(doc["outputs"].GetArray()[0].GetObject()["datatype"].GetString()), "FP32");
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[0].GetInt(), 1);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[1].GetInt(), 10);
}

TEST_F(HttpRestApiHandlerTest, inferRequestWithMultidimensionalMatrix) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[2,10],\"datatype\":\"FP32\",\"data\":[[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ASSERT_EQ(handler->dispatchToProcessor(request_body, &response, comp), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    int i = 1;
    for (auto& data : output) {
        ASSERT_EQ(data.GetFloat(), i++);
    }
}

TEST_F(HttpRestApiHandlerTest, inferRequest) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,10],\"datatype\":\"FP32\",\"data\":[0,1,2,3,4,5,6,7,8,9]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ASSERT_EQ(handler->dispatchToProcessor(request_body, &response, comp), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(doc["model_name"].GetString(), std::string("dummy"));
    ASSERT_EQ(doc["id"].GetString(), std::string("1"));
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    int i = 1;
    for (auto& data : output) {
        ASSERT_EQ(data.GetFloat(), i++);
    }
}

TEST_F(HttpRestApiHandlerTest, inferPreprocess) {
    std::string request_body("{\"inputs\":[{\"name\":\"b\",\"shape\":[1,10],\"datatype\":\"FP32\",\"data\":[0,1,2,3,4,5,6,7,8,9]}],\"parameters\":{\"binary_data_output\":1, \"bool_test\":true, \"string_test\":\"test\"}}");

    ::inference::ModelInferRequest grpc_request;
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(params["binary_data_output"].int64_param(), 1);
    ASSERT_EQ(params["bool_test"].bool_param(), true);
    ASSERT_EQ(params["string_test"].string_param(), "test");
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "FP32");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 10);
    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().fp32_contents()) {
        ASSERT_EQ(content, i++);
    }
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT8) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":4}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT8");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().int_contents()) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"BYTES\",\"parameters\":{\"binary_data_size\":4}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "BYTES");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);
    ASSERT_EQ(grpc_request.inputs()[0].contents().bytes_contents()[0], binaryData);
    ASSERT_EQ(grpc_request.inputs()[0].contents().bytes_contents_size(), 1);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT16) {
    std::string binaryData{0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT16\",\"parameters\":{\"binary_data_size\":8}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT16");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().int_contents()) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT32) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":16}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT32");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().int_contents()) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT64) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT64\",\"parameters\":{\"binary_data_size\":32}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT64");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().int64_contents()) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsFP32) {
    float values[] = {0.0, 1.0, 2.0, 3.0};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"FP32\",\"parameters\":{\"binary_data_size\":16}}]}";
    request_body.append((char*)values, 16);

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - 16);
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "FP32");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().fp32_contents()) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsFP64) {
    double values[] = {0.0, 1.0, 2.0, 3.0};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"FP64\",\"parameters\":{\"binary_data_size\":32}}]}";
    request_body.append((char*)values, 32);

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - 32);
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "FP64");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().fp64_contents()) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBinaryDataAndContentField) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"data\":[0,1,2,3,4,5,6,7,8,9], \"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":4}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request), ovms::StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBufferSmallerThanExpected) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":16}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_BUFFER_EXCEEDED);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsInvalidBinaryDataSizeParameter) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":true}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT8BatchSize2) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[2,4],\"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":\"4, 4\"}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.parameters();
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT8");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 2);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.inputs()[0].contents().int_contents()) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 8);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBinaryDataSizeStringParameterInvalid) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[2,4],\"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":\"a, 4\"}}]}";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsEmptyRequest) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    std::string request_body = "";
    request_body += binaryData;

    ::inference::ModelInferRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::JSON_INVALID);
}

TEST_F(HttpRestApiHandlerTest, serverReady) {
    ovms::HttpRequestComponents comp;
    comp.type = ovms::KFS_GetServerReady;
    std::string request;
    std::string response;
    ovms::Status status = handler->dispatchToProcessor(request, &response, comp);

    ASSERT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(HttpRestApiHandlerTest, serverLive) {
    ovms::HttpRequestComponents comp;
    comp.type = ovms::KFS_GetServerLive;
    std::string request;
    std::string response;
    ovms::Status status = handler->dispatchToProcessor(request, &response, comp);

    ASSERT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(HttpRestApiHandlerTest, serverMetadata) {
    ovms::HttpRequestComponents comp;
    comp.type = ovms::KFS_GetServerMetadata;
    std::string request;
    std::string response;
    ovms::Status status = handler->dispatchToProcessor(request, &response, comp);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(std::string(doc["name"].GetString()), PROJECT_NAME);
    ASSERT_EQ(std::string(doc["version"].GetString()), PROJECT_VERSION);
}
