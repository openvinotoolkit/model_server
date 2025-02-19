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
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <stdint.h>

#include "../config.hpp"
#include "../grpcservermodule.hpp"
#include "../http_async_writer_interface.hpp"
#include "../http_rest_api_handler.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "../version.hpp"
#include "test_utils.hpp"

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
        randomizePort(port);
        char* argv[] = {
            (char*)"OpenVINO Model Server",
            (char*)"--model_name",
            (char*)"dummy",
            (char*)"--model_path",
            (char*)getGenericFullPathForSrcTest("/ovms/src/test/dummy").c_str(),
            (char*)"--log_level",
            (char*)"DEBUG",
            (char*)"--batch_size",
            (char*)"auto",
            (char*)"--rest_port",
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

class HttpRestApiHandlerWithScalarModelTest : public HttpRestApiHandlerTest {
public:
    static void SetUpTestSuite() {
        HttpRestApiHandlerTest::server = std::make_unique<MockedServer>();
        std::string port = "9000";
        randomizePort(port);
        char* argv[] = {
            (char*)"OpenVINO Model Server",
            (char*)"--model_name",
            (char*)"scalar",
            (char*)"--model_path",
            (char*)getGenericFullPathForSrcTest("/ovms/src/test/scalar").c_str(),
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
};

class HttpRestApiHandlerWithDynamicModelTest : public HttpRestApiHandlerTest {
public:
    static void SetUpTestSuite() {
        HttpRestApiHandlerTest::server = std::make_unique<MockedServer>();
        std::string port = "9000";
        randomizePort(port);
        char* argv[] = {
            (char*)"OpenVINO Model Server",
            (char*)"--model_name",
            (char*)"dummy",
            (char*)"--model_path",
            (char*)getGenericFullPathForSrcTest("/ovms/src/test/dummy").c_str(),
            (char*)"--shape",
            (char*)"(-1,-1)",
            (char*)"--log_level",
            (char*)"DEBUG",
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
};

class HttpRestApiHandlerWithStringModelTest : public HttpRestApiHandlerTest {
public:
    static void SetUpTestSuite() {
        HttpRestApiHandlerTest::server = std::make_unique<MockedServer>();
        std::string port = "9000";
        randomizePort(port);
        char* argv[] = {
            (char*)"OpenVINO Model Server",
            (char*)"--model_name",
            (char*)"string",
            (char*)"--model_path",
            (char*)getGenericFullPathForSrcTest("/ovms/src/test/passthrough_string").c_str(),
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
};

std::unique_ptr<MockedServer> HttpRestApiHandlerTest::server = nullptr;
std::unique_ptr<std::thread> HttpRestApiHandlerTest::thread = nullptr;

#if (PYTHON_DISABLE == 0)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

static void testInference(int headerLength, std::string& request_body, std::unique_ptr<HttpRestApiHandler>& handler, const std::string endpoint = "/v2/models/mediapipeAdd/versions/1/infer") {
    std::vector<std::pair<std::string, std::string>> headers;
    std::pair<std::string, std::string> binaryInputsHeader{"inference-header-content-length", std::to_string(headerLength)};
    headers.emplace_back(binaryInputsHeader);

    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);

    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(doc.HasParseError(), false);

    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    ASSERT_EQ(output.Size(), 10);
    auto datatype = doc["outputs"].GetArray()[0].GetObject()["datatype"].GetString();
    for (auto& data : output) {
        if (strcmp(datatype, "BOOL") == 0) {
            ASSERT_EQ(data.GetBool(), true);
        } else {
            ASSERT_EQ(data.GetFloat(), 2);
        }
    }
}

static void testInferenceNegative(int headerLength, std::string& request_body, std::unique_ptr<HttpRestApiHandler>& handler, ovms::Status processorStatus) {
    std::string request = "/v2/models/mediapipeAdd/versions/1/infer";

    std::vector<std::pair<std::string, std::string>> headers;
    std::pair<std::string, std::string> binaryInputsHeader{"inference-header-content-length", std::to_string(headerLength)};
    headers.emplace_back(binaryInputsHeader);

    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request, headers), ovms::StatusCode::OK);

    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), processorStatus);
}

class HttpRestApiHandlerWithMediapipe : public ::testing::TestWithParam<std::string> {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }

        handler = std::make_unique<HttpRestApiHandler>(server, 5);
    }

    void SetUp() {
        SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_python_summator.json").c_str());
    }

    void TearDown() {
        handler.reset();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

class HttpRestApiHandlerWithMediapipePassthrough : public HttpRestApiHandlerWithMediapipe {
protected:
    void SetUp() {
        SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mp_pytensor_passthrough.json").c_str());
    }
};

TEST_P(HttpRestApiHandlerWithMediapipe, inferRequestWithSupportedPrecision) {
    std::string datatype = GetParam();
    std::string tensor1 = "{\"name\":\"in1\",\"shape\":[1,10],\"datatype\":\"" + datatype + "\", \"data\": [1,1,1,1,1,1,1,1,1,1]}";
    std::string tensor2 = "{\"name\":\"in2\",\"shape\":[1,10],\"datatype\":\"" + datatype + "\", \"data\": [1,1,1,1,1,1,1,1,1,1]}";

    std::string request_body = "{\"inputs\":[" + tensor1 + ", " + tensor2 + "]}";
    int headerLength = request_body.length();

    testInference(headerLength, request_body, handler);
}

TEST_F(HttpRestApiHandlerWithMediapipe, inferRequestFP16) {
    std::string tensor1 = "{\"name\":\"in1\",\"shape\":[1,10],\"datatype\":\"FP16\", \"data\": [1,1,1,1,1,1,1,1,1,1]}";
    std::string tensor2 = "{\"name\":\"in2\",\"shape\":[1,10],\"datatype\":\"FP16\", \"data\": [1,1,1,1,1,1,1,1,1,1]}";

    std::string request_body = "{\"inputs\":[" + tensor1 + ", " + tensor2 + "]}";
    int headerLength = request_body.length();
    // Supported only when data is in binary extension
    testInferenceNegative(headerLength, request_body, handler, ovms::StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(HttpRestApiHandlerWithMediapipe, inferRequestBF16) {
    std::string tensor1 = "{\"name\":\"in1\",\"shape\":[1,10],\"datatype\":\"BF16\", \"data\": [1,1,1,1,1,1,1,1,1,1]}";
    std::string tensor2 = "{\"name\":\"in2\",\"shape\":[1,10],\"datatype\":\"BF16\", \"data\": [1,1,1,1,1,1,1,1,1,1]}";

    std::string request_body = "{\"inputs\":[" + tensor1 + ", " + tensor2 + "]}";
    int headerLength = request_body.length();
    // Supported only when data is in binary extension
    testInferenceNegative(headerLength, request_body, handler, ovms::StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(HttpRestApiHandlerWithMediapipe, inferRequestBOOL) {
    std::string tensor1 = "{\"name\":\"in1\",\"shape\":[1,10],\"datatype\":\"BOOL\", \"data\": [true,true,true,true,true,true,true,true,true,true]}";
    std::string tensor2 = "{\"name\":\"in2\",\"shape\":[1,10],\"datatype\":\"BOOL\", \"data\": [true,true,true,true,true,true,true,true,true,true]}";

    std::string request_body = "{\"inputs\":[" + tensor1 + ", " + tensor2 + "]}";
    int headerLength = request_body.length();

    testInference(headerLength, request_body, handler);
}

TEST_F(HttpRestApiHandlerWithMediapipe, inferRequestFP32DataInJsonAndBinaryExtension) {
    // 10 element array of floats: [1,1,1,1,1,1,1,1,1,1]
    std::string binaryData{
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F)};

    std::string tensor1 = "{\"name\":\"in1\",\"shape\":[1,10],\"datatype\":\"FP32\",\"parameters\":{\"binary_data_size\":40}}";
    std::string tensor2 = "{\"name\":\"in2\",\"shape\":[1,10],\"datatype\":\"FP32\", \"data\": [1,1,1,1,1,1,1,1,1,1]}";

    std::string request_body = "{\"inputs\":[" + tensor1 + ", " + tensor2 + "]}";
    int headerLength = request_body.length();

    request_body += binaryData;
    request_body += binaryData;

    testInferenceNegative(headerLength, request_body, handler, ovms::StatusCode::INVALID_MESSAGE_STRUCTURE);
}

TEST_F(HttpRestApiHandlerWithMediapipe, inferRequestFP32BinaryExtension) {
    std::string binaryData{
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F),
        static_cast<char>(0x00), static_cast<char>(0x00), static_cast<char>(0x80), static_cast<char>(0x3F)};

    std::string tensor1 = "{\"name\":\"in1\",\"shape\":[1,10],\"datatype\":\"FP32\",\"parameters\":{\"binary_data_size\":40}}";
    std::string tensor2 = "{\"name\":\"in2\",\"shape\":[1,10],\"datatype\":\"FP32\",\"parameters\":{\"binary_data_size\":40}}";

    std::string request_body = "{\"inputs\":[" + tensor1 + ", " + tensor2 + "]}";
    int headerLength = request_body.length();

    request_body += binaryData;
    request_body += binaryData;

    testInference(headerLength, request_body, handler);
}

std::vector<std::string> supportedDatatypes = {"FP32", "FP64", "INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"};

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    HttpRestApiHandlerWithMediapipe,
    ::testing::ValuesIn(supportedDatatypes),
    [](const ::testing::TestParamInfo<HttpRestApiHandlerWithMediapipe::ParamType>& info) {
        return info.param;
    });

TEST_F(HttpRestApiHandlerWithMediapipePassthrough, inferRequestBYTES) {
    std::string request = "/v2/models/mpPytensorPassthrough/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"in\",\"shape\":[3],\"datatype\":\"BYTES\", \"data\": [\"abc\", \"def\", \"ghi\"]}]}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_FALSE(doc.HasParseError());
    ASSERT_TRUE(doc["outputs"][0].GetObject().HasMember("data"));
    ASSERT_TRUE(doc["outputs"][0].GetObject()["data"].IsArray());
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    std::vector<std::string> expectedStrings{"abc", "def", "ghi"};
    ASSERT_EQ(output.Size(), expectedStrings.size());
    for (size_t i = 0; i < expectedStrings.size(); i++) {
        ASSERT_TRUE(output[i].IsString());
        ASSERT_EQ(output[i].GetString(), expectedStrings[i]);
    }
}

#pragma GCC diagnostic pop
#endif

TEST_F(HttpRestApiHandlerTest, MetricsParameters) {
    std::string request = "/metrics?test=test";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);
    ASSERT_EQ(comp.type, ovms::Metrics);
}

TEST_F(HttpRestApiHandlerTest, GetModelMetadataWithLongVersion) {
    std::string request = "/v1/models/dummy/versions/72487667423532349025128558057";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::MODEL_VERSION_MISSING);
}

TEST_F(HttpRestApiHandlerTest, GetModelMetadataWithEscapedPath) {
    std::string request = "/v1/models/..iO!.0?E*/versions/1/metadata";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::OK);
}

TEST_F(HttpRestApiHandlerTest, UnsupportedMethods) {
    ovms::HttpRequestComponents comp;
    std::string request = "/v2/models/dummy/ready";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), StatusCode::REST_UNSUPPORTED_METHOD);
    request = "/v2/models/dummy";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), StatusCode::REST_UNSUPPORTED_METHOD);
    request = "/v2/models/dummy/infer";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", request), StatusCode::REST_UNSUPPORTED_METHOD);
    request = "/v2";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), StatusCode::REST_UNSUPPORTED_METHOD);
    request = "/v2/health/live";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), StatusCode::REST_UNSUPPORTED_METHOD);
    request = "/v2/health/ready";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), StatusCode::REST_UNSUPPORTED_METHOD);
}
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
    std::pair<std::string, std::string> binaryInputsHeader{"inference-header-content-length", "15"};
    headers.emplace_back(binaryInputsHeader);
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request, headers), StatusCode::OK);
}

TEST_F(HttpRestApiHandlerTest, RegexParseInferWithBinaryInputsSizeNegative) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    ovms::HttpRequestComponents comp;
    std::vector<std::pair<std::string, std::string>> headers;
    std::pair<std::string, std::string> binaryInputsHeader{"inference-header-content-length", "-15"};
    headers.emplace_back(binaryInputsHeader);
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request, headers), StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_INVALID);
}

TEST_F(HttpRestApiHandlerTest, RegexParseInferWithBinaryInputsSizeNotInt) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    ovms::HttpRequestComponents comp;
    std::vector<std::pair<std::string, std::string>> headers;
    std::pair<std::string, std::string> binaryInputsHeader{"inference-header-content-length", "value"};
    headers.emplace_back(binaryInputsHeader);
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request, headers), StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_INVALID);
}

TEST_F(HttpRestApiHandlerTest, dispatchMetadata) {
    std::string request = "/v2/models/dummy/versions/1";
    ovms::HttpRequestComponents comp;
    int c = 0;

    handler->registerHandler(KFS_GetModelMetadata, [&](const std::string_view uri, const ovms::HttpRequestComponents& request_components, std::string& response, const std::string& request_body, ovms::HttpResponseComponents& response_components, std::shared_ptr<ovms::HttpAsyncWriter>) {
        c++;
        return ovms::StatusCode::OK;
    });
    comp.type = ovms::KFS_GetModelMetadata;
    std::string discard;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    handler->dispatchToProcessor("", std::string(), &discard, comp, responseComponents, writer);

    ASSERT_EQ(c, 1);
}

TEST_F(HttpRestApiHandlerTest, dispatchReady) {
    std::string request = "/v2/models/dummy/versions/1/ready";
    ovms::HttpRequestComponents comp;
    int c = 0;

    handler->registerHandler(KFS_GetModelReady, [&](const std::string_view, const ovms::HttpRequestComponents& request_components, std::string& response, const std::string& request_body, ovms::HttpResponseComponents& response_components, std::shared_ptr<ovms::HttpAsyncWriter> writer) {
        c++;
        return ovms::StatusCode::OK;
    });
    comp.type = ovms::KFS_GetModelReady;
    std::string discard;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    handler->dispatchToProcessor("", std::string(), &discard, comp, responseComponents, writer);

    ASSERT_EQ(c, 1);
}

TEST_F(HttpRestApiHandlerTest, modelMetadataRequest) {
    std::string request = "/v2/models/dummy/versions/1";
    ovms::HttpRequestComponents comp;

    handler->parseRequestComponents(comp, "GET", request);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", std::string(), &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(std::string(doc["name"].GetString()), "dummy");
    ASSERT_EQ(std::string(doc["versions"].GetArray()[0].GetString()), "1");
    ASSERT_EQ(std::string(doc["platform"].GetString()), "OpenVINO");

    ASSERT_EQ(doc["inputs"].GetArray().Size(), 1);
    ASSERT_EQ(std::string(doc["inputs"].GetArray()[0].GetObject()["name"].GetString()), "b");
    ASSERT_EQ(std::string(doc["inputs"].GetArray()[0].GetObject()["datatype"].GetString()), "FP32");
    ASSERT_EQ(doc["inputs"].GetArray()[0].GetObject()["shape"].GetArray().Size(), 2);
    ASSERT_EQ(doc["inputs"].GetArray()[0].GetObject()["shape"].GetArray()[0].GetInt(), 1);
    ASSERT_EQ(doc["inputs"].GetArray()[0].GetObject()["shape"].GetArray()[1].GetInt(), 10);

    ASSERT_EQ(doc["outputs"].GetArray().Size(), 1);
    ASSERT_EQ(std::string(doc["outputs"].GetArray()[0].GetObject()["name"].GetString()), "a");
    ASSERT_EQ(std::string(doc["outputs"].GetArray()[0].GetObject()["datatype"].GetString()), "FP32");
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray().Size(), 2);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[0].GetInt(), 1);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[1].GetInt(), 10);
    ASSERT_EQ(doc["rt_info"].GetArray().Size(), 1);
    ASSERT_EQ(std::string(doc["rt_info"].GetObject()["model_info"].GetObject()["resolution"].GetObject()["height"].GetString()), "200");
    ASSERT_EQ(std::string(doc["rt_info"].GetObject()["model_info"].GetObject()["precision"].GetString()), "FP16");
}

// Disabled due to bad cast when getting RT info
#ifndef _WIN32
TEST_F(HttpRestApiHandlerWithScalarModelTest, modelMetadataRequest) {
    std::string request = "/v2/models/scalar/versions/1";
    ovms::HttpRequestComponents comp;

    handler->parseRequestComponents(comp, "GET", request);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", std::string(), &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(std::string(doc["name"].GetString()), "scalar");
    ASSERT_EQ(std::string(doc["versions"].GetArray()[0].GetString()), "1");
    ASSERT_EQ(std::string(doc["platform"].GetString()), "OpenVINO");

    ASSERT_EQ(std::string(doc["inputs"].GetArray()[0].GetObject()["name"].GetString()), SCALAR_MODEL_INPUT_NAME);
    ASSERT_EQ(std::string(doc["inputs"].GetArray()[0].GetObject()["datatype"].GetString()), "FP32");
    ASSERT_EQ(doc["inputs"].GetArray()[0].GetObject()["shape"].GetArray().Size(), 0);

    ASSERT_EQ(std::string(doc["outputs"].GetArray()[0].GetObject()["name"].GetString()), SCALAR_MODEL_OUTPUT_NAME);
    ASSERT_EQ(std::string(doc["outputs"].GetArray()[0].GetObject()["datatype"].GetString()), "FP32");
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray().Size(), 0);
}
#endif

TEST_F(HttpRestApiHandlerTest, inferRequestWithMultidimensionalMatrix) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[2,10],\"datatype\":\"FP32\",\"data\":[[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    ASSERT_EQ(output.Size(), 20);
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
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(doc["model_name"].GetString(), std::string("dummy"));
    ASSERT_EQ(doc["id"].GetString(), std::string("1"));
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    ASSERT_EQ(output.Size(), 10);
    int i = 1;
    for (auto& data : output) {
        ASSERT_EQ(data.GetFloat(), i++);
    }
}

TEST_F(HttpRestApiHandlerWithScalarModelTest, inferRequestScalar) {
    std::string request = "/v2/models/scalar/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"model_scalar_input\",\"shape\":[],\"datatype\":\"FP32\",\"data\":[4.1]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(doc["model_name"].GetString(), std::string("scalar"));
    ASSERT_EQ(doc["id"].GetString(), std::string("1"));
    ASSERT_TRUE(doc["outputs"].GetArray()[0].GetObject().HasMember("data"));
    ASSERT_TRUE(doc["outputs"].GetArray()[0].GetObject().HasMember("shape"));
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    ASSERT_EQ(output.Size(), 1);
    ASSERT_EQ(output[0].GetFloat(), 4.1f);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray().Size(), 0);
}

TEST_F(HttpRestApiHandlerWithDynamicModelTest, inferRequestZeroBatch) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[0,10],\"datatype\":\"FP32\",\"data\":[]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(doc["model_name"].GetString(), std::string("dummy"));
    ASSERT_EQ(doc["id"].GetString(), std::string("1"));
    ASSERT_TRUE(doc["outputs"].GetArray()[0].GetObject().HasMember("data"));
    ASSERT_TRUE(doc["outputs"].GetArray()[0].GetObject().HasMember("shape"));
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    ASSERT_EQ(output.Size(), 0);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray().Size(), 2);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[0], 0);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[1], 10);
}

TEST_F(HttpRestApiHandlerWithDynamicModelTest, inferRequestZeroDim) {
    std::string request = "/v2/models/dummy/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,0],\"datatype\":\"FP32\",\"data\":[]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(doc["model_name"].GetString(), std::string("dummy"));
    ASSERT_EQ(doc["id"].GetString(), std::string("1"));
    ASSERT_TRUE(doc["outputs"].GetArray()[0].GetObject().HasMember("data"));
    ASSERT_TRUE(doc["outputs"].GetArray()[0].GetObject().HasMember("shape"));
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    ASSERT_EQ(output.Size(), 0);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray().Size(), 2);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[0], 1);
    ASSERT_EQ(doc["outputs"].GetArray()[0].GetObject()["shape"].GetArray()[1], 0);
}

TEST_F(HttpRestApiHandlerTest, inferPreprocess) {
    std::string request_body("{\"inputs\":[{\"name\":\"b\",\"shape\":[1,10],\"datatype\":\"FP32\",\"data\":[0,1,2,3,4,5,6,7,8,9]}],\"parameters\":{\"binary_data_output\":1, \"bool_test\":true, \"string_test\":\"test\"}}");

    ::KFSRequest grpc_request;
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

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params = grpc_request.inputs()[0].parameters();
    ASSERT_EQ(params.count("binary_data_size"), 1);
    ASSERT_EQ(params["binary_data_size"].int64_param(), 4);
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT8");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.raw_input_contents()[0]) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT8_twoInputs) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":4}}, {\"name\":\"c\",\"shape\":[1,4],\"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":4}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);

    ASSERT_EQ(grpc_request.inputs_size(), 2);
    ASSERT_EQ(grpc_request.raw_input_contents_size(), 2);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    auto params1 = grpc_request.inputs()[0].parameters();
    ASSERT_EQ(params1.count("binary_data_size"), 1);
    ASSERT_EQ(params1["binary_data_size"].int64_param(), 4);
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT8");

    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);

    int i = 0;
    for (auto content : grpc_request.raw_input_contents()[0]) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
    auto params2 = grpc_request.inputs()[1].parameters();
    ASSERT_EQ(params2.count("binary_data_size"), 1);
    ASSERT_EQ(params2["binary_data_size"].int64_param(), 4);
    ASSERT_EQ(grpc_request.inputs()[1].name(), "c");
    ASSERT_EQ(grpc_request.inputs()[1].datatype(), "INT8");

    ASSERT_EQ(grpc_request.inputs()[1].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[1].shape()[1], 4);

    i = 0;
    for (auto content : grpc_request.raw_input_contents()[1]) {
        ASSERT_EQ(content, i++);
    }
    ASSERT_EQ(i, 4);
}

static void assertSingleBinaryInput(const std::string& modelName, const std::optional<uint64_t>& modelVersion, ::KFSRequest& grpc_request) {
    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    ASSERT_EQ(grpc_request.raw_input_contents().size(), 1);

    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
}

static void assertBinaryInputsBYTES(const std::string& modelName, const std::optional<uint64_t>& modelVersion, ::KFSRequest& grpc_request, std::string binaryData) {
    assertSingleBinaryInput(modelName, modelVersion, grpc_request);

    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "BYTES");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    uint32_t dataSize = *((int32_t*)(binaryData.data()));
    ASSERT_EQ(dataSize, binaryData.size() - sizeof(uint32_t));
    ASSERT_EQ(grpc_request.raw_input_contents()[0].size(), binaryData.size());
    ASSERT_EQ(grpc_request.raw_input_contents()[0], binaryData);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES) {
    std::string binaryData{0x04, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1],\"datatype\":\"BYTES\",\"parameters\":{\"binary_data_size\":8}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsBYTES(modelName, modelVersion, grpc_request, binaryData);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES_Batch2) {
    std::string binaryData{0x04, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x02};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[2],\"datatype\":\"BYTES\",\"parameters\":{\"binary_data_size\":20}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertSingleBinaryInput(modelName, modelVersion, grpc_request);

    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "BYTES");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 2);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].size(), binaryData.size());
    ASSERT_EQ(grpc_request.raw_input_contents()[0], binaryData);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES_noBinaryDataSizeParameter) {
    std::string binaryData{0x04, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1],\"datatype\":\"BYTES\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsBYTES(modelName, modelVersion, grpc_request, binaryData);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES_noBinaryDataSizeParameter_twoInputs) {
    // This scenario will fail because binary_data_size parameter is required for BYTES datatype when there are more than 1 inputs in request
    std::string binaryData{0x04, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1],\"datatype\":\"BYTES\"}, {\"name\":\"c\",\"shape\":[1],\"datatype\":\"BYTES\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES_dataInInvalidFormat) {
    std::string binaryData{0x11, 0x11, 0x11, 0x11, 0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1],\"datatype\":\"BYTES\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    // Data correctness will be checked at the stage of grpc input deserialization
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES_sizeInBytesBiggerThanBuffer) {
    std::string binaryData{0x16, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1],\"datatype\":\"BYTES\",\"parameters\":{\"binary_data_size\":8}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    // Data correctness will be checked at the stage of grpc input deserialization
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBYTES_BinaryDataSizeBiggerThanActualBuffer) {
    std::string binaryData{0x16, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1],\"datatype\":\"BYTES\",\"parameters\":{\"binary_data_size\":8}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_BUFFER_EXCEEDED);
    // Data correctness will be checked at the stage of grpc input deserialization
}

static void assertBinaryInputsINT16(const std::string& modelName, const std::optional<uint64_t>& modelVersion, ::KFSRequest& grpc_request, std::string binaryData) {
    assertSingleBinaryInput(modelName, modelVersion, grpc_request);

    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT16");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);
    size_t i;
    for (i = 0; i < (grpc_request.raw_input_contents()[0].size() / sizeof(int16_t)); i++) {
        ASSERT_EQ(((int16_t*)grpc_request.raw_input_contents()[0].data())[i], i);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT16) {
    std::string binaryData{0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT16\",\"parameters\":{\"binary_data_size\":8}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsINT16(modelName, modelVersion, grpc_request, binaryData);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT16ZeroDim) {
    std::string binaryData{/*[1,0]=no data*/};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,0],\"datatype\":\"INT16\",\"parameters\":{\"binary_data_size\":0}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    ASSERT_EQ(grpc_request.raw_input_contents().size(), 1);
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT16");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 0);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].size(), 0);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT16ZeroDim_noBinaryDataSizeParameter) {
    std::string binaryData{/*[1,0]=no data*/};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,0],\"datatype\":\"INT16\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    ASSERT_EQ(grpc_request.raw_input_contents().size(), 1);
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT16");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 0);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].size(), 0);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT16Scalar) {
    std::string binaryData{0x14, 0x15};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[],\"datatype\":\"INT16\",\"parameters\":{\"binary_data_size\":2}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    ASSERT_EQ(grpc_request.raw_input_contents().size(), 1);
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT16");
    ASSERT_EQ(grpc_request.inputs()[0].shape().size(), 0);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].size(), 2);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].data()[0], 0x14);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].data()[1], 0x15);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT16Scalar_noBinaryDataSizeParameter) {
    std::string binaryData{0x14, 0x15};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[],\"datatype\":\"INT16\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    ASSERT_EQ(grpc_request.inputs_size(), 1);
    ASSERT_EQ(grpc_request.model_name(), modelName);
    ASSERT_EQ(grpc_request.model_version(), std::to_string(modelVersion.value()));
    ASSERT_EQ(grpc_request.raw_input_contents().size(), 1);
    ASSERT_EQ(grpc_request.inputs()[0].name(), "b");
    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT16");
    ASSERT_EQ(grpc_request.inputs()[0].shape().size(), 0);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].size(), 2);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].data()[0], 0x14);
    ASSERT_EQ(grpc_request.raw_input_contents()[0].data()[1], 0x15);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT16_noBinaryDataSizeParameter) {
    std::string binaryData{0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT16\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsINT16(modelName, modelVersion, grpc_request, binaryData);
}

static void assertBinaryInputsINT32(const std::string& modelName, const std::optional<uint64_t>& modelVersion, ::KFSRequest& grpc_request, std::string binaryData) {
    assertSingleBinaryInput(modelName, modelVersion, grpc_request);

    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT32");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);
    size_t i;
    for (i = 0; i < (grpc_request.raw_input_contents()[0].size() / sizeof(int32_t)); i++) {
        ASSERT_EQ(((int32_t*)grpc_request.raw_input_contents()[0].data())[i], i);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT32) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":16}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsINT32(modelName, modelVersion, grpc_request, binaryData);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT32_noBinaryDataSizeParameter) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsINT32(modelName, modelVersion, grpc_request, binaryData);
}

static void assertBinaryInputsINT64(const std::string& modelName, const std::optional<uint64_t>& modelVersion, ::KFSRequest& grpc_request, std::string binaryData) {
    assertSingleBinaryInput(modelName, modelVersion, grpc_request);

    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "INT64");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);
    size_t i;
    for (i = 0; i < (grpc_request.raw_input_contents()[0].size() / sizeof(int64_t)); i++) {
        ASSERT_EQ(((int64_t*)grpc_request.raw_input_contents()[0].data())[i], i);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT64) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT64\",\"parameters\":{\"binary_data_size\":32}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsINT64(modelName, modelVersion, grpc_request, binaryData);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT64_noBinaryDataSizeParameter) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT64\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsINT64(modelName, modelVersion, grpc_request, binaryData);
}

static void assertBinaryInputsFP32(const std::string& modelName, const std::optional<uint64_t>& modelVersion, ::KFSRequest& grpc_request) {
    assertSingleBinaryInput(modelName, modelVersion, grpc_request);

    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "FP32");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);
    size_t i;
    for (i = 0; i < (grpc_request.raw_input_contents()[0].size() / sizeof(float)); i++) {
        ASSERT_EQ(((float*)grpc_request.raw_input_contents()[0].data())[i], i);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsFP32) {
    float values[] = {0.0, 1.0, 2.0, 3.0};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"FP32\",\"parameters\":{\"binary_data_size\":16}}]}";
    request_body.append((char*)values, 16);

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - 16);
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsFP32(modelName, modelVersion, grpc_request);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsFP32_noBinaryDataSizeParameter) {
    float values[] = {0.0, 1.0, 2.0, 3.0};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"FP32\"}]}";
    request_body.append((char*)values, 16);

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - 16);
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsFP32(modelName, modelVersion, grpc_request);
}

static void assertBinaryInputsFP64(const std::string& modelName, const std::optional<uint64_t>& modelVersion, ::KFSRequest& grpc_request) {
    assertSingleBinaryInput(modelName, modelVersion, grpc_request);

    ASSERT_EQ(grpc_request.inputs()[0].datatype(), "FP64");
    ASSERT_EQ(grpc_request.inputs()[0].shape()[0], 1);
    ASSERT_EQ(grpc_request.inputs()[0].shape()[1], 4);
    size_t i;
    for (i = 0; i < (grpc_request.raw_input_contents()[0].size() / sizeof(double)); i++) {
        ASSERT_EQ(((double*)grpc_request.raw_input_contents()[0].data())[i], i);
    }
    ASSERT_EQ(i, 4);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsFP64) {
    double values[] = {0.0, 1.0, 2.0, 3.0};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"FP64\",\"parameters\":{\"binary_data_size\":32}}]}";
    request_body.append((char*)values, 32);

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - 32);
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsFP64(modelName, modelVersion, grpc_request);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsFP64_noBinaryDataSizeParameter) {
    double values[] = {0.0, 1.0, 2.0, 3.0};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"FP64\"}]}";
    request_body.append((char*)values, 32);

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - 32);
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    assertBinaryInputsFP64(modelName, modelVersion, grpc_request);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBinaryDataAndContentField) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"data\":[0,1,2,3,4,5,6,7,8,9], \"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":4}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request), ovms::StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBufferSmallerThanExpected) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":16}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_BUFFER_EXCEEDED);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsBufferSmallerThanExpected_noBinaryDataSizeParameter) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\"}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_BUFFER_EXCEEDED);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsInferenceHeaderContentLengthSmallerThanJsonBody) {
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":true}}]}";
    int inferenceHeaderContentLength = request_body.size() - 1;

    ::KFSRequest grpc_request;
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::JSON_INVALID);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsInferenceHeaderContentLengthLargerThanJsonBody) {
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":true}}]}";
    int inferenceHeaderContentLength = request_body.size() + 1;

    ::KFSRequest grpc_request;
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_EXCEEDED);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsInvalidBinaryDataSizeParameter) {
    std::string binaryData{0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[1,4],\"datatype\":\"INT32\",\"parameters\":{\"binary_data_size\":true}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsINT8BatchSize2) {
    // Format with string binary_data_size parameter containing list of sizes is now deprecated
    std::string binaryData{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    std::string request_body = "{\"inputs\":[{\"name\":\"b\",\"shape\":[2,4],\"datatype\":\"INT8\",\"parameters\":{\"binary_data_size\":\"4, 4\"}}]}";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    ASSERT_EQ(HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength), ovms::StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID);
}

TEST_F(HttpRestApiHandlerTest, binaryInputsEmptyRequest) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    std::string request_body = "";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    auto status = HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - Error: The document is empty. Offset: 0");
}

TEST_F(HttpRestApiHandlerTest, binaryInputsInvalidJson) {
    std::string binaryData{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    std::string request_body = R"({"inputs": notValid})";
    request_body += binaryData;

    ::KFSRequest grpc_request;
    int inferenceHeaderContentLength = (request_body.size() - binaryData.size());
    auto status = HttpRestApiHandler::prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, inferenceHeaderContentLength);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - Error: Invalid value. Offset: 12");
}

TEST_F(HttpRestApiHandlerWithStringModelTest, invalidPrecision) {
    std::string request = "/v2/models/string/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"my_name\",\"shape\":[2],\"datatype\":\"FP32\",\"data\":[\"Hello\", \"World\"]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(HttpRestApiHandlerWithStringModelTest, invalidShape) {
    std::string request = "/v2/models/string/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"my_name\",\"shape\":[3],\"datatype\":\"BYTES\",\"data\":[\"Hello\", \"World\"]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(HttpRestApiHandlerWithStringModelTest, invalidShape_noData) {
    std::string request = "/v2/models/string/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"my_name\",\"shape\":[1],\"datatype\":\"BYTES\"}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::INVALID_VALUE_COUNT);
}

TEST_F(HttpRestApiHandlerWithStringModelTest, invalidShape_emptyData) {
    std::string request = "/v2/models/string/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"my_name\",\"shape\":[1],\"datatype\":\"BYTES\",\"data\":[]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::INVALID_VALUE_COUNT);
}

void assertStringMetadataOutput(rapidjson::Document& doc) {
    ASSERT_TRUE(doc.HasMember("model_name"));
    ASSERT_TRUE(doc["model_name"].IsString());
    ASSERT_EQ(doc["model_name"].GetString(), std::string("string"));
    ASSERT_TRUE(doc.HasMember("id"));
    ASSERT_TRUE(doc["id"].IsString());
    ASSERT_EQ(doc["id"].GetString(), std::string("1"));
    ASSERT_TRUE(doc.HasMember("outputs"));
    ASSERT_TRUE(doc["outputs"].IsArray());
    ASSERT_EQ(doc["outputs"].Size(), 1);
    ASSERT_TRUE(doc["outputs"][0].IsObject());
    ASSERT_TRUE(doc["outputs"][0].GetObject().HasMember("shape"));
    ASSERT_TRUE(doc["outputs"][0].GetObject()["shape"].IsArray());
    ASSERT_EQ(doc["outputs"][0].GetObject()["shape"].GetArray().Size(), 1);
    ASSERT_EQ(doc["outputs"][0].GetObject()["shape"].GetArray()[0], 2);
    ASSERT_TRUE(doc["outputs"][0].GetObject().HasMember("datatype"));
    ASSERT_TRUE(doc["outputs"][0].GetObject()["datatype"].IsString());
    ASSERT_EQ(std::string(doc["outputs"][0].GetObject()["datatype"].GetString()), std::string("BYTES"));
    ASSERT_TRUE(doc["outputs"][0].GetObject().HasMember("name"));
    ASSERT_TRUE(doc["outputs"][0].GetObject()["name"].IsString());
    ASSERT_EQ(std::string(doc["outputs"][0].GetObject()["name"].GetString()), std::string("my_name"));
}

TEST_F(HttpRestApiHandlerWithStringModelTest, positivePassthrough) {
    std::string request = "/v2/models/string/versions/1/infer";
    std::string request_body = "{\"inputs\":[{\"name\":\"my_name\",\"shape\":[2],\"datatype\":\"BYTES\",\"data\":[\"Hello\", \"World\"]}], \"id\":\"1\"}";
    ovms::HttpRequestComponents comp;

    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", request), ovms::StatusCode::OK);
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->dispatchToProcessor("", request_body, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_FALSE(doc.HasParseError());
    assertStringMetadataOutput(doc);
    ASSERT_TRUE(doc["outputs"][0].GetObject().HasMember("data"));
    ASSERT_TRUE(doc["outputs"][0].GetObject()["data"].IsArray());
    auto output = doc["outputs"].GetArray()[0].GetObject()["data"].GetArray();
    std::vector<std::string> expectedStrings{"Hello", "World"};
    ASSERT_EQ(output.Size(), expectedStrings.size());
    for (size_t i = 0; i < expectedStrings.size(); i++) {
        ASSERT_TRUE(output[i].IsString());
        ASSERT_EQ(output[i].GetString(), expectedStrings[i]);
    }
}

TEST_F(HttpRestApiHandlerWithStringModelTest, positivePassthrough_binaryInput) {
    std::string request = "/v2/models/string/versions/1/infer";
    std::string request_body = R"(
        {
            "id": "1",
            "inputs": [{
                "name": "my_name",
                "shape": [2],
                "datatype": "BYTES",
                "parameters": {
                    "binary_data_size": 15
                }
            }],
            "outputs": [{
                "name": "my_name",
                "parameters": {
                    "binary_data": true
                }
            }]
        }
    )";
    size_t jsonEnd = request_body.size();

    std::string binaryInputData{0x05, 0x00, 0x00, 0x00, 'H', 'e', 'l', 'l', 'o', 0x02, 0x00, 0x00, 0x00, '1', '2'};
    request_body += binaryInputData;

    std::vector<std::pair<std::string, std::string>> headers{
        {"inference-header-content-length", std::to_string(jsonEnd)},
        {"Content-Type", "application/json"},
    };
    ovms::HttpResponseComponents responseComponents;
    std::string output;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ASSERT_EQ(handler->processRequest("POST", request, request_body, &headers, &output, responseComponents, writer), ovms::StatusCode::OK);
    ASSERT_TRUE(responseComponents.inferenceHeaderContentLength.has_value());
    ASSERT_EQ(responseComponents.inferenceHeaderContentLength.value(), 272);

    // Data test
    std::string binaryOutputData = output.substr(
        responseComponents.inferenceHeaderContentLength.value(),
        output.size() - responseComponents.inferenceHeaderContentLength.value());
    ASSERT_EQ(binaryOutputData.size(), binaryInputData.size());
    ASSERT_EQ(std::memcmp(binaryInputData.data(), binaryOutputData.data(), binaryOutputData.size()), 0);

    // Metadata test
    rapidjson::Document doc;
    std::string response = output.substr(0, responseComponents.inferenceHeaderContentLength.value());
    doc.Parse(response.c_str());
    ASSERT_FALSE(doc.HasParseError());
    assertStringMetadataOutput(doc);
    ASSERT_FALSE(doc["outputs"][0].GetObject().HasMember("data"));
}

TEST_F(HttpRestApiHandlerTest, serverReady) {
    ovms::HttpRequestComponents comp;
    comp.type = ovms::KFS_GetServerReady;
    std::string request;
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ovms::Status status = handler->dispatchToProcessor("", request, &response, comp, responseComponents, writer);

    ASSERT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(HttpRestApiHandlerTest, serverLive) {
    ovms::HttpRequestComponents comp;
    comp.type = ovms::KFS_GetServerLive;
    std::string request;
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ovms::Status status = handler->dispatchToProcessor("", request, &response, comp, responseComponents, writer);

    ASSERT_EQ(status, ovms::StatusCode::OK);
}

TEST_F(HttpRestApiHandlerTest, serverMetadata) {
    ovms::HttpRequestComponents comp;
    comp.type = ovms::KFS_GetServerMetadata;
    std::string request;
    std::string response;
    ovms::HttpResponseComponents responseComponents;
    std::shared_ptr<ovms::HttpAsyncWriter> writer{nullptr};
    ovms::Status status = handler->dispatchToProcessor("", request, &response, comp, responseComponents, writer);

    rapidjson::Document doc;
    doc.Parse(response.c_str());
    ASSERT_EQ(std::string(doc["name"].GetString()), PROJECT_NAME);
    ASSERT_EQ(std::string(doc["version"].GetString()), PROJECT_VERSION);
}
