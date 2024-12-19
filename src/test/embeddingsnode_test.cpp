//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../http_rest_api_handler.hpp"
#include "../server.hpp"
#include "rapidjson/document.h"
#include "test_http_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

class V3HttpTest : public ::testing::Test {
public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpointEmbeddings = "/v3/embeddings";
    const std::string endpointRerank = "/v3/rerank";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpSuite(std::string& port, std::string& configPath, std::unique_ptr<std::thread>& t) {
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, configPath.c_str());
        auto start = std::chrono::high_resolution_clock::now();
        const int numberOfRetries = 5;
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < numberOfRetries)) {
        }
    }
    static void SetUpTestSuite() {
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointEmbeddings, headers), ovms::StatusCode::OK);
    }

    static void TearDownSuite(std::unique_ptr<std::thread>& t) {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void TearDown() {
        handler.reset();
    }
};

class EmbeddingsHttpTest : public V3HttpTest {
protected:
    static std::unique_ptr<std::thread> t;

public:
    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/embeddings/config_embeddings.json");
        SetUpSuite(port, configPath, t);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> EmbeddingsHttpTest::t;

const int EMBEDDING_OUTPUT_SIZE = 384;

TEST_F(EmbeddingsHttpTest, simplePositive) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": "dummyInput"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 1);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsArray());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
    ASSERT_TRUE(d.HasMember("usage"));
    ASSERT_TRUE(d["usage"].IsObject());
    ASSERT_TRUE(d["usage"].HasMember("prompt_tokens"));
    ASSERT_TRUE(d["usage"]["prompt_tokens"].IsInt());
    ASSERT_TRUE(d["usage"].HasMember("total_tokens"));
    ASSERT_TRUE(d["usage"]["total_tokens"].IsInt());
    double sum = 0;
    for (auto& value : d["data"][0]["embedding"].GetArray()) {
        sum += value.GetDouble() * value.GetDouble();
    }
    double norm = std::max(std::sqrt(sum), double(1e-12));
    ASSERT_NEAR(norm, 1.0, 1e-6);
    ASSERT_EQ(d["data"][0]["index"], 0);
}

TEST_F(EmbeddingsHttpTest, simplePositiveNoNorm) {
    std::string requestBody = R"(
        {
            "model": "embeddings_no_norm",
            "input": "dummyInput"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 1);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsArray());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
    ASSERT_TRUE(d.HasMember("usage"));
    ASSERT_TRUE(d["usage"].IsObject());
    ASSERT_TRUE(d["usage"].HasMember("prompt_tokens"));
    ASSERT_TRUE(d["usage"]["prompt_tokens"].IsInt());
    ASSERT_TRUE(d["usage"].HasMember("total_tokens"));
    ASSERT_TRUE(d["usage"]["total_tokens"].IsInt());
    double sum = 0;
    for (auto& value : d["data"][0]["embedding"].GetArray()) {
        sum += value.GetDouble() * value.GetDouble();
    }
    double norm = std::max(std::sqrt(sum), double(1e-12));
    ASSERT_NEAR(norm, 9.5, 1);  // norm of a not normalized vector
    ASSERT_EQ(d["data"][0]["index"], 0);
}

TEST_F(EmbeddingsHttpTest, simplePositiveBase64) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": "dummyInput",
            "encoding_format": "base64"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 1);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsString());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), ((4 * (EMBEDDING_OUTPUT_SIZE * sizeof(float)) / 3) + 3) & ~3);  // In base64 each symbol represents 3/4 of a byte rounded up
    ASSERT_EQ(d["data"][0]["index"], 0);
    ASSERT_TRUE(d.HasMember("usage"));
    ASSERT_TRUE(d["usage"].IsObject());
    ASSERT_TRUE(d["usage"].HasMember("prompt_tokens"));
    ASSERT_TRUE(d["usage"]["prompt_tokens"].IsInt());
    ASSERT_TRUE(d["usage"].HasMember("total_tokens"));
    ASSERT_TRUE(d["usage"]["total_tokens"].IsInt());
}

TEST_F(EmbeddingsHttpTest, simplePositiveInt) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [111, 222, 121]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status,
        ovms::StatusCode::OK)
        << status.string();
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 1);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsArray());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
}

TEST_F(EmbeddingsHttpTest, simplePositiveMultipleInts) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[111, 222, 121], [123, 221, 311]]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status,
        ovms::StatusCode::OK)
        << status.string();
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 2);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsArray());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
    ASSERT_EQ(d["data"][1]["object"], "embedding");
    ASSERT_TRUE(d["data"][1]["embedding"].IsArray());
    ASSERT_EQ(d["data"][1]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
}

TEST_F(EmbeddingsHttpTest, simplePositiveMultipleIntLengths) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7], [7, 8]]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status,
        ovms::StatusCode::OK)
        << status.string();
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 3);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsArray());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
    ASSERT_EQ(d["data"][1]["object"], "embedding");
    ASSERT_TRUE(d["data"][1]["embedding"].IsArray());
    ASSERT_EQ(d["data"][1]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
    ASSERT_TRUE(d["data"][2]["embedding"].IsArray());
    ASSERT_EQ(d["data"][2]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
}

TEST_F(EmbeddingsHttpTest, simplePositiveMultipleStrings) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "two"]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status,
        ovms::StatusCode::OK)
        << status.string();
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 2);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsArray());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
    ASSERT_EQ(d["data"][1]["object"], "embedding");
    ASSERT_TRUE(d["data"][1]["embedding"].IsArray());
    ASSERT_EQ(d["data"][1]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
}

class EmbeddingsExtensionTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpointEmbeddings = "/v3/embeddings";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpTestSuite() {
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/embeddings/config_embeddings.json");
        const char* extensionPath = std::filesystem::exists("/opt/libcustom_relu_cpu_extension.so") ? "/opt/libcustom_relu_cpu_extension.so" : "/ovms/src/example/SampleCpuExtension/libcustom_relu_cpu_extension.so";
        server.setShutdownRequest(0);
        randomizePort(port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath.c_str(),
            (char*)"--cpu_extension",
            (char*)extensionPath,
            (char*)"--port ",
            (char*)port.c_str()};
        int argc = 5;
        t.reset(new std::thread([&argc, &argv, &server]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        auto start = std::chrono::high_resolution_clock::now();
        const int numberOfRetries = 5;
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < numberOfRetries)) {
        }
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointEmbeddings, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void TearDown() {
        handler.reset();
    }
};
std::unique_ptr<std::thread> EmbeddingsExtensionTest::t;
TEST_F(EmbeddingsExtensionTest, simplePositive) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": "dummyInput"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 1);
    ASSERT_EQ(d["data"][0]["object"], "embedding");
    ASSERT_TRUE(d["data"][0]["embedding"].IsArray());
    ASSERT_EQ(d["data"][0]["embedding"].Size(), EMBEDDING_OUTPUT_SIZE);
    ASSERT_EQ(d["data"][0]["index"], 0);
}

class EmbeddingsInvalidConfigTest : public V3HttpTest {
protected:
    static std::unique_ptr<std::thread> t;

public:
    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/embeddings/invalid_config_embeddings.json");
        SetUpSuite(port, configPath, t);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> EmbeddingsInvalidConfigTest::t;

TEST_F(EmbeddingsInvalidConfigTest, simpleNegative) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": "dummyInput"
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
}

class EmbeddingsInvalidTokenizerConfigTest : public V3HttpTest {
protected:
    static std::unique_ptr<std::thread> t;

public:
    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/embeddings/invalid_config_tokenizer.json");
        SetUpSuite(port, configPath, t);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> EmbeddingsInvalidTokenizerConfigTest::t;

TEST_F(EmbeddingsInvalidTokenizerConfigTest, simpleNegative) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": "dummyInput"
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
}
