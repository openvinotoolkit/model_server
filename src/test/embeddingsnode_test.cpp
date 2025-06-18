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

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointEmbeddings = "/v3/embeddings";
    const std::string endpointRerank = "/v3/rerank";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpSuite(std::string& port, std::string& configPath, std::unique_ptr<std::thread>& t) {
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, configPath.c_str());
    }
    static void SetUpTestSuite() {
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
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

class EmbeddingsHttpTest : public V3HttpTest, public ::testing::WithParamInterface<std::string> {
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

TEST_P(EmbeddingsHttpTest, simplePositive) {
    auto modelName = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": "dummyInput"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser),
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

TEST_P(EmbeddingsHttpTest, simplePositiveNoNorm) {
    auto modelName = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(_no_norm",
            "input": "dummyInput"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser),
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

TEST_P(EmbeddingsHttpTest, simplePositiveBase64) {
    auto modelName = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": "dummyInput",
            "encoding_format": "base64"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser),
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

TEST_P(EmbeddingsHttpTest, simplePositiveInt) {
    auto modelName = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": [111, 222, 121]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
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

TEST_P(EmbeddingsHttpTest, simplePositiveMultipleInts) {
    auto modelName = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": [[111, 222, 121], [123, 221, 311]]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
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

TEST_P(EmbeddingsHttpTest, simplePositiveMultipleIntLengths) {
    auto modelName = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7], [7, 8]]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
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

TEST_P(EmbeddingsHttpTest, simplePositiveMultipleStrings) {
    auto modelName = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": ["one", "two"]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
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

TEST_P(EmbeddingsHttpTest, positiveLongInput) {
    auto modelName = GetParam();
    std::string words;
    for (int i = 0; i < 500; i++) {
        words += "hello ";
    }
    std::string requestBody = "{ \"model\": \"" + modelName + "\", \"input\": \"" + words + " \"}";

    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status,
        ovms::StatusCode::OK)
        << status.string();
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    std::cout << response << std::endl;
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_TRUE(d["usage"]["prompt_tokens"].IsInt());
    ASSERT_EQ(d["usage"]["prompt_tokens"], 502);  // 500 words + 2 special tokens
}

TEST_P(EmbeddingsHttpTest, negativeTooLongInput) {
    auto modelName = GetParam();
    std::string words;
    for (int i = 0; i < 511; i++) {
        words += "hello ";
    }
    std::string requestBody = "{ \"model\": \"" + modelName + "\", \"input\": \"" + words + " \"}";

    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status,
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR)
        << status.string();
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    std::cout << response << std::endl;
    ASSERT_EQ(ok.Code(), 1);
    ASSERT_THAT(status.string(), ::testing::HasSubstr("longer than allowed"));
}

TEST_P(EmbeddingsHttpTest, negativeTooLongInputPair) {
    auto modelName = GetParam();
    std::string words;
    for (int i = 0; i < 511; i++) {
        words += "hello ";
    }
    std::string requestBody = "{ \"model\": \"" + modelName + "\", \"input\": [\"" + words + " \", \"short prompt\"]}";

    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status,
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR)
        << status.string();
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    std::cout << response << std::endl;
    ASSERT_EQ(ok.Code(), 1);
    ASSERT_THAT(status.string(), ::testing::HasSubstr("longer than allowed"));
}

TEST_F(EmbeddingsHttpTest, relativePath) {
    std::string requestBody = R"(
        {
            "model": "embeddings_ov_relative",
            "input": [111, 222, 121]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
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

TEST_F(EmbeddingsHttpTest, accessingCalculatorWithInvalidJson) {
    std::string requestBody = R"(
        {
           WRONG JSON
        }
    )";

    // new routing will forward invalid JSON to graph named "embeddings"
    const std::string uriThatMatchesGraphName = "/v3/embeddings";

    headers.clear();  // no sign of application/json
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", uriThatMatchesGraphName, headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(uriThatMatchesGraphName, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingsHttpTestInstances,
    EmbeddingsHttpTest,
    ::testing::Values(
        "embeddings", "embeddings_ov"));

class EmbeddingsExtensionTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointEmbeddings = "/v3/embeddings";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpTestSuite() {
#ifdef _WIN32
        GTEST_SKIP() << "Skipping test because we have no custom extension built for Windows";
#endif
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/embeddings/config_embeddings.json");
        const char* extensionPath = std::filesystem::exists("/opt/libcustom_relu_cpu_extension.so") ? "/opt/libcustom_relu_cpu_extension.so" : "/ovms/src/example/SampleCpuExtension/libcustom_relu_cpu_extension.so";
        server.setShutdownRequest(0);
        randomizeAndEnsureFree(port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath.c_str(),
            (char*)"--cpu_extension",
            (char*)extensionPath,
            (char*)"--port",
            (char*)port.c_str()};
        int argc = 7;
        t.reset(new std::thread([&argc, &argv, &server]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        EnsureServerStartedWithTimeout(server, 15);
    }

    void SetUp() {
#ifdef _WIN32
        GTEST_SKIP() << "Skipping test because we have no custom extension built for Windows";
#endif
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointEmbeddings, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
#ifdef _WIN32
        GTEST_SKIP() << "Skipping test because we have no custom extension built for Windows";
#endif
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void TearDown() {
#ifdef _WIN32
        GTEST_SKIP() << "Skipping test because we have no custom extension built for Windows";
#endif
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
        handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser),
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
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
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
    Status status = handler->dispatchToProcessor(endpointEmbeddings, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
}
