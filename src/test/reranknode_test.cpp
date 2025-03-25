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
#include "../ov_utils.hpp"
#include "../server.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"  // TODO: Move out together with rerank tests
#include "rapidjson/writer.h"        // TODO: Move out together with rerank tests
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
        const int numberOfRetries = 20;
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < numberOfRetries)) {
        }

        ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED) << "Loading manager takes too long. Server cannot start in 20 seconds.";
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

class RerankHttpTest : public V3HttpTest {
protected:
    static std::unique_ptr<std::thread> t;

public:
    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/rerank/config.json");
        SetUpSuite(port, configPath, t);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> RerankHttpTest::t;

TEST_F(RerankHttpTest, simplePositive) {
    std::string requestBody = R"(
        {
            "model": "rerank",
            "query": "What is the capital of the United States?",
            "documents": ["Carson City is the capital city of the American state of Nevada.",
                        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
                        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
                        "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."]
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointRerank, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_TRUE(d.HasMember("results"));
    ASSERT_TRUE(d["results"].IsArray());
    ASSERT_EQ(d["results"].Size(), 5);
    for (auto& v : d["results"].GetArray()) {
        ASSERT_TRUE(v.IsObject());
        EXPECT_EQ(v.Size(), 2);
        ASSERT_TRUE(v.HasMember("index"));
        EXPECT_TRUE(v["index"].IsInt());
        ASSERT_TRUE(v.HasMember("relevance_score"));
        EXPECT_TRUE(v["relevance_score"].IsDouble());
    }
}

TEST_F(RerankHttpTest, positiveTopN) {
    std::string requestBody = R"(
        {
            "model": "rerank",
            "query": "What is the capital of the United States?",
            "top_n": 3,
            "documents": ["Carson City is the capital city of the American state of Nevada.",
                        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
                        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
                        "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."]
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointRerank, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_TRUE(d.HasMember("results"));
    ASSERT_TRUE(d["results"].IsArray());
    ASSERT_EQ(d["results"].Size(), 3);
    for (auto& v : d["results"].GetArray()) {
        ASSERT_TRUE(v.IsObject());
        EXPECT_EQ(v.Size(), 2);
        ASSERT_TRUE(v.HasMember("index"));
        EXPECT_TRUE(v["index"].IsInt());
        ASSERT_TRUE(v.HasMember("relevance_score"));
        EXPECT_TRUE(v["relevance_score"].IsDouble());
    }
}

TEST_F(RerankHttpTest, positiveReturnDocuments) {
    std::string requestBody = R"(
        {
            "model": "rerank",
            "query": "What is the capital of the United States?",
            "return_documents": true,
            "documents": ["Carson City is the capital city of the American state of Nevada.",
                        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
                        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
                        "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."]
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointRerank, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_TRUE(d.HasMember("results"));
    ASSERT_TRUE(d["results"].IsArray());
    ASSERT_EQ(d["results"].Size(), 5);
    for (auto& v : d["results"].GetArray()) {
        ASSERT_TRUE(v.IsObject());
        EXPECT_EQ(v.Size(), 3);
        ASSERT_TRUE(v.HasMember("index"));
        EXPECT_TRUE(v["index"].IsInt());
        ASSERT_TRUE(v.HasMember("relevance_score"));
        EXPECT_TRUE(v["relevance_score"].IsDouble());
        ASSERT_TRUE(v.HasMember("document"));
        EXPECT_TRUE(v["document"].IsObject());
        EXPECT_EQ(v["document"].Size(), 1);
        ASSERT_TRUE(v["document"].HasMember("text"));
        EXPECT_TRUE(v["document"]["text"].IsString());
    }
}

class RerankWithParamsHttpTest : public V3HttpTest {
protected:
    static std::unique_ptr<std::thread> t;

public:
    const size_t MAX_POSITION_EMBEDDINGS = 12;
    const size_t MAX_ALLOWED_CHUNKS = 4;

    static void SetUpTestSuite() {
        std::string port = "9173";
        /*
            Setup with:
            max_position_embeddings: 12
            max_allowed_chunks: 4

            Meaning query is trimmed to contain at most 6 tokens (half of max_position_embeddings)
            And maximum number of documents or chunks (after chunking process) can be 4
            Allowed space for chunk is 12-6-4=2 tokens
        */
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/rerank/with_params/config.json");
        SetUpSuite(port, configPath, t);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> RerankWithParamsHttpTest::t;

TEST_F(RerankWithParamsHttpTest, PositiveMaxAllowedChunksNotExceeded) {
    // Create a JSON document
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

    // Populate the JSON document with data
    document.AddMember("model", "rerank", allocator);
    document.AddMember("query", "What is the capital of the United States?", allocator);  // Will be trimmed to 6 tokens

    rapidjson::Value documents(rapidjson::kArrayType);

    for (size_t i = 0; i < MAX_ALLOWED_CHUNKS; i++) {
        documents.PushBack("Test", allocator);  // Short document to not exceed 2 token space for chunks
    }
    document.AddMember("documents", documents, allocator);

    // Convert JSON document to string
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> wr(buffer);
    document.Accept(wr);

    std::string requestBody = buffer.GetString();
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointRerank, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(RerankWithParamsHttpTest, MaxAllowedChunksExceededByDocumentsBeforeChunking) {
    // Create a JSON document
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

    // Populate the JSON document with data
    document.AddMember("model", "rerank", allocator);
    document.AddMember("query", "What is the capital of the United States?", allocator);  // Will be trimmed to 6 tokens

    // Test fail due number of documents exceeding number of max chunks
    rapidjson::Value documents(rapidjson::kArrayType);

    for (size_t i = 0; i < MAX_ALLOWED_CHUNKS + 1; i++) {
        documents.PushBack("Test", allocator);  // Short document to not exceed 2 token space for chunks
    }
    document.AddMember("documents", documents, allocator);

    // Convert JSON document to string
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> wr(buffer);
    document.Accept(wr);

    std::string requestBody = buffer.GetString();
    auto status = handler->dispatchToProcessor(endpointRerank, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    ASSERT_THAT(status.string(), ::testing::HasSubstr("Number of documents exceeds max_allowed_chunks"));  // 5 because we prepared 1 document more than allowed
}

TEST_F(RerankWithParamsHttpTest, MaxAllowedChunksExceededAfterChunking) {
    // Create a JSON document
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

    // Populate the JSON document with data
    document.AddMember("model", "rerank", allocator);
    document.AddMember("query", "What is the capital of the United States?", allocator);  // Will be trimmed to 6 tokens

    // Test fail due number of documents exceeding number of max chunks
    rapidjson::Value documents(rapidjson::kArrayType);

    // There are 4 documents - which is supported by max_allowed_chunks,
    // but one document is long and chunking will the number of allowed documents (4)
    for (size_t i = 0; i < MAX_ALLOWED_CHUNKS - 1; i++) {
        documents.PushBack("Test", allocator);  // Short document to not exceed 2 token space for chunks
    }
    documents.PushBack("This is a long document that will be chunked", allocator);  // Long document
    document.AddMember("documents", documents, allocator);

    // Convert JSON document to string
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> wr(buffer);
    document.Accept(wr);

    std::string requestBody = buffer.GetString();
    auto status = handler->dispatchToProcessor(endpointRerank, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_THAT(status.string(), ::testing::HasSubstr("Chunking failed: exceeding max_allowed_chunks after chunking limit: 4; actual: 8"));  // 8 because of the last document which was chunked to 5 documents, 3 + 5 = 8
}

class RerankWithInvalidParamsHttpTest : public V3HttpTest {
protected:
    static std::unique_ptr<std::thread> t;

public:
    const size_t MAX_POSITION_EMBEDDINGS = 8;
    const size_t MAX_ALLOWED_CHUNKS = 4;

    static void SetUpTestSuite() {
        std::string port = "9173";
        /*
            Setup with:
            max_position_embeddings: 8
            max_allowed_chunks: 4

            This is invalid setup since there is reservation for 4 special tokens and space for query is max half of max_position_embeddings (4) - meaning 0 token space for document
        */
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/rerank/with_params/invalid_config.json");
        SetUpSuite(port, configPath, t);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> RerankWithInvalidParamsHttpTest::t;

TEST_F(RerankWithInvalidParamsHttpTest, AnyRequestNegativeWithInvalidSetup) {
    // Create a JSON document
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

    // Populate the JSON document with data
    document.AddMember("model", "rerank", allocator);
    document.AddMember("query", "What is the capital of the United States?", allocator);

    rapidjson::Value documents(rapidjson::kArrayType);

    for (size_t i = 0; i < MAX_ALLOWED_CHUNKS; i++) {
        documents.PushBack("Test", allocator);  // Not even 1 token documents fit the space
    }
    document.AddMember("documents", documents, allocator);

    // Convert JSON document to string
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> wr(buffer);
    document.Accept(wr);

    std::string requestBody = buffer.GetString();
    auto status = handler->dispatchToProcessor(endpointRerank, requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    ASSERT_THAT(status.string(), ::testing::HasSubstr("max_position_embeddings should be larger than 2 * NUMBER_OF_SPECIAL_TOKENS"));
}
