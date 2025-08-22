//*****************************************************************************
// Copyright 2025 Intel Corporation
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

class ListModelsEndpointTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string listModelsEndpoint = "/v3/models";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mediapipe_graph_name_with_slash.json");
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, configPath.c_str());
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "GET", listModelsEndpoint, headers), ovms::StatusCode::OK);
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
std::unique_ptr<std::thread> ListModelsEndpointTest::t;

TEST_F(ListModelsEndpointTest, simplePositive) {
    std::string requestBody = "";
    ASSERT_EQ(
        handler->dispatchToProcessor(listModelsEndpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 2);
    ASSERT_EQ(d["data"][0]["object"], "model");
    ASSERT_EQ(d["data"][0]["id"], "add");
    ASSERT_TRUE(d["data"][0]["created"].IsInt());
    ASSERT_EQ(d["data"][0]["owned_by"], "OVMS");
    ASSERT_EQ(d["data"][1]["object"], "model");
    ASSERT_EQ(d["data"][1]["id"], "my/graph");
    ASSERT_TRUE(d["data"][1]["created"].IsInt());
    ASSERT_EQ(d["data"][1]["owned_by"], "OVMS");
}

TEST_F(ListModelsEndpointTest, positivev3v1) {
    std::string requestBody = "";
    std::string v3v1endpoint = "/v3/v1/models";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", v3v1endpoint, headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(v3v1endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "list");
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 2);
    ASSERT_EQ(d["data"][0]["object"], "model");
    ASSERT_EQ(d["data"][0]["id"], "add");
    ASSERT_TRUE(d["data"][0]["created"].IsInt());
    ASSERT_EQ(d["data"][0]["owned_by"], "OVMS");
    ASSERT_EQ(d["data"][1]["object"], "model");
    ASSERT_EQ(d["data"][1]["id"], "my/graph");
    ASSERT_TRUE(d["data"][1]["created"].IsInt());
    ASSERT_EQ(d["data"][1]["owned_by"], "OVMS");
}

TEST_F(ListModelsEndpointTest, simplePositiveRetrieveModel) {
    std::string requestBody = "";
    std::string endpoint = listModelsEndpoint + "/add";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", endpoint, headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "model");
    ASSERT_EQ(d["id"], "add");
    ASSERT_TRUE(d["created"].IsInt());
    ASSERT_EQ(d["owned_by"], "OVMS");
}

TEST_F(ListModelsEndpointTest, retrieveNonExisitingModel) {
    std::string requestBody = "";
    std::string endpoint = listModelsEndpoint + "/non_existing";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", endpoint, headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MODEL_NOT_LOADED);
    EXPECT_STREQ(response.c_str(), "{\"error\":\"Model not found\"}");
}

TEST_F(ListModelsEndpointTest, retrieveModelEmptyName) {
    std::string requestBody = "";
    std::string endpoint = listModelsEndpoint + "/";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", endpoint, headers), ovms::StatusCode::REST_INVALID_URL);
}

TEST_F(ListModelsEndpointTest, simplePositiveRetrieveGraph) {
    std::string requestBody = "";
    std::string endpoint = listModelsEndpoint + "/my/graph";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", endpoint, headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "model");
    ASSERT_EQ(d["id"], "my/graph");
    ASSERT_TRUE(d["created"].IsInt());
    ASSERT_EQ(d["owned_by"], "OVMS");
}

TEST_F(ListModelsEndpointTest, simplePositiveRetrieveModelv1v3) {
    std::string requestBody = "";
    std::string v3v1endpoint = "/v3/v1/models/add";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", v3v1endpoint, headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(v3v1endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(d["object"], "model");
    ASSERT_EQ(d["id"], "add");
    ASSERT_TRUE(d["created"].IsInt());
    ASSERT_EQ(d["owned_by"], "OVMS");
}
