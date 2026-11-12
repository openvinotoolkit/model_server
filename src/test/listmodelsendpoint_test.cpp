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
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/http_rest_api_handler.hpp"
#include "src/mediapipe_internal/mediapipegraphdefinition.hpp"
#include "src/server.hpp"
#include "src/servable_management/modelmanager.hpp"
#include "src/servable_management/servablemanagermodule.hpp"
#include "rapidjson/document.h"
#include "test_http_utils.hpp"
#include "test_utils.hpp"
#include "test_with_temp_dir.hpp"
#include "platform_utils.hpp"

using namespace ovms;

class ListModelsEndpointTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string listModelsEndpoint = "/v1/models";
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
    ASSERT_EQ(d.MemberCount(), 2);
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

class ListModelsEndpointMetadataTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;
    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;

    static void SetUpTestSuite() {
        std::string port = "9179";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/llm/config_models_endpoint.json");
        ::SetUpServer(t, ovms::Server::instance(), port, configPath.c_str(), 60);
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void SetUp() override {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        handler = std::make_unique<ovms::HttpRestApiHandler>(ovms::Server::instance(), 5);
    }

    void TearDown() override {
        handler.reset();
    }

    void get(const std::string& endpoint, rapidjson::Document& document) {
        ovms::HttpRequestComponents comp;
        ovms::HttpResponseComponents responseComponents;
        std::string response;
        ASSERT_EQ(handler->parseRequestComponents(comp, "GET", endpoint, headers), ovms::StatusCode::OK);
        ASSERT_EQ(handler->dispatchToProcessor(endpoint, "", &response, comp, responseComponents, writer, multiPartParser), ovms::StatusCode::OK);
        ASSERT_FALSE(document.Parse(response.c_str()).HasParseError());
    }

    static void expectStandardFields(const rapidjson::Value& entry) {
        ASSERT_TRUE(entry.HasMember("id"));
        ASSERT_TRUE(entry.HasMember("object"));
        ASSERT_TRUE(entry.HasMember("created"));
        ASSERT_TRUE(entry.HasMember("owned_by"));
        ASSERT_TRUE(entry["id"].IsString());
        EXPECT_STREQ(entry["object"].GetString(), "model");
        EXPECT_TRUE(entry["created"].IsInt64());
        EXPECT_STREQ(entry["owned_by"].GetString(), "OVMS");
    }
};
std::unique_ptr<std::thread> ListModelsEndpointMetadataTest::t;

TEST_F(ListModelsEndpointMetadataTest, ListContainsMaxModelLenOnlyForLLM) {
    rapidjson::Document d;
    get("/v1/models", d);
    ASSERT_TRUE(d["data"].IsArray());
    ASSERT_EQ(d["data"].Size(), 2);
    bool llmFound = false;
    bool passthroughFound = false;
    for (const auto& entry : d["data"].GetArray()) {
        ASSERT_NO_FATAL_FAILURE(expectStandardFields(entry));
        const std::string id = entry["id"].GetString();
        if (id == "lm_cb_regular") {
            llmFound = true;
            ASSERT_TRUE(entry.HasMember("max_model_len"));
            ASSERT_TRUE(entry["max_model_len"].IsUint());
            EXPECT_GT(entry["max_model_len"].GetUint(), 0u);
        } else if (id == "passthrough") {
            passthroughFound = true;
            EXPECT_FALSE(entry.HasMember("max_model_len"));
        }
    }
    EXPECT_TRUE(llmFound);
    EXPECT_TRUE(passthroughFound);
}

TEST_F(ListModelsEndpointMetadataTest, RetrieveLLMContainsMaxModelLen) {
    rapidjson::Document d;
    get("/v1/models/lm_cb_regular", d);
    ASSERT_NO_FATAL_FAILURE(expectStandardFields(d));
    EXPECT_STREQ(d["id"].GetString(), "lm_cb_regular");
    ASSERT_TRUE(d.HasMember("max_model_len"));
    ASSERT_TRUE(d["max_model_len"].IsUint());
    EXPECT_GT(d["max_model_len"].GetUint(), 0u);
}

TEST_F(ListModelsEndpointMetadataTest, RetrieveNonLLMHasNoMaxModelLen) {
    rapidjson::Document d;
    get("/v1/models/passthrough", d);
    ASSERT_NO_FATAL_FAILURE(expectStandardFields(d));
    EXPECT_STREQ(d["id"].GetString(), "passthrough");
    EXPECT_FALSE(d.HasMember("max_model_len"));
}

TEST_F(ListModelsEndpointMetadataTest, SleepingLLMStillReportsMaxModelLen) {
    ovms::Server& server = ovms::Server::instance();
    auto& manager = dynamic_cast<const ovms::ServableManagerModule*>(server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME))->getServableManager();
    auto& factory = manager.getMediapipeFactory();
    auto* definition = factory.findDefinitionByName("lm_cb_regular");
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(factory.putToSleepDefinition("lm_cb_regular"), ovms::StatusCode::OK);
    ASSERT_EQ(definition->getStateCode(), ovms::PipelineDefinitionStateCode::SLEEPING);

    rapidjson::Document list;
    get("/v1/models", list);
    bool llmFound = false;
    for (const auto& entry : list["data"].GetArray()) {
        if (std::string(entry["id"].GetString()) == "lm_cb_regular") {
            llmFound = true;
            ASSERT_TRUE(entry.HasMember("max_model_len"));
            EXPECT_GT(entry["max_model_len"].GetUint(), 0u);
        }
    }
    EXPECT_TRUE(llmFound);
    rapidjson::Document retrieved;
    get("/v1/models/lm_cb_regular", retrieved);
    ASSERT_TRUE(retrieved.HasMember("max_model_len"));
    EXPECT_GT(retrieved["max_model_len"].GetUint(), 0u);

    // Reporting metadata must not wake the graph up
    EXPECT_EQ(definition->getStateCode(), ovms::PipelineDefinitionStateCode::SLEEPING);
    ASSERT_EQ(factory.wakeUpDefinition("lm_cb_regular", manager), ovms::StatusCode::OK);
}

class ListModelsEndpointIdleManagementTest : public TestWithTempDir {
protected:
    std::unique_ptr<std::thread> t;
    std::unique_ptr<ovms::HttpRestApiHandler> handler;
    std::string configFilePath;
    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    const std::string listModelsEndpoint = "/v1/models";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;

    static std::string makeConfig(bool includeModel, bool includeMediapipe) {
        std::string models;
        if (includeModel) {
            models = R"({"config": {"name": "dummy", "base_path": ")" + getGenericFullPathForSrcTest("/ovms/src/test/dummy") + R"("}})";
        }
        std::string graphs;
        if (includeMediapipe) {
            graphs = R"({"name": "passthroughGraph", "graph_path": ")" + getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/graphpassthrough.pbtxt") + R"("})";
        }
        return R"({"model_config_list": [)" + models + R"(], "mediapipe_config_list": [)" + graphs + "]}";
    }

    void writeConfig(const std::string& content) {
        std::ofstream ofs(configFilePath);
        ofs << content;
    }

    void SetUp() override {
        TestWithTempDir::SetUp();
        configFilePath = directoryPath + "/config.json";
        const bool includeModel = true;
        const bool includeMediapipe = true;
        writeConfig(makeConfig(includeModel, includeMediapipe));
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();

        ovms::Server& server = ovms::Server::instance();
        std::string port = "9178";
        ::SetUpServerWithExtraArgs(t, server, port, configFilePath.c_str(), {"--idle_unload_timeout_seconds", "30"});
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
    }

    void TearDown() override {
        handler.reset();
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        TestWithTempDir::TearDown();
    }

    ovms::ModelManager& getManager() {
        ovms::Server& server = ovms::Server::instance();
        return dynamic_cast<const ovms::ServableManagerModule*>(server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME))->getServableManager();
    }

    void reloadConfig(const std::string& content) {
        writeConfig(content);
        std::string response;
        auto status = handler->processConfigReloadRequest(response, getManager());
        ASSERT_TRUE(status.ok()) << status.string();
    }

    std::vector<std::string> listModelIds() {
        ovms::HttpRequestComponents comp;
        ovms::HttpResponseComponents responseComponents;
        std::string response;
        EXPECT_EQ(handler->parseRequestComponents(comp, "GET", listModelsEndpoint, headers), ovms::StatusCode::OK);
        EXPECT_EQ(handler->dispatchToProcessor(listModelsEndpoint, "", &response, comp, responseComponents, writer, multiPartParser), ovms::StatusCode::OK);
        rapidjson::Document d;
        d.Parse(response.c_str());
        EXPECT_FALSE(d.HasParseError());
        std::vector<std::string> ids;
        if (d.HasParseError() || !d.HasMember("data") || !d["data"].IsArray()) {
            return ids;
        }
        for (const auto& entry : d["data"].GetArray()) {
            ids.emplace_back(entry["id"].GetString());
        }
        return ids;
    }
};

TEST_F(ListModelsEndpointIdleManagementTest, EachServableListedOnce) {
    EXPECT_THAT(listModelIds(), ::testing::UnorderedElementsAre("dummy", "passthroughGraph"));
}

TEST_F(ListModelsEndpointIdleManagementTest, RetiredModelNotListed) {
    const bool includeModel = false;
    const bool includeMediapipe = true;
    reloadConfig(makeConfig(includeModel, includeMediapipe));
    EXPECT_THAT(listModelIds(), ::testing::UnorderedElementsAre("passthroughGraph"));
}

TEST_F(ListModelsEndpointIdleManagementTest, RetiredMediapipeNotListed) {
    const bool includeModel = true;
    const bool includeMediapipe = false;
    reloadConfig(makeConfig(includeModel, includeMediapipe));
    EXPECT_THAT(listModelIds(), ::testing::UnorderedElementsAre("dummy"));
}
