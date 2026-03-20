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
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../http_rest_api_handler.hpp"
#include "../llm/apis/openai_completions.hpp"
#include "../module_names.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "environment.hpp"
#include "test_http_utils.hpp"
#include "test_utils.hpp"
#include "platform_utils.hpp"

class HttpOpenAIHandlerTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    std::string endpoint = "/v3/chat/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
        EnsureServerStartedWithTimeout(this->server, 5);
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mediapipe_openai_chat_completions_mock.json").c_str());
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);
    }

    void TearDown() {
        handler.reset();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

class HttpOpenAIHandlerAuthorizationTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpoint = "/v3/chat/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUpServer(const char* configPath) {
        // create temp file with api key
        std::string apiKeyFile = getGenericFullPathForSrcTest("test_api_key.txt");
        std::ofstream ofs(apiKeyFile);
        std::string absoluteApiKeyPath = std::filesystem::absolute(apiKeyFile).string();
        ofs << "1234";
        ofs.close();
        randomizeAndEnsureFree(this->port);
        ::SetUpServer(this->t, this->server, this->port, configPath, 10, absoluteApiKeyPath);
        EnsureServerStartedWithTimeout(this->server, 20);
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5, "1234");
        // remove temp file with api key
        std::filesystem::remove(absoluteApiKeyPath);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mediapipe_openai_chat_completions_mock.json").c_str());
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);
    }

    void TearDown() {
        handler.reset();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

TEST_F(HttpOpenAIHandlerAuthorizationTest, CorrectApiKey) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "messages": []
        }
    )";
    const std::string URI = "/v3/chat/completions";
    comp.headers["authorization"] = "Bearer 1234";
    std::cout << "URI" << URI << std::endl;
    std::cout << "BODY" << requestBody << std::endl;
    std::cout << "KEY" << comp.headers["authorization"] << std::endl;
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();
    std::shared_ptr<MockedMultiPartParser> multiPartParser = std::make_shared<MockedMultiPartParser>();
    auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);
    std::string response;
    auto status = handler->processV3("/v3/completions", comp, response, requestBody, streamPtr, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(HttpOpenAIHandlerAuthorizationTest, CorrectApiKeyMissingModel) {
    std::string requestBody = R"(
        {
            "model": "gpt-missing",
            "messages": []
        }
    )";
    const std::string URI = "/v3/chat/completions";
    comp.headers["authorization"] = "Bearer 1234";
    std::cout << "URI" << URI << std::endl;
    std::cout << "BODY" << requestBody << std::endl;
    std::cout << "KEY" << comp.headers["authorization"] << std::endl;
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();
    std::shared_ptr<MockedMultiPartParser> multiPartParser = std::make_shared<MockedMultiPartParser>();
    auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);
    std::string response;
    auto status = handler->processV3("/v3/completions", comp, response, requestBody, streamPtr, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING) << status.string();
}

TEST_F(HttpOpenAIHandlerAuthorizationTest, IncorrectApiKey) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "messages": []
        }
    )";
    const std::string URI = "/v3/chat/completions";
    comp.headers["authorization"] = "Bearer ABCD";
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();
    std::shared_ptr<MockedMultiPartParser> multiPartParser = std::make_shared<MockedMultiPartParser>();
    auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);
    std::string response;
    auto status = handler->processV3("/v3/completions", comp, response, requestBody, streamPtr, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::UNAUTHORIZED) << status.string();
}

TEST_F(HttpOpenAIHandlerAuthorizationTest, MissingApiKey) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "messages": []
        }
    )";
    const std::string URI = "/v3/chat/completions";
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();
    std::shared_ptr<MockedMultiPartParser> multiPartParser = std::make_shared<MockedMultiPartParser>();
    auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);
    std::string response;
    auto status = handler->processV3("/v3/completions", comp, response, requestBody, streamPtr, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::UNAUTHORIZED) << status.string();
}

TEST_F(HttpOpenAIHandlerTest, Unary) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";

    const std::string URI = "/v3/something";
    ASSERT_EQ(
        handler->dispatchToProcessor(URI, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(URI: /v3/something
Key: content-type; Value: application/json
Body:

        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    
JSON Parser:
{"model":"gpt","stream":false,"messages":[]}0)";
    ASSERT_EQ(response, expectedResponse);
}

TEST_F(HttpOpenAIHandlerTest, UnaryWithHeaders) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";
    comp.headers["test1"] = "header";
    comp.headers["test2"] = "header";

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/completions/", requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(URI: /v3/completions/
Key: content-type; Value: application/json
Key: test1; Value: header
Key: test2; Value: header
Body:

        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    
JSON Parser:
{"model":"gpt","stream":false,"messages":[]}0)";
    ASSERT_EQ(response, expectedResponse);
}

TEST_F(HttpOpenAIHandlerTest, Stream) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyBegin(::testing::_)).WillOnce(testing::Invoke([](std::function<void()> fn) { fn(); }));
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(9);
    EXPECT_CALL(*writer, IsDisconnected()).Times(9);

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");
}

TEST_F(HttpOpenAIHandlerTest, ResponsesStream) {
    std::string requestBody = R"(
      {
        "model": "gpt",
        "stream": true,
        "input": "What is OpenVINO?"
      }
    )";

    EXPECT_CALL(*writer, PartialReplyBegin(::testing::_)).WillOnce(testing::Invoke([](std::function<void()> fn) { fn(); }));
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(9);
    EXPECT_CALL(*writer, IsDisconnected()).Times(9);

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/responses", requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");
}

TEST_F(HttpOpenAIHandlerTest, BodyNotAJson) {
    std::string requestBody = "not a json";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - Cannot parse JSON body");
}

TEST_F(HttpOpenAIHandlerTest, JsonBodyValidButNotAnObject) {
    std::string requestBody = "[1, 2, 3]";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - JSON body must be an object");
}

TEST_F(HttpOpenAIHandlerTest, GraphWithANameDoesNotExist) {
    std::string requestBody = R"(
        {
            "model": "not_exist",
            "stream": false,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING);
}

class HttpOpenAIHandlerParsingTest : public ::testing::Test {
protected:
    rapidjson::Document doc;
    std::shared_ptr<ov::genai::Tokenizer> tokenizer = std::make_shared<ov::genai::Tokenizer>(getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m"));

    void assertRequestWithTools(std::string providedTools, std::string toolsChoice, absl::StatusCode status = absl::StatusCode::kOk) {
        assertRequestWithTools(providedTools, toolsChoice, "", status);
    }

    void assertRequestWithTools(std::string providedTools, std::string toolsChoice, std::string expectedJson, absl::StatusCode status = absl::StatusCode::kOk) {
        std::string json = R"({
    "messages": [
      {"role": "user", "content": "What is the weather like in Paris today?"},
      {"role": "assistant", "reasoning_content": null, "content": "", "tool_calls": [{"id": "chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec", "type": "function", "function": {"name": "get_weather2", "arguments": "{\"location\": \"Paris, France\"}"}}]},
      {"role": "tool", "tool_call_id": "chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec", "content": "15 degrees Celsius"}],
    "model": "llama",
    "tools": [)";
        json += providedTools;
        json += R"(],
)";
        if (toolsChoice != "") {
            json += R"("tool_choice": )";
            json += toolsChoice;
        }
        json += R"(})";

        doc.Parse(json.c_str());
        ASSERT_FALSE(doc.HasParseError()) << json;
        uint32_t maxTokensLimit = 100;
        uint32_t bestOfLimit = 0;
        std::optional<uint32_t> maxModelLength;
        std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
        ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength).code(), status) << json;
        json = apiHandler->getProcessedJson();
        EXPECT_EQ(json, expectedJson);
    }
};

class HttpOpenAIHandlerCommonParsingValidationTest : public HttpOpenAIHandlerParsingTest,
                                                     public ::testing::WithParamInterface<ovms::Endpoint> {
protected:
    ovms::Endpoint endpoint() const {
        return GetParam();
    }

    std::string createRequestWithRawStreamValue(const std::string& streamRawValue) const {
        if (endpoint() == ovms::Endpoint::COMPLETIONS) {
            return std::string("{\"model\":\"llama\",\"stream\":") + streamRawValue + ",\"prompt\":\"valid prompt\"}";
        }
        if (endpoint() == ovms::Endpoint::RESPONSES) {
            return std::string("{\"model\":\"llama\",\"stream\":") + streamRawValue + ",\"input\":\"valid prompt\"}";
        }
        return std::string("{\"model\":\"llama\",\"stream\":") + streamRawValue + ",\"messages\":[{\"role\":\"user\",\"content\":\"valid prompt\"}]}";
    }

    std::string createRequestWithoutModel() const {
        if (endpoint() == ovms::Endpoint::COMPLETIONS) {
            return "{\"prompt\":\"valid prompt\"}";
        }
        if (endpoint() == ovms::Endpoint::RESPONSES) {
            return "{\"input\":\"valid prompt\"}";
        }
        return "{\"messages\":[{\"role\":\"user\",\"content\":\"valid prompt\"}]}";
    }

    std::string createRequestWithNonStringModel() const {
        if (endpoint() == ovms::Endpoint::COMPLETIONS) {
            return "{\"model\":2,\"prompt\":\"valid prompt\"}";
        }
        if (endpoint() == ovms::Endpoint::RESPONSES) {
            return "{\"model\":2,\"input\":\"valid prompt\"}";
        }
        return "{\"model\":2,\"messages\":[{\"role\":\"user\",\"content\":\"valid prompt\"}]}";
    }
};

TEST_P(HttpOpenAIHandlerCommonParsingValidationTest, StreamFieldNotABooleanFails) {
    std::string json = createRequestWithRawStreamValue("2");
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
        std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, endpoint(), std::chrono::system_clock::now(), *tokenizer);

    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("Stream is not bool"));
}

TEST_P(HttpOpenAIHandlerCommonParsingValidationTest, ModelFieldMissingFails) {
    std::string json = createRequestWithoutModel();
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
        std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, endpoint(), std::chrono::system_clock::now(), *tokenizer);

    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("model missing in request"));
}

TEST_P(HttpOpenAIHandlerCommonParsingValidationTest, ModelFieldNotStringFails) {
    std::string json = createRequestWithNonStringModel();
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
        std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, endpoint(), std::chrono::system_clock::now(), *tokenizer);

    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("model is not a string"));
}

INSTANTIATE_TEST_SUITE_P(
    CommonParsingValidation,
    HttpOpenAIHandlerCommonParsingValidationTest,
    ::testing::Values(ovms::Endpoint::CHAT_COMPLETIONS, ovms::Endpoint::COMPLETIONS, ovms::Endpoint::RESPONSES),
    [](const testing::TestParamInfo<ovms::Endpoint>& info) {
        switch (info.param) {
        case ovms::Endpoint::CHAT_COMPLETIONS:
            return "ChatCompletions";
        case ovms::Endpoint::COMPLETIONS:
            return "Completions";
        case ovms::Endpoint::RESPONSES:
            return "Responses";
        default:
            return "Unknown";
        }
    });

class HttpOpenAIHandlerChatAndResponsesParsingTest : public HttpOpenAIHandlerParsingTest,
                                                     public ::testing::WithParamInterface<ovms::Endpoint> {
protected:
    ovms::Endpoint endpoint() const {
        return GetParam();
    }

    std::string createTextRequest(const std::string& text, const std::string& extraJsonFields = "") const {
        if (endpoint() == ovms::Endpoint::RESPONSES) {
            return std::string("{\"model\":\"llama\",\"input\":\"") + text + "\"" + extraJsonFields + "}";
        }
        return std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"") + text + "\"}]" + extraJsonFields + "}";
    }

    std::string createMultimodalRequestWithImageUrl(const std::string& dataUrl) const {
        if (endpoint() == ovms::Endpoint::RESPONSES) {
            return std::string("{\"model\":\"llama\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"what is in this image?\"},{\"type\":\"input_image\",\"image_url\":\"") + dataUrl + "\"}]}] }";
        }
        return std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what is in this image?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"") + dataUrl + "\"}}]}]}";
    }

    std::string createToolRequest(const std::string& toolChoiceJson) const {
        std::string base = createTextRequest("What is the weather like in Boston today?", ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_current_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}]");
        if (toolChoiceJson.empty()) {
            return base;
        }
        base.pop_back();  // remove trailing '}'
        base += ",\"tool_choice\":" + toolChoiceJson + "}";
        return base;
    }

    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> parseCurrentRequest(const std::string& json) {
        doc.Parse(json.c_str());
        EXPECT_FALSE(doc.HasParseError()) << json;
        std::optional<uint32_t> maxTokensLimit;
        uint32_t bestOfLimit = 0;
        std::optional<uint32_t> maxModelLength;
        std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
            std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, endpoint(), std::chrono::system_clock::now(), *tokenizer);
        EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus()) << json;
        return apiHandler;
    }
};

TEST_P(HttpOpenAIHandlerChatAndResponsesParsingTest, ParsingTextInputCreatesUserChatMessage) {
    std::string json = createTextRequest("What is OpenVINO?");
    auto apiHandler = parseCurrentRequest(json);

    auto& chatHistory = apiHandler->getChatHistory();
    ASSERT_EQ(chatHistory.size(), 1);
    ASSERT_TRUE(chatHistory[0].contains("role"));
    ASSERT_TRUE(chatHistory[0].contains("content"));
    EXPECT_EQ(chatHistory[0]["role"], "user");
    EXPECT_EQ(chatHistory[0]["content"], "What is OpenVINO?");
    EXPECT_TRUE(apiHandler->getProcessedJson().empty());
}

TEST_P(HttpOpenAIHandlerChatAndResponsesParsingTest, ParsingTokenLimitSetsMaxTokens) {
    std::string tokenField = endpoint() == ovms::Endpoint::RESPONSES ? "max_output_tokens" : "max_completion_tokens";
    std::string json = createTextRequest("valid prompt", ",\"" + tokenField + "\":7");
    auto apiHandler = parseCurrentRequest(json);

    EXPECT_TRUE(apiHandler->getMaxTokens().has_value());
    EXPECT_EQ(apiHandler->getMaxTokens().value(), 7);
}

TEST_P(HttpOpenAIHandlerChatAndResponsesParsingTest, ParsingFunctionToolsWithAutoChoiceSucceeds) {
    std::string json = createToolRequest("\"auto\"");
    auto apiHandler = parseCurrentRequest(json);

    EXPECT_TRUE(apiHandler->areToolsAvailable());
    EXPECT_EQ(apiHandler->getToolChoice(), "auto");
}

TEST_P(HttpOpenAIHandlerChatAndResponsesParsingTest, ParsingToolChoiceFunctionObjectSucceeds) {
    std::string json = createToolRequest("{\"type\":\"function\",\"function\":{\"name\":\"get_current_weather\"}}");
    auto apiHandler = parseCurrentRequest(json);

    EXPECT_TRUE(apiHandler->areToolsAvailable());
    EXPECT_EQ(apiHandler->getToolChoice(), "get_current_weather");
}

TEST_P(HttpOpenAIHandlerChatAndResponsesParsingTest, ParsingToolChoiceNoneRemovesTools) {
    std::string json = createToolRequest("\"none\"");
    auto apiHandler = parseCurrentRequest(json);

    EXPECT_FALSE(apiHandler->areToolsAvailable());
    EXPECT_EQ(apiHandler->getToolChoice(), "none");
}

TEST_P(HttpOpenAIHandlerChatAndResponsesParsingTest, ParsingMultimodalInputImageSucceeds) {
    const std::string base64Image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg==";
    std::string json = createMultimodalRequestWithImageUrl(base64Image);
    auto apiHandler = parseCurrentRequest(json);

    EXPECT_EQ(apiHandler->getImageHistory().size(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    ChatAndResponses,
    HttpOpenAIHandlerChatAndResponsesParsingTest,
    ::testing::Values(ovms::Endpoint::CHAT_COMPLETIONS, ovms::Endpoint::RESPONSES),
    [](const testing::TestParamInfo<ovms::Endpoint>& info) {
        switch (info.param) {
        case ovms::Endpoint::CHAT_COMPLETIONS:
            return "ChatCompletions";
        case ovms::Endpoint::RESPONSES:
            return "Responses";
        default:
            return "Unknown";
        }
    });

static std::vector<int64_t> createHermes3ToolCallTokens(ov::genai::Tokenizer& tokenizer) {
    std::string toolCall = R"(<tool_call>{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}</tool_call>)";
    auto generatedTensor = tokenizer.encode(toolCall, ov::genai::add_special_tokens(true)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    return generatedTokens;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeStreamingChunkReturnsIntermediateNullAndFinallyToolCallsFinishReason) {
    std::string json = R"({
    "model": "llama",
    "stream": true,
    "messages": [{"role": "user", "content": "What is weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_humidity",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }]
    })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer, "hermes3");
    uint32_t maxTokensLimit = 100;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    std::vector<std::pair<std::string, ov::genai::GenerationFinishReason>> stream = {
        {"<tool_call>", ov::genai::GenerationFinishReason::NONE},
        {"{\"name\":", ov::genai::GenerationFinishReason::NONE},
        {" \"get", ov::genai::GenerationFinishReason::NONE},
        {"_humidity\",", ov::genai::GenerationFinishReason::NONE},
        {" \"arguments\":", ov::genai::GenerationFinishReason::NONE},
        {" {\"location\":", ov::genai::GenerationFinishReason::NONE},
        {" \"Paris\"}}", ov::genai::GenerationFinishReason::NONE},
        {"</tool_call>", ov::genai::GenerationFinishReason::STOP},
    };

    std::vector<std::string> serializedChunks;
    for (const auto& [chunk, finishReason] : stream) {
        std::string serialized = apiHandler->serializeStreamingChunk(chunk, finishReason);
        if (!serialized.empty()) {
            serializedChunks.push_back(serialized);
        }
    }
    ASSERT_FALSE(serializedChunks.empty());
    const std::string& lastChunk = serializedChunks.back();
    ASSERT_NE(lastChunk.find("\"tool_calls\""), std::string::npos) << lastChunk;
    ASSERT_NE(lastChunk.find("\"finish_reason\":\"tool_calls\""), std::string::npos) << lastChunk;
    // Verify that intermediate chunks with NONE finish_reason are serialized correctly
    ASSERT_GE(serializedChunks.size(), 2u);
    for (size_t i = 0; i + 1 < serializedChunks.size(); ++i) {
        const std::string& chunkStr = serializedChunks[i];
        ASSERT_NE(chunkStr.find("\"finish_reason\":null"), std::string::npos) << chunkStr;
    }
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeStreamingChunkReturnsToolCallsFinishReasonWhenEmptyChunkFollowsToolCall) {
    std::string json = R"({
    "model": "llama",
    "stream": true,
    "messages": [{"role": "user", "content": "What is weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_humidity",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }]
    })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer, "hermes3");
    uint32_t maxTokensLimit = 100;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    // Simulate scenario where tool call completes with NONE finish reason, then an empty chunk arrives with STOP
    std::vector<std::pair<std::string, ov::genai::GenerationFinishReason>> stream = {
        {"<tool_call>", ov::genai::GenerationFinishReason::NONE},
        {"{\"name\":", ov::genai::GenerationFinishReason::NONE},
        {" \"get", ov::genai::GenerationFinishReason::NONE},
        {"_humidity\",", ov::genai::GenerationFinishReason::NONE},
        {" \"arguments\":", ov::genai::GenerationFinishReason::NONE},
        {" {\"location\":", ov::genai::GenerationFinishReason::NONE},
        {" \"Paris\"}}", ov::genai::GenerationFinishReason::NONE},
        {"</tool_call>", ov::genai::GenerationFinishReason::NONE},
        {"", ov::genai::GenerationFinishReason::STOP},
    };

    std::vector<std::string> serializedChunks;
    for (const auto& [chunk, finishReason] : stream) {
        std::string serialized = apiHandler->serializeStreamingChunk(chunk, finishReason);
        if (!serialized.empty()) {
            serializedChunks.push_back(serialized);
        }
    }
    ASSERT_FALSE(serializedChunks.empty());
    const std::string& lastChunk = serializedChunks.back();
    ASSERT_NE(lastChunk.find("\"finish_reason\":\"tool_calls\""), std::string::npos) << lastChunk;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeUnaryResponseGenerationOutputReturnsToolCallsFinishReason) {
    std::string json = R"({
    "model": "llama",
    "stream": false,
    "messages": [{"role": "user", "content": "What is weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "example_tool",
        "parameters": {"type": "object"}
      }
    }]
    })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer, "hermes3");
    uint32_t maxTokensLimit = 100;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::GenerationOutput generationOutput;
    generationOutput.generated_ids = createHermes3ToolCallTokens(*tokenizer);
    generationOutput.finish_reason = ov::genai::GenerationFinishReason::STOP;  // Change it once GenAI introduces tool_calls finish reason
    std::string serialized = apiHandler->serializeUnaryResponse(std::vector<ov::genai::GenerationOutput>{generationOutput});

    ASSERT_NE(serialized.find("\"finish_reason\":\"tool_calls\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"tool_calls\":[{"), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeUnaryResponseEncodedResultsReturnsToolCallsFinishReason) {
    std::string json = R"({
    "model": "llama",
    "stream": false,
    "messages": [{"role": "user", "content": "What is weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "example_tool",
        "parameters": {"type": "object"}
      }
    }]
    })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer, "hermes3");
    uint32_t maxTokensLimit = 100;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::EncodedResults results;
    results.tokens = {createHermes3ToolCallTokens(*tokenizer)};
    std::string serialized = apiHandler->serializeUnaryResponse(results);

    ASSERT_NE(serialized.find("\"finish_reason\":\"tool_calls\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"tool_calls\":[{"), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeUnaryResponseVLMSupportsToolCallsFinishReason) {
    std::string json = R"({
    "model": "llama",
    "stream": false,
    "messages": [{"role": "user", "content": "What is weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "example_tool",
        "parameters": {"type": "object"}
      }
    }]
    })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer, "hermes3");
    uint32_t maxTokensLimit = 100;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::VLMDecodedResults results;
    std::string toolCall = R"(<tool_call>{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}</tool_call>)";
    results.texts = {toolCall};
    std::string serialized = apiHandler->serializeUnaryResponse(results);

    ASSERT_NE(serialized.find("\"finish_reason\":\"tool_calls\""), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeUnaryResponseForResponsesContainsOutputText) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "max_output_tokens": 5
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::EncodedResults results;
    ov::Tensor outputIds = tokenizer->encode("OVMS", ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(outputIds.get_shape().size(), 2);
    ASSERT_EQ(outputIds.get_shape()[0], 1);
    ASSERT_EQ(outputIds.get_element_type(), ov::element::i64);
    int64_t* outputIdsData = reinterpret_cast<int64_t*>(outputIds.data());
    results.tokens = {std::vector<int64_t>(outputIdsData, outputIdsData + outputIds.get_shape()[1])};

    std::string serialized = apiHandler->serializeUnaryResponse(results);
    ASSERT_NE(serialized.find("\"object\":\"response\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"output\":"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"type\":\"output_text\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"text\":"), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeStreamingChunkForResponsesContainsRequiredEvents) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "stream": true
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    // Phase 1: Init events emitted via dedicated method (called right after scheduleExecution in calculator)
    std::string initChunk = apiHandler->serializeResponsesStreamingInitEvents();
    ASSERT_NE(initChunk.find("\"type\":\"response.created\""), std::string::npos) << initChunk;
    ASSERT_NE(initChunk.find("\"type\":\"response.in_progress\""), std::string::npos) << initChunk;
    ASSERT_NE(initChunk.find("\"type\":\"response.output_item.added\""), std::string::npos) << initChunk;
    ASSERT_NE(initChunk.find("\"type\":\"response.content_part.added\""), std::string::npos) << initChunk;
    // No delta event when text is empty
    ASSERT_EQ(initChunk.find("\"type\":\"response.output_text.delta\""), std::string::npos) << initChunk;

    // Verify correct event ordering: created < in_progress < output_item.added < content_part.added
    auto createdPos = initChunk.find("\"type\":\"response.created\"");
    auto inProgressPos = initChunk.find("\"type\":\"response.in_progress\"");
    auto outputItemAddedPos = initChunk.find("\"type\":\"response.output_item.added\"");
    auto contentPartAddedPos = initChunk.find("\"type\":\"response.content_part.added\"");
    ASSERT_LT(createdPos, inProgressPos) << "response.created must come before response.in_progress";
    ASSERT_LT(inProgressPos, outputItemAddedPos) << "response.in_progress must come before response.output_item.added";
    ASSERT_LT(outputItemAddedPos, contentPartAddedPos) << "response.output_item.added must come before response.content_part.added";

    // Phase 2: Second call should only contain delta, no repeated init events
    std::string secondChunk = apiHandler->serializeStreamingChunk("", ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(secondChunk.empty()) << "Empty text after init should produce no output: " << secondChunk;

    // Phase 3: Text delta
    std::string deltaChunk = apiHandler->serializeStreamingChunk("Hello", ov::genai::GenerationFinishReason::NONE);
    ASSERT_NE(deltaChunk.find("\"type\":\"response.output_text.delta\""), std::string::npos) << deltaChunk;
    ASSERT_NE(deltaChunk.find("\"delta\":\"Hello\""), std::string::npos) << deltaChunk;
    ASSERT_EQ(deltaChunk.find("\"type\":\"response.created\""), std::string::npos) << "No repeated init events: " << deltaChunk;

    // Phase 4: Final chunk with finish reason
    std::string finalChunk = apiHandler->serializeStreamingChunk(" world", ov::genai::GenerationFinishReason::STOP);
    ASSERT_NE(finalChunk.find("\"type\":\"response.output_text.delta\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"type\":\"response.output_text.done\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"type\":\"response.content_part.done\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"type\":\"response.output_item.done\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"type\":\"response.completed\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"text\":\"Hello world\""), std::string::npos) << finalChunk;

    // Verify correct event ordering in final chunk: delta < output_text.done < content_part.done < output_item.done < completed
    auto deltaPos = finalChunk.find("\"type\":\"response.output_text.delta\"");
    auto textDonePos = finalChunk.find("\"type\":\"response.output_text.done\"");
    auto partDonePos = finalChunk.find("\"type\":\"response.content_part.done\"");
    auto itemDonePos = finalChunk.find("\"type\":\"response.output_item.done\"");
    auto completedPos = finalChunk.find("\"type\":\"response.completed\"");
    ASSERT_LT(deltaPos, textDonePos) << "delta must come before output_text.done";
    ASSERT_LT(textDonePos, partDonePos) << "output_text.done must come before content_part.done";
    ASSERT_LT(partDonePos, itemDonePos) << "content_part.done must come before output_item.done";
    ASSERT_LT(itemDonePos, completedPos) << "output_item.done must come before response.completed";
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeStreamingUsageChunkForResponsesIsEmpty) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "stream": true
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ASSERT_EQ(apiHandler->serializeStreamingUsageChunk(), "");
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeStreamingChunkForResponsesEmitsIncompleteOnLengthFinish) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "stream": true
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    // Init events
    apiHandler->serializeResponsesStreamingInitEvents();
    // Delta
    apiHandler->serializeStreamingChunk("Hello", ov::genai::GenerationFinishReason::NONE);

    // Final chunk with LENGTH finish reason
    std::string finalChunk = apiHandler->serializeStreamingChunk("", ov::genai::GenerationFinishReason::LENGTH);

    // Should emit response.incomplete instead of response.completed
    ASSERT_NE(finalChunk.find("\"type\":\"response.incomplete\""), std::string::npos) << finalChunk;
    ASSERT_EQ(finalChunk.find("\"type\":\"response.completed\""), std::string::npos) << "Should not contain response.completed: " << finalChunk;

    // Should contain incomplete_details with max_tokens reason
    ASSERT_NE(finalChunk.find("\"incomplete_details\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"reason\":\"max_tokens\""), std::string::npos) << finalChunk;

    // Response status should be "incomplete"
    ASSERT_NE(finalChunk.find("\"status\":\"incomplete\""), std::string::npos) << finalChunk;

    // Should NOT contain completed_at
    // Find the response.incomplete event section and check it doesn't have completed_at
    auto incompletePos = finalChunk.find("\"type\":\"response.incomplete\"");
    auto responseSection = finalChunk.substr(incompletePos);
    ASSERT_EQ(responseSection.find("\"completed_at\""), std::string::npos) << "Incomplete response should not have completed_at: " << responseSection;

    // output_item.done should have status "incomplete"
    auto itemDonePos = finalChunk.find("\"type\":\"response.output_item.done\"");
    ASSERT_NE(itemDonePos, std::string::npos) << finalChunk;
    auto itemSection = finalChunk.substr(itemDonePos);
    ASSERT_NE(itemSection.find("\"status\":\"incomplete\""), std::string::npos) << "output_item.done should have incomplete status: " << itemSection;

    // Still should have the other finalization events
    ASSERT_NE(finalChunk.find("\"type\":\"response.output_text.done\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"type\":\"response.content_part.done\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"type\":\"response.output_item.done\""), std::string::npos) << finalChunk;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeStreamingChunkForResponsesEmitsCompletedOnStopFinish) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "stream": true
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    // Init events
    apiHandler->serializeResponsesStreamingInitEvents();
    // Delta + finish with STOP
    std::string finalChunk = apiHandler->serializeStreamingChunk("Hello", ov::genai::GenerationFinishReason::STOP);

    // Should emit response.completed, NOT response.incomplete
    ASSERT_NE(finalChunk.find("\"type\":\"response.completed\""), std::string::npos) << finalChunk;
    ASSERT_EQ(finalChunk.find("\"type\":\"response.incomplete\""), std::string::npos) << "Should not contain response.incomplete: " << finalChunk;
    ASSERT_EQ(finalChunk.find("\"incomplete_details\""), std::string::npos) << "Should not contain incomplete_details: " << finalChunk;

    // Response status should be "completed"
    ASSERT_NE(finalChunk.find("\"status\":\"completed\""), std::string::npos) << finalChunk;

    // Should contain new spec-aligned fields
    ASSERT_NE(finalChunk.find("\"error\":null"), std::string::npos) << "Should contain error:null: " << finalChunk;
    ASSERT_NE(finalChunk.find("\"previous_response_id\":null"), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"reasoning\":null"), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"store\":true"), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"truncation\":\"disabled\""), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"user\":null"), std::string::npos) << finalChunk;
    ASSERT_NE(finalChunk.find("\"metadata\":{}"), std::string::npos) << finalChunk;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeResponsesFailedEventContainsCorrectStructure) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "stream": true
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    std::string failedEvent = apiHandler->serializeResponsesFailedEvent("Something went wrong");

    // Should contain response.failed event type
    ASSERT_NE(failedEvent.find("\"type\":\"response.failed\""), std::string::npos) << failedEvent;
    // Should NOT contain response.completed or response.incomplete
    ASSERT_EQ(failedEvent.find("\"type\":\"response.completed\""), std::string::npos) << failedEvent;
    ASSERT_EQ(failedEvent.find("\"type\":\"response.incomplete\""), std::string::npos) << failedEvent;

    // Should contain error object with code and message
    ASSERT_NE(failedEvent.find("\"error\":{"), std::string::npos) << "Should contain error object: " << failedEvent;
    ASSERT_NE(failedEvent.find("\"code\":\"server_error\""), std::string::npos) << failedEvent;
    ASSERT_NE(failedEvent.find("\"message\":\"Something went wrong\""), std::string::npos) << failedEvent;

    // Response status should be "failed"
    ASSERT_NE(failedEvent.find("\"status\":\"failed\""), std::string::npos) << failedEvent;

    // Should include init events since they were not emitted before
    ASSERT_NE(failedEvent.find("\"type\":\"response.created\""), std::string::npos) << failedEvent;

    // Should contain sequence_number
    ASSERT_NE(failedEvent.find("\"sequence_number\""), std::string::npos) << failedEvent;

    // Should NOT contain completed_at
    auto failedPos = failedEvent.find("\"type\":\"response.failed\"");
    auto responseSection = failedEvent.substr(failedPos);
    ASSERT_EQ(responseSection.find("\"completed_at\""), std::string::npos) << "Failed response should not have completed_at: " << responseSection;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeResponsesFailedEventWithCustomErrorCode) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "stream": true
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    std::string failedEvent = apiHandler->serializeResponsesFailedEvent("Invalid prompt content", "invalid_prompt");

    ASSERT_NE(failedEvent.find("\"code\":\"invalid_prompt\""), std::string::npos) << failedEvent;
    ASSERT_NE(failedEvent.find("\"message\":\"Invalid prompt content\""), std::string::npos) << failedEvent;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeResponsesFailedEventAfterPartialStreaming) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "stream": true
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    // Emit init events and some deltas first
    apiHandler->serializeResponsesStreamingInitEvents();
    apiHandler->serializeStreamingChunk("Hello", ov::genai::GenerationFinishReason::NONE);

    // Then fail
    std::string failedEvent = apiHandler->serializeResponsesFailedEvent("Generation aborted");

    // Should contain response.failed but NOT init events (already sent)
    ASSERT_NE(failedEvent.find("\"type\":\"response.failed\""), std::string::npos) << failedEvent;
    ASSERT_EQ(failedEvent.find("\"type\":\"response.created\""), std::string::npos) << "Should not re-emit init events: " << failedEvent;

    // Error should be present
    ASSERT_NE(failedEvent.find("\"error\":{"), std::string::npos) << failedEvent;
    ASSERT_NE(failedEvent.find("\"code\":\"server_error\""), std::string::npos) << failedEvent;
    ASSERT_NE(failedEvent.find("\"message\":\"Generation aborted\""), std::string::npos) << failedEvent;

    // Should NOT contain usage (failed responses don't include usage)
    ASSERT_EQ(failedEvent.find("\"usage\""), std::string::npos) << "Failed response should not include usage: " << failedEvent;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeUnaryResponseForResponsesIncompleteOnLength) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "max_output_tokens": 5
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::GenerationOutput genOutput;
    ov::Tensor outputIds = tokenizer->encode("OVMS", ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(outputIds.get_shape().size(), 2);
    ASSERT_EQ(outputIds.get_shape()[0], 1);
    ASSERT_EQ(outputIds.get_element_type(), ov::element::i64);
    int64_t* outputIdsData = reinterpret_cast<int64_t*>(outputIds.data());
    genOutput.generated_ids = std::vector<int64_t>(outputIdsData, outputIdsData + outputIds.get_shape()[1]);
    genOutput.finish_reason = ov::genai::GenerationFinishReason::LENGTH;

    std::vector<ov::genai::GenerationOutput> generationOutputs = {genOutput};
    std::string serialized = apiHandler->serializeUnaryResponse(generationOutputs);

    // Should have status "incomplete"
    ASSERT_NE(serialized.find("\"status\":\"incomplete\""), std::string::npos) << serialized;
    // Should have incomplete_details with reason
    ASSERT_NE(serialized.find("\"incomplete_details\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"reason\":\"max_tokens\""), std::string::npos) << serialized;
    // Should NOT have completed_at
    ASSERT_EQ(serialized.find("\"completed_at\""), std::string::npos) << serialized;
    // Should NOT have status "completed"
    ASSERT_EQ(serialized.find("\"status\":\"completed\""), std::string::npos) << serialized;

    // Should contain new spec-aligned fields
    ASSERT_NE(serialized.find("\"error\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"previous_response_id\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"reasoning\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"store\":true"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"truncation\":\"disabled\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"user\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"metadata\":{}"), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, serializeUnaryResponseForResponsesCompletedOnStop) {
    std::string json = R"({
    "model": "llama",
    "input": "What is OpenVINO?",
    "max_output_tokens": 5
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::GenerationOutput genOutput;
    ov::Tensor outputIds = tokenizer->encode("OVMS", ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(outputIds.get_shape().size(), 2);
    ASSERT_EQ(outputIds.get_shape()[0], 1);
    ASSERT_EQ(outputIds.get_element_type(), ov::element::i64);
    int64_t* outputIdsData = reinterpret_cast<int64_t*>(outputIds.data());
    genOutput.generated_ids = std::vector<int64_t>(outputIdsData, outputIdsData + outputIds.get_shape()[1]);
    genOutput.finish_reason = ov::genai::GenerationFinishReason::STOP;

    std::vector<ov::genai::GenerationOutput> generationOutputs = {genOutput};
    std::string serialized = apiHandler->serializeUnaryResponse(generationOutputs);

    // Should have status "completed"
    ASSERT_NE(serialized.find("\"status\":\"completed\""), std::string::npos) << serialized;
    // Should have completed_at
    ASSERT_NE(serialized.find("\"completed_at\""), std::string::npos) << serialized;
    // Should NOT have incomplete_details
    ASSERT_EQ(serialized.find("\"incomplete_details\""), std::string::npos) << serialized;

    // Should contain new spec-aligned fields
    ASSERT_NE(serialized.find("\"error\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"previous_response_id\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"reasoning\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"store\":true"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"truncation\":\"disabled\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"user\":null"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"metadata\":{}"), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesSucceedsBase64) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url":  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="
            }
          }
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 3);
    std::vector<uint8_t> expectedBytes = {110, 181, 160};
    for (size_t i = 0; i < image.get_size(); i++) {
        EXPECT_EQ(expectedBytes[i], ((uint8_t*)image.data())[i]);
    }
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesSucceedsUrlHttp) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  "http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"raw.githubusercontent.com"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 225792);
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesSucceedsUrlHttpMultipleAllowedDomains) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  "http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"raw.githubusercontent.com", "githubusercontent.com", "google.com"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 225792);
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesSucceedsUrlHttps) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    std::string json = R"({
"model": "llama",
"messages": [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this image?"
      },
      {
        "type": "image_url",
        "image_url": {
          "url":  "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
        }
      }
    ]
  }
]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"raw.githubusercontent.com"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 225792);
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesSucceedsUrlHttpsAllowedDomainAll) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    std::string json = R"({
"model": "llama",
"messages": [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this image?"
      },
      {
        "type": "image_url",
        "image_url": {
          "url":  "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
        }
      }
    ]
  }
]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"all"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 225792);
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingImageJpegWithNoTextSucceeds) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url":  "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGIy+/oREAAA//8DiQIftNKCRwAAAABJRU5ErkJggg=="
            }
          }
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 3);
    std::vector<uint8_t> expectedBytes = {54, 245, 241};
    for (size_t i = 0; i < image.get_size(); i++) {
        EXPECT_EQ(expectedBytes[i], ((uint8_t*)image.data())[i]);
    }
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageStringWithNoPrefixFails) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url":  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="
            }
          }
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseMessages(), absl::InvalidArgumentError("Loading images from local filesystem is disabled."));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesFailsUrlHttpNotAllowedDomain) {
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  "http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"wikipedia.com"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesFailsUrlMatchAllowedDomainPartially1) {
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  "http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"githubusercontent.com"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesFailsUrlMatchAllowedDomainPartially2) {
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  "http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"host.raw.githubusercontent.com"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesFailsRegexNotSupported) {
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  "http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::vector<std::string> allowedDomains = {"*githubusercontent.com"};
    ASSERT_EQ(apiHandler->parseMessages(std::nullopt, allowedDomains), absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageLocalFilesystem) {
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  ")" +
                       getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb.jpg") + R"("
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(getGenericFullPathForSrcTest("/ovms/src/test")), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 3);
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageLocalFilesystemWithinAllowedPath) {
    std::string json = R"({
"model": "llama",
"messages": [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this image?"
      },
      {
        "type": "image_url",
        "image_url": {
          "url":  ")" + getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb.jpg") +
                       R"("
        }
      }
    ]
  }
]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils")), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 1);
    auto [index, image] = imageHistory[0];
    EXPECT_EQ(index, 0);
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 3);
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageLocalFilesystemNotWithinAllowedPath) {
    std::string json = R"({
"model": "llama",
"messages": [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this image?"
      },
      {
        "type": "image_url",
        "image_url": {
          "url":  "/ovms/src/test/binaryutils/rgb.jpg"
        }
      }
    ]
  }
]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages("src/test"), absl::InvalidArgumentError("Given filepath is not subpath of allowed_local_media_path"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageLocalFilesystemInvalidPath) {
    std::string json = R"({
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  "/ovms/not_exisiting.jpeg"
          }
        }
      ]
    }
  ]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseMessages("/ovms/"), absl::InvalidArgumentError("Image file /ovms/not_exisiting.jpeg parsing failed: can't fopen"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageLocalFilesystemInvalidEscaped) {
    std::string json = R"({
"model": "llama",
"messages": [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What is in this image?"
      },
      {
        "type": "image_url",
        "image_url": {
          "url":  ")" + getGenericFullPathForSrcTest("/ovms/src/test/../test/binaryutils/rgb.jpg") +
                       R"("
        }
      }
    ]
  }
]
})";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    std::string expectedMessage = "Path " + getGenericFullPathForSrcTest("/ovms/src/test/../test/binaryutils/rgb.jpg") + " escape with .. is forbidden.";
    EXPECT_EQ(apiHandler->parseMessages("/ovms/"), absl::InvalidArgumentError(expectedMessage.c_str()));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMultipleMessagesSucceeds) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url":  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="
            }
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "No idea my friend."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What about this one?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url":  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="
            }
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "Same thing. I'm not very good with images."
          }
        ]
      },
      {
        "role": "user",
        "content": "You were not trained with images, were you?"
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());
    const ovms::ImageHistory& imageHistory = apiHandler->getImageHistory();
    ASSERT_EQ(imageHistory.size(), 2);
    std::vector<uint8_t> expectedBytes = {110, 181, 160};
    std::vector<size_t> expectedImageIndexes = {0, 2};
    size_t i = 0;
    for (auto [index, image] : imageHistory) {
        EXPECT_EQ(index, expectedImageIndexes[i++]);
        EXPECT_EQ(image.get_element_type(), ov::element::u8);
        EXPECT_EQ(image.get_size(), 3);
        for (size_t i = 0; i < image.get_size(); i++) {
            EXPECT_EQ(expectedBytes[i], ((uint8_t*)image.data())[i]);
        }
    }
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"},"
                                "{\"role\":\"assistant\",\"content\":\"No idea my friend.\"},"
                                "{\"role\":\"user\",\"content\":\"What about this one?\"},"
                                "{\"role\":\"assistant\",\"content\":\"Same thing. I'm not very good with images.\"},"
                                "{\"role\":\"user\",\"content\":\"You were not trained with images, were you?\"}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesWithInvalidContentTypeFails) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "INVALID"
          }
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseMessages(), absl::InvalidArgumentError("Unsupported content type"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesEmptyImageUrlFails) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url":  ""
            }
          }
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseMessages(), absl::InvalidArgumentError("Loading images from local filesystem is disabled."));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageUrlNotBase64Fails) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url":  "base64,NOTBASE64"
            }
          }
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseMessages(), absl::InvalidArgumentError("Invalid base64 string in request"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesEmptyContentArrayFails) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": []
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseMessages(), absl::InvalidArgumentError("Invalid message structure - content array is empty"));
}

TEST_F(HttpOpenAIHandlerParsingTest, maxTokensValueDefaultToMaxTokensLimit) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "valid prompt"
          }
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    uint32_t maxTokensLimit = 10;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_TRUE(apiHandler->getMaxTokens().has_value());
    EXPECT_EQ(apiHandler->getMaxTokens().value(), maxTokensLimit);
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingRequestWithNullParametersChat) {
    std::vector<std::string> chatParamsThatAcceptNull = {"stream", "stream_options", "ignore_eos", "frequency_penalty", "presence_penalty", "repetition_penalty",
        "length_penalty", "temperature", "top_p", "top_k", "seed", "stop", "include_stop_str_in_output", "best_of", "n", "num_assistant_tokens", "assistant_confidence_threshold",
        "logprobs", "max_completion_tokens", "tools", "tool_choice"};
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    for (auto param : chatParamsThatAcceptNull) {
        std::string json = R"({
      "model": "llama",
      ")" + param + R"(": null,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "valid prompt"
            }
          ]
        }
      ]
    })";
        doc.Parse(json.c_str());
        ASSERT_FALSE(doc.HasParseError());
        std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
        EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    }
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingRequestWithNullParametersCompletions) {
    std::vector<std::string> chatParamsThatAcceptNull = {"stream", "stream_options", "ignore_eos", "frequency_penalty", "presence_penalty", "repetition_penalty",
        "length_penalty", "temperature", "top_p", "top_k", "seed", "stop", "include_stop_str_in_output", "best_of", "n", "num_assistant_tokens", "assistant_confidence_threshold",
        "logprobs", "echo"};
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    for (auto param : chatParamsThatAcceptNull) {
        std::string json = R"({
      "model": "llama",
      ")" + param + R"(": null,
      "prompt": "valid prompt"
    })";
        doc.Parse(json.c_str());
        ASSERT_FALSE(doc.HasParseError());
        std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
        EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    }
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesMaxOutputTokensSetsLimit) {
    std::string json = R"({
    "model": "llama",
    "input": "valid prompt",
    "max_output_tokens": 42
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_TRUE(apiHandler->getMaxTokens().has_value());
    EXPECT_EQ(apiHandler->getMaxTokens().value(), 42);
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesMaxCompletionTokensIsIgnored) {
    std::string json = R"({
    "model": "llama",
    "input": "valid prompt",
    "max_completion_tokens": 50
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    // max_completion_tokens should be ignored for RESPONSES endpoint, so maxTokens should not be 50
    EXPECT_FALSE(apiHandler->getMaxTokens().has_value() && apiHandler->getMaxTokens().value() == 50);
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesMaxTokensIsIgnored) {
    std::string json = R"({
    "model": "llama",
    "input": "valid prompt",
    "max_tokens": 50
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    // max_tokens should be ignored for RESPONSES endpoint, so maxTokens should not be 50
    EXPECT_FALSE(apiHandler->getMaxTokens().has_value() && apiHandler->getMaxTokens().value() == 50);
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesNStreamingIsRejected) {
    std::string json = R"({
    "model": "llama",
    "input": "valid prompt",
    "stream": true,
    "n": 2
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("n greater than 1 is not supported for Responses API streaming"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesNUnaryIsAccepted) {
    std::string json = R"({
    "model": "llama",
    "input": "valid prompt",
    "best_of": 3,
    "n": 2
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 100;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesFlatFunctionToolsSucceeds) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": "auto",
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location", "unit"]
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_TRUE(apiHandler->areToolsAvailable());
    EXPECT_EQ(apiHandler->getToolChoice(), "auto");
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesToolChoiceFunctionObjectSucceeds) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": {
      "type": "function",
      "name": "get_current_weather"
    },
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      },
      {
        "type": "function",
        "name": "unused_tool",
        "parameters": {
          "type": "object",
          "properties": {
            "arg": {"type": "string"}
          }
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_TRUE(apiHandler->areToolsAvailable());
    EXPECT_EQ(apiHandler->getToolChoice(), "get_current_weather");
}

TEST_F(HttpOpenAIHandlerParsingTest, SerializeResponsesUnaryResponseContainsFunctionTools) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": "auto",
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::EncodedResults results;
    ov::Tensor outputIds = tokenizer->encode("Sunny", ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(outputIds.get_shape().size(), 2);
    ASSERT_EQ(outputIds.get_shape()[0], 1);
    ASSERT_EQ(outputIds.get_element_type(), ov::element::i64);
    int64_t* outputIdsData = reinterpret_cast<int64_t*>(outputIds.data());
    results.tokens = {std::vector<int64_t>(outputIdsData, outputIdsData + outputIds.get_shape()[1])};

    std::string serialized = apiHandler->serializeUnaryResponse(results);
    ASSERT_NE(serialized.find("\"object\":\"response\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"tools\":[{"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"type\":\"function\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"name\":\"get_current_weather\""), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, SerializeResponsesUnaryResponseContainsFunctionToolChoiceObject) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": {
      "type": "function",
      "name": "get_current_weather"
    },
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::EncodedResults results;
    ov::Tensor outputIds = tokenizer->encode("Sunny", ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(outputIds.get_shape().size(), 2);
    ASSERT_EQ(outputIds.get_shape()[0], 1);
    ASSERT_EQ(outputIds.get_element_type(), ov::element::i64);
    int64_t* outputIdsData = reinterpret_cast<int64_t*>(outputIds.data());
    results.tokens = {std::vector<int64_t>(outputIdsData, outputIdsData + outputIds.get_shape()[1])};

    std::string serialized = apiHandler->serializeUnaryResponse(results);
    ASSERT_NE(serialized.find("\"tool_choice\":{"), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"type\":\"function\""), std::string::npos) << serialized;
    ASSERT_NE(serialized.find("\"name\":\"get_current_weather\""), std::string::npos) << serialized;
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesToolChoiceFunctionObjectMissingNameFails) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": {
      "type": "function"
    },
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("tool_choice.name is not a valid string"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesToolChoiceFunctionObjectNameNotStringFails) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": {
      "type": "function",
      "name": 7
    },
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("tool_choice.name is not a valid string"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesInputImageUrlObjectSucceeds) {
    std::string json = R"({
    "model": "llama",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "what is in this image?"},
          {"type": "input_image", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="}}
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
        std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_EQ(apiHandler->getImageHistory().size(), 1);
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesInputImageWithoutImageUrlFails) {
    std::string json = R"({
    "model": "llama",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "what is in this image?"},
          {"type": "input_image"}
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
        std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("input_image requires image_url field"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesInputImageUrlInvalidTypeFails) {
    std::string json = R"({
    "model": "llama",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "what is in this image?"},
          {"type": "input_image", "image_url": 123}
        ]
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
        std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("input_image.image_url must be a string or object"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesUnsupportedToolTypeFails) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": "auto",
    "tools": [
      {
        "type": "web_search_preview"
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("Only function tools are supported"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingResponsesToolChoiceNoneRemovesTools) {
    std::string json = R"({
    "model": "llama",
    "input": "What is the weather like in Boston today?",
    "tool_choice": "none",
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::RESPONSES, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_FALSE(apiHandler->areToolsAvailable());
    EXPECT_EQ(apiHandler->getToolChoice(), "none");
}

// Provide get_weather2 but take none
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided1_ChoiceNone) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"("none")";
    std::string expectedJson = std::string("{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather like in Paris today?\"},{\"role\":\"assistant\",\"reasoning_content\":null,\"content\":\"\",\"tool_calls\":[{\"id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"arguments\":\"{\\\"location\\\": \\\"Paris, France\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"content\":\"15 degrees Celsius\"}],\"model\":\"llama\","
                                           "\"tool_choice\":\"none\"}");

    assertRequestWithTools(providedTools, toolsChoice, expectedJson);
}

TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_ParsesToolsJsonContainerOnDemand) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler =
        std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);

    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto toolsStatus = apiHandler->parseToolsToJsonContainer();
    ASSERT_TRUE(toolsStatus.ok());
    const auto& tools = toolsStatus.value();
    ASSERT_TRUE(tools.has_value());
    EXPECT_TRUE(tools->is_array());
    ASSERT_EQ(tools->size(), 1);
    ASSERT_TRUE((*tools)[0]["function"]["name"].as_string().has_value());
    EXPECT_EQ((*tools)[0]["function"]["name"].as_string().value(), "get_weather");
}

TEST_F(HttpOpenAIHandlerParsingTest, OutputParserInitializationDependsOnParserNames) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": "hello"
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    auto withoutParserNames = std::make_shared<ovms::OpenAIChatCompletionsHandler>(
        doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(withoutParserNames->getOutputParser(), nullptr);

    auto withParserNames = std::make_shared<ovms::OpenAIChatCompletionsHandler>(
        doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer, "llama3", "");
    EXPECT_NE(withParserNames->getOutputParser(), nullptr);
}

TEST_F(HttpOpenAIHandlerParsingTest, SerializeUnaryResponseVLMDecodedResultsWithToolParser) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": "hello"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    uint32_t maxTokensLimit = 64;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;

    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(
        doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer, "hermes3", "");

    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    ov::genai::VLMDecodedResults results;
    results.texts.push_back(
        "I will call a tool.<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}</tool_call>");

    std::string serialized = apiHandler->serializeUnaryResponse(results);

    rapidjson::Document responseDoc;
    responseDoc.Parse(serialized.c_str());
    ASSERT_FALSE(responseDoc.HasParseError());
    ASSERT_TRUE(responseDoc.IsObject());

    ASSERT_TRUE(responseDoc.HasMember("choices"));
    ASSERT_TRUE(responseDoc["choices"].IsArray());
    ASSERT_EQ(responseDoc["choices"].Size(), 1);

    const auto& choice = responseDoc["choices"][0];
    ASSERT_TRUE(choice.IsObject());
    ASSERT_TRUE(choice.HasMember("finish_reason"));
    ASSERT_TRUE(choice["finish_reason"].IsString());
    EXPECT_STREQ(choice["finish_reason"].GetString(), "tool_calls");

    ASSERT_TRUE(choice.HasMember("message"));
    ASSERT_TRUE(choice["message"].IsObject());
    const auto& message = choice["message"];

    ASSERT_TRUE(message.HasMember("content"));
    ASSERT_TRUE(message["content"].IsString());
    EXPECT_STREQ(message["content"].GetString(), "I will call a tool.");

    ASSERT_TRUE(message.HasMember("tool_calls"));
    ASSERT_TRUE(message["tool_calls"].IsArray());
    ASSERT_EQ(message["tool_calls"].Size(), 1);

    const auto& toolCall = message["tool_calls"][0];
    ASSERT_TRUE(toolCall.IsObject());
    ASSERT_TRUE(toolCall.HasMember("id"));
    ASSERT_TRUE(toolCall["id"].IsString());
    EXPECT_GT(std::string(toolCall["id"].GetString()).size(), 0);
    ASSERT_TRUE(toolCall.HasMember("function"));
    ASSERT_TRUE(toolCall["function"].IsObject());
    ASSERT_TRUE(toolCall["function"].HasMember("name"));
    EXPECT_STREQ(toolCall["function"]["name"].GetString(), "get_weather");
    ASSERT_TRUE(toolCall["function"].HasMember("arguments"));
    EXPECT_STREQ(toolCall["function"]["arguments"].GetString(), "{\"location\":\"Paris\"}");

    ASSERT_TRUE(responseDoc.HasMember("object"));
    EXPECT_STREQ(responseDoc["object"].GetString(), "chat.completion");
    ASSERT_TRUE(responseDoc.HasMember("model"));
    EXPECT_STREQ(responseDoc["model"].GetString(), "llama");

    ASSERT_TRUE(responseDoc.HasMember("usage"));
    ASSERT_TRUE(responseDoc["usage"].IsObject());
    ASSERT_TRUE(responseDoc["usage"].HasMember("completion_tokens"));
    // Not checking exact values as results are mocked, just ensuring they are present and of correct type
    EXPECT_TRUE(responseDoc["usage"]["completion_tokens"].IsUint());
}

// Provide get_weather1, get_weather2, get_weather3 but take only first one - get_weather1
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceFirst) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"({"type": "function", "function": {"name": "get_weather1"}})";
    std::string expectedJson = std::string("{\"messages\":["
                                           "{\"role\":\"user\",\"content\":\"What is the weather like in Paris today?\"},"
                                           "{\"role\":\"assistant\",\"reasoning_content\":null,\"content\":\"\",\"tool_calls\":[{\"id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"arguments\":\"{\\\"location\\\": \\\"Paris, France\\\"}\"}}]},"
                                           "{\"role\":\"tool\",\"tool_call_id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"content\":\"15 degrees Celsius\"}],\"model\":\"llama\","
                                           "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather1\",\"description\":\"Get current temperature for a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"City and country e.g. Bogot\xC3\xA1, Colombia\"}},\"required\":[\"location\"],\"additionalProperties\":false},\"strict\":true}}],"
                                           "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather1\"}}}");

    assertRequestWithTools(providedTools, toolsChoice, expectedJson);
}

// Provide get_weather1, get_weather2, get_weather3 but take only second one - get_weather2
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceMiddle) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"({"type": "function", "function": {"name": "get_weather2"}})";
    std::string expectedJson = std::string("{\"messages\":["
                                           "{\"role\":\"user\",\"content\":\"What is the weather like in Paris today?\"},"
                                           "{\"role\":\"assistant\",\"reasoning_content\":null,\"content\":\"\",\"tool_calls\":[{\"id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"arguments\":\"{\\\"location\\\": \\\"Paris, France\\\"}\"}}]},"
                                           "{\"role\":\"tool\",\"tool_call_id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"content\":\"15 degrees Celsius\"}],\"model\":\"llama\","
                                           "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"description\":\"Get current temperature for a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"City and country e.g. Bogot\xC3\xA1, Colombia\"}},\"required\":[\"location\"],\"additionalProperties\":false},\"strict\":true}}],"
                                           "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather2\"}}}");

    assertRequestWithTools(providedTools, toolsChoice, expectedJson);
}

// Provide get_weather1, get_weather2, get_weather3 but take only second one - get_weather2
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceLast) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"({"type": "function", "function": {"name": "get_weather3"}})";
    std::string expectedJson = std::string("{\"messages\":["
                                           "{\"role\":\"user\",\"content\":\"What is the weather like in Paris today?\"},"
                                           "{\"role\":\"assistant\",\"reasoning_content\":null,\"content\":\"\",\"tool_calls\":[{\"id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"arguments\":\"{\\\"location\\\": \\\"Paris, France\\\"}\"}}]},"
                                           "{\"role\":\"tool\",\"tool_call_id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"content\":\"15 degrees Celsius\"}],\"model\":\"llama\","
                                           "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather3\",\"description\":\"Get current temperature for a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"City and country e.g. Bogot\xC3\xA1, Colombia\"}},\"required\":[\"location\"],\"additionalProperties\":false},\"strict\":true}}],"
                                           "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather3\"}}}");

    assertRequestWithTools(providedTools, toolsChoice, expectedJson);
}

// Provide get_weather1, get_weather2, get_weather3 but take one - get_weather4 which does not exist
// Expect OK and no tool selected
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceNotInProvidedList) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"({"type": "function", "function": {"name": "get_weather4"}})";
    std::string expectedJson = std::string("{\"messages\":["
                                           "{\"role\":\"user\",\"content\":\"What is the weather like in Paris today?\"},"
                                           "{\"role\":\"assistant\",\"reasoning_content\":null,\"content\":\"\",\"tool_calls\":[{\"id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"arguments\":\"{\\\"location\\\": \\\"Paris, France\\\"}\"}}]},"
                                           "{\"role\":\"tool\",\"tool_call_id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"content\":\"15 degrees Celsius\"}],\"model\":\"llama\","
                                           "\"tools\":[],"
                                           "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather4\"}}}");

    assertRequestWithTools(providedTools, toolsChoice, expectedJson);
}

// Provide get_weather1, get_weather2, get_weather3 but tool_choice is not of type function
// Expect that tool is picked anyway
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceIsNotOfTypeFunction) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"({"type": "INVALID_TYPE", "function": {"name": "get_weather3"}})";
    std::string expectedJson = std::string("{\"messages\":["
                                           "{\"role\":\"user\",\"content\":\"What is the weather like in Paris today?\"},"
                                           "{\"role\":\"assistant\",\"reasoning_content\":null,\"content\":\"\",\"tool_calls\":[{\"id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"arguments\":\"{\\\"location\\\": \\\"Paris, France\\\"}\"}}]},"
                                           "{\"role\":\"tool\",\"tool_call_id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"content\":\"15 degrees Celsius\"}],\"model\":\"llama\","
                                           "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather3\",\"description\":\"Get current temperature for a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"City and country e.g. Bogot\xC3\xA1, Colombia\"}},\"required\":[\"location\"],\"additionalProperties\":false},\"strict\":true}}],"
                                           "\"tool_choice\":{\"type\":\"INVALID_TYPE\",\"function\":{\"name\":\"get_weather3\"}}}");

    assertRequestWithTools(providedTools, toolsChoice, expectedJson);
}

// Provide get_weather1, get_weather2, get_weather3 but tool_choice is not an object, string but a number
// Expect error
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceIsANumber) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = "2";
    assertRequestWithTools(providedTools, toolsChoice, absl::StatusCode::kInvalidArgument);
}

// Provide get_weather1, get_weather2, get_weather3 but tool_choice is not an object, but a string selecting first tool
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceIsAString) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = "\"get_weather1\"";
    std::string expectedJson = std::string("{\"messages\":["
                                           "{\"role\":\"user\",\"content\":\"What is the weather like in Paris today?\"},"
                                           "{\"role\":\"assistant\",\"reasoning_content\":null,\"content\":\"\",\"tool_calls\":[{\"id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"type\":\"function\",\"function\":{\"name\":\"get_weather2\",\"arguments\":\"{\\\"location\\\": \\\"Paris, France\\\"}\"}}]},"
                                           "{\"role\":\"tool\",\"tool_call_id\":\"chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec\",\"content\":\"15 degrees Celsius\"}],\"model\":\"llama\","
                                           "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather1\",\"description\":\"Get current temperature for a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"City and country e.g. Bogot\xC3\xA1, Colombia\"}},\"required\":[\"location\"],\"additionalProperties\":false},\"strict\":true}}],"
                                           "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather1\"}}}");
    assertRequestWithTools(providedTools, toolsChoice, absl::StatusCode::kInvalidArgument);
}

// Provide get_weather1, get_weather2, get_weather3 but tool_choice object has name which is not string
// Expect error
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceObjectNameIsNotString) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"({"type": "function", "function": {"name": 4}})";
    assertRequestWithTools(providedTools, toolsChoice, absl::StatusCode::kInvalidArgument);
}

// Provide get_weather1, get_weather2, get_weather3 but tool_choice object has no function field
// Expect error
TEST_F(HttpOpenAIHandlerParsingTest, ParseRequestWithTools_Provided3_ChoiceObjectMissingFunctionField) {
    std::string providedTools = R"(
       {"type": "function", "function": {"name": "get_weather1", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather2", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}},
       {"type": "function", "function": {"name": "get_weather3", "description": "Get current temperature for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City and country e.g. Bogot\u00e1, Colombia"}}, "required": ["location"], "additionalProperties": false}, "strict": true}}
)";
    std::string toolsChoice = R"({"type": "function"})";
    assertRequestWithTools(providedTools, toolsChoice, absl::StatusCode::kInvalidArgument);
}

TEST_F(HttpOpenAIHandlerTest, V3ApiWithNonLLMCalculator) {
    handler.reset();
    server.setShutdownRequest(1);
    t->join();
    server.setShutdownRequest(0);
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mediapipe_dummy_kfs.json").c_str());
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);
    std::string requestBody = R"(
        {
            "model": "mediapipeDummyKFS",
            "stream": false,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
}

TEST_F(HttpOpenAIHandlerTest, DefaultContentTypeJSON) {
    std::string requestBody = "";
    endpoint = "/v3/chat/completions";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);
    ASSERT_NE(  // Not equal because we do not expect for the workload to be processed
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    ASSERT_EQ(responseComponents.contentType, ovms::ContentType::JSON);
}

TEST_F(HttpOpenAIHandlerTest, MetricsEndpointContentTypePlainText) {
    std::string requestBody = "";
    endpoint = "/metrics";
    ASSERT_EQ(handler->parseRequestComponents(comp, "GET", endpoint, headers), ovms::StatusCode::OK);
    ASSERT_NE(  // Not equal because we do not expect for the workload to be processed
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    ASSERT_EQ(responseComponents.contentType, ovms::ContentType::PLAIN_TEXT);
}

TEST_F(HttpOpenAIHandlerParsingTest, responseFormatValid) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": "prompt"
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "schema": {
          "type": "object",
          "properties": {
            "text": {
              "type": "string"
            }
          },
          "required": ["text"]
        }
      }
    }
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    // Response format is converted from OpenAI compatible to XGrammar compatible
    std::string expectedResponseFormat = R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}})";
    uint32_t bestOfLimit = 0;
    uint32_t maxTokensLimit = 30;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_TRUE(apiHandler->getResponseFormat().has_value());

    // Compare JSONs
    rapidjson::Document expectedDoc;
    expectedDoc.Parse(expectedResponseFormat.c_str());
    ASSERT_FALSE(expectedDoc.HasParseError());

    rapidjson::Document actualDoc;
    actualDoc.Parse(apiHandler->getResponseFormat().value().c_str());
    ASSERT_FALSE(actualDoc.HasParseError());

    EXPECT_TRUE(expectedDoc == actualDoc);
}

TEST_F(HttpOpenAIHandlerParsingTest, responseFormatMissingSchema) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": "prompt"
      }
    ],  
    "response_format": {
      "type": "json_schema",
      "json_schema": "invalid_schema"
      }
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    uint32_t bestOfLimit = 0;
    uint32_t maxTokensLimit = 10;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    // Response format content is not validated by OVMS. Any error would be raised by XGrammar during generation config validation which happens after request parsing.
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
}

TEST_F(HttpOpenAIHandlerParsingTest, responseFormatNullValue) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {
        "role": "user",
        "content": "prompt"
      }
    ],
    "response_format": null
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    uint32_t bestOfLimit = 0;
    uint32_t maxTokensLimit = 10;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_FALSE(apiHandler->getResponseFormat().has_value());
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsWithBooleanValue) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": {"enable_thinking": true}
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsStatus.ok());
    const auto& kwargs = kwargsStatus.value();
    ASSERT_TRUE(kwargs.has_value());
    EXPECT_TRUE(kwargs->is_object());
    ASSERT_TRUE((*kwargs)["enable_thinking"].as_bool().has_value());
    EXPECT_EQ((*kwargs)["enable_thinking"].as_bool().value(), true);
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsWithMultipleValues) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": {"enable_thinking": false, "custom_param": "value"}
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsStatus.ok());
    const auto& kwargs = kwargsStatus.value();
    ASSERT_TRUE(kwargs.has_value());
    EXPECT_TRUE(kwargs->is_object());
    ASSERT_TRUE((*kwargs)["enable_thinking"].as_bool().has_value());
    EXPECT_EQ((*kwargs)["enable_thinking"].as_bool().value(), false);
    ASSERT_TRUE((*kwargs)["custom_param"].as_string().has_value());
    EXPECT_EQ((*kwargs)["custom_param"].as_string().value(), "value");
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsEmptyObject) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": {}
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsStatus.ok());
    const auto& kwargs = kwargsStatus.value();
    ASSERT_TRUE(kwargs.has_value());
    EXPECT_TRUE(kwargs->is_object());
    EXPECT_EQ(kwargs->size(), 0);
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsNull) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": null
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsStatus.ok());
    EXPECT_FALSE(kwargsStatus.value().has_value());
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsAbsent) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsStatus.ok());
    EXPECT_FALSE(kwargsStatus.value().has_value());
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsInvalidString) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": "not_an_object"
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_FALSE(kwargsStatus.ok());
    EXPECT_EQ(kwargsStatus.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(std::string(kwargsStatus.status().message()), ::testing::HasSubstr("chat_template_kwargs must be an object"));
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsInvalidArray) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": [1, 2, 3]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_FALSE(kwargsStatus.ok());
    EXPECT_EQ(kwargsStatus.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(std::string(kwargsStatus.status().message()), ::testing::HasSubstr("chat_template_kwargs must be an object"));
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsInvalidNumber) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": 42
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_FALSE(kwargsStatus.ok());
    EXPECT_EQ(kwargsStatus.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(std::string(kwargsStatus.status().message()), ::testing::HasSubstr("chat_template_kwargs must be an object"));
}

TEST_F(HttpOpenAIHandlerParsingTest, parseChatTemplateKwargsWithNestedObject) {
    std::string json = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": {"documents": [{"title": "doc1", "text": "content1"}], "enable_thinking": true}
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsStatus.ok());
    const auto& kwargs = kwargsStatus.value();
    ASSERT_TRUE(kwargs.has_value());
    EXPECT_TRUE(kwargs->is_object());
    ASSERT_TRUE((*kwargs)["enable_thinking"].as_bool().has_value());
    EXPECT_EQ((*kwargs)["enable_thinking"].as_bool().value(), true);
    EXPECT_TRUE((*kwargs)["documents"].is_array());
    EXPECT_EQ((*kwargs)["documents"].size(), 1);
}

// Integration test: verifies that chat_template_kwargs extracted from request
// are actually consumed by GenAI tokenizer.apply_chat_template as extra_context.
// This mirrors the non-Python (#else) path in GenAiServable::prepareInputs.
TEST_F(HttpOpenAIHandlerParsingTest, chatTemplateKwargsConsumedByApplyChatTemplate) {
    // Set a custom chat template that branches on enable_thinking kwarg
    std::string chatTemplate = R"({% if enable_thinking is defined and enable_thinking %}<think>{% endif %}{% for message in messages %}{{ message['content'] }}{% endfor %}{% if enable_thinking is defined and enable_thinking %}</think>{% endif %})";
    tokenizer->set_chat_template(chatTemplate);

    // Parse request with enable_thinking = true
    std::string jsonTrue = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": {"enable_thinking": true}
  })";
    doc.Parse(jsonTrue.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::optional<uint32_t> maxTokensLimit;
    uint32_t bestOfLimit = 0;
    std::optional<uint32_t> maxModelLength;
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());

    // Extract kwargs and chat history - same as servable code does
    auto kwargsStatus = apiHandler->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsStatus.ok());
    const auto& chatTemplateKwargs = kwargsStatus.value();
    ov::genai::ChatHistory& chatHistory = apiHandler->getChatHistory();

    // Call apply_chat_template with extra_context, exactly as the servable does
    constexpr bool add_generation_prompt = true;
    std::string result = tokenizer->apply_chat_template(chatHistory, add_generation_prompt, {}, std::nullopt, chatTemplateKwargs);
    EXPECT_EQ(result, "<think>hello</think>");

    // Now test with enable_thinking = false
    rapidjson::Document docFalse;
    std::string jsonFalse = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}],
    "chat_template_kwargs": {"enable_thinking": false}
  })";
    docFalse.Parse(jsonFalse.c_str());
    ASSERT_FALSE(docFalse.HasParseError());
    auto apiHandlerFalse = std::make_shared<ovms::OpenAIChatCompletionsHandler>(docFalse, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandlerFalse->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsFalseStatus = apiHandlerFalse->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsFalseStatus.ok());
    std::string resultFalse = tokenizer->apply_chat_template(apiHandlerFalse->getChatHistory(), add_generation_prompt, {}, std::nullopt, kwargsFalseStatus.value());
    EXPECT_EQ(resultFalse, "hello");

    // And with no kwargs at all
    rapidjson::Document docNone;
    std::string jsonNone = R"({
    "model": "llama",
    "messages": [{"role": "user", "content": "hello"}]
  })";
    docNone.Parse(jsonNone.c_str());
    ASSERT_FALSE(docNone.HasParseError());
    auto apiHandlerNone = std::make_shared<ovms::OpenAIChatCompletionsHandler>(docNone, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandlerNone->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    auto kwargsNoneStatus = apiHandlerNone->parseChatTemplateKwargsToJsonContainer();
    ASSERT_TRUE(kwargsNoneStatus.ok());
    std::string resultNone = tokenizer->apply_chat_template(apiHandlerNone->getChatHistory(), add_generation_prompt, {}, std::nullopt, kwargsNoneStatus.value());
    EXPECT_EQ(resultNone, "hello");
}

TEST_F(HttpOpenAIHandlerParsingTest, ParseMessagesToolCallsStoredInChatHistory) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {"role": "user", "content": "What is the weather like in Paris today?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}}]},
      {"role": "tool", "tool_call_id": "call_123", "name": "get_weather", "content": "15 degrees Celsius"}
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());

    ov::genai::ChatHistory& history = apiHandler->getChatHistory();
    ASSERT_EQ(history.size(), 3);

    // Message 0: user message with role and content
    auto msg0 = history[0];
    ASSERT_TRUE(msg0.contains("role"));
    EXPECT_EQ(msg0["role"].get_string(), "user");
    ASSERT_TRUE(msg0.contains("content"));
    EXPECT_EQ(msg0["content"].get_string(), "What is the weather like in Paris today?");

    // Message 1: assistant message with tool_calls array
    // Note: null content in JSON gets replaced with empty string by parseMessages
    auto msg1 = history[1];
    ASSERT_TRUE(msg1.contains("role"));
    EXPECT_EQ(msg1["role"].get_string(), "assistant");
    ASSERT_TRUE(msg1.contains("content"));
    EXPECT_EQ(msg1["content"].get_string(), "");
    ASSERT_TRUE(msg1.contains("tool_calls"));
    EXPECT_TRUE(msg1["tool_calls"].is_array());
    ASSERT_EQ(msg1["tool_calls"].size(), 1);
    EXPECT_EQ(msg1["tool_calls"][0]["id"].get_string(), "call_123");
    EXPECT_EQ(msg1["tool_calls"][0]["type"].get_string(), "function");
    EXPECT_EQ(msg1["tool_calls"][0]["function"]["name"].get_string(), "get_weather");
    EXPECT_EQ(msg1["tool_calls"][0]["function"]["arguments"].get_string(), "{\"location\": \"Paris\"}");

    // Message 2: tool message with tool_call_id, name, and content
    auto msg2 = history[2];
    ASSERT_TRUE(msg2.contains("role"));
    EXPECT_EQ(msg2["role"].get_string(), "tool");
    ASSERT_TRUE(msg2.contains("tool_call_id"));
    EXPECT_EQ(msg2["tool_call_id"].get_string(), "call_123");
    ASSERT_TRUE(msg2.contains("name"));
    EXPECT_EQ(msg2["name"].get_string(), "get_weather");
    ASSERT_TRUE(msg2.contains("content"));
    EXPECT_EQ(msg2["content"].get_string(), "15 degrees Celsius");
}

TEST_F(HttpOpenAIHandlerParsingTest, ParseMessagesMultipleToolCallsStoredInChatHistory) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {"role": "user", "content": "Compare weather in Paris and London"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}},
        {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "15C"},
      {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "12C"}
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());

    ov::genai::ChatHistory& history = apiHandler->getChatHistory();
    ASSERT_EQ(history.size(), 4);

    // Assistant message should have 2 tool calls
    auto msg1 = history[1];
    ASSERT_TRUE(msg1.contains("tool_calls"));
    ASSERT_EQ(msg1["tool_calls"].size(), 2);
    EXPECT_EQ(msg1["tool_calls"][0]["id"].get_string(), "call_1");
    EXPECT_EQ(msg1["tool_calls"][1]["id"].get_string(), "call_2");

    // Both tool response messages should have tool_call_id
    EXPECT_EQ(history[2]["tool_call_id"].get_string(), "call_1");
    EXPECT_EQ(history[3]["tool_call_id"].get_string(), "call_2");
}

TEST_F(HttpOpenAIHandlerParsingTest, ParseMessagesAssistantWithNullContentAndNoToolCalls) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": null}
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());

    ov::genai::ChatHistory& history = apiHandler->getChatHistory();
    ASSERT_EQ(history.size(), 2);

    // Null content gets replaced with empty string by parseMessages fallback logic
    auto msg1 = history[1];
    ASSERT_TRUE(msg1.contains("content"));
    EXPECT_EQ(msg1["content"].get_string(), "");
    EXPECT_FALSE(msg1.contains("tool_calls"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParseMessagesToolCallsWithMissingArgumentsGetsDefault) {
    // Verify that ensureArgumentsInToolCalls still works after tool_calls are stored in chat history
    std::string json = R"({
    "model": "llama",
    "messages": [
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "no_args_tool"}}]}
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());

    ov::genai::ChatHistory& history = apiHandler->getChatHistory();
    ASSERT_EQ(history.size(), 2);

    // tool_calls should be present in chat history
    auto msg1 = history[1];
    ASSERT_TRUE(msg1.contains("tool_calls"));
    EXPECT_EQ(msg1["tool_calls"].size(), 1);
    EXPECT_EQ(msg1["tool_calls"][0]["function"]["name"].get_string(), "no_args_tool");
}

TEST_F(HttpOpenAIHandlerParsingTest, ParseMessagesRegularMessageHasNoToolFields) {
    std::string json = R"({
    "model": "llama",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "hello"}
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    auto apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());

    ov::genai::ChatHistory& history = apiHandler->getChatHistory();
    ASSERT_EQ(history.size(), 2);

    // Regular messages should not have tool-related fields
    EXPECT_FALSE(history[0].contains("tool_calls"));
    EXPECT_FALSE(history[0].contains("tool_call_id"));
    EXPECT_FALSE(history[0].contains("name"));
    EXPECT_FALSE(history[1].contains("tool_calls"));
    EXPECT_FALSE(history[1].contains("tool_call_id"));
    EXPECT_FALSE(history[1].contains("name"));
}
