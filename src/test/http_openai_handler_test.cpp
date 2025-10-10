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
    const std::string endpoint = "/v3/chat/completions";
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
        ::SetUpServer(this->t, this->server, this->port, configPath, 10, absoluteApiKeyPath);
        EnsureServerStartedWithTimeout(this->server, 20);
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        // remove temp file with api key
        std::filesystem::remove(absoluteApiKeyPath);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/empty_subconfig.json").c_str());
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
    auto status = handler->processV3("/v3/completions", comp, response, requestBody, streamPtr, multiPartParser, "1234");
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
    auto status = handler->processV3("/v3/completions", comp, response, requestBody, streamPtr, multiPartParser, "1234");
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
    auto status = handler->processV3("/v3/completions", comp, response, requestBody, streamPtr, multiPartParser, "1234");
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

TEST_F(HttpOpenAIHandlerTest, ModelFieldMissing) {
    std::string requestBody = R"(
        {
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - model field is missing in JSON body");
}

TEST_F(HttpOpenAIHandlerTest, ModelFieldNotAString) {
    std::string requestBody = R"(
        {
            "model": 2,
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - model field is not a string");
}

TEST_F(HttpOpenAIHandlerTest, StreamFieldNotABoolean) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": 2,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyBegin(::testing::_)).Times(0);
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - stream field is not a boolean");
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
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());
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
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());
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
    std::string expectedReponseFormatSchema = R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})";
    uint32_t bestOfLimit = 0;
    uint32_t maxTokensLimit = 30;
    std::optional<uint32_t> maxModelLength;
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::OkStatus());
    EXPECT_TRUE(apiHandler->getResponseSchema().has_value());
    EXPECT_EQ(apiHandler->getResponseSchema().value(), expectedReponseFormatSchema);
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
    EXPECT_EQ(apiHandler->parseRequest(maxTokensLimit, bestOfLimit, maxModelLength), absl::InvalidArgumentError("response_format.json_schema is not an object"));
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
    EXPECT_FALSE(apiHandler->getResponseSchema().has_value());
}
