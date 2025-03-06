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

class HttpOpenAIHandlerTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

    std::unordered_map<std::string, std::string> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpoint = "/v3/chat/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }

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

TEST_F(HttpOpenAIHandlerTest, Unary) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/v1/completions/", requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(/v3/v1/completions/

        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
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

    std::string expectedResponse = R"(/v3/completions/
test1headertest2header
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
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
};

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesSucceeds) {
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
    std::vector<ov::Tensor> images = apiHandler->getImages();
    ASSERT_EQ(images.size(), 1);
    ov::Tensor image = images[0];
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 3);
    std::vector<uint8_t> expectedBytes = {110, 181, 160};
    for (size_t i = 0; i < image.get_size(); i++) {
        EXPECT_EQ(expectedBytes[i], ((uint8_t*)image.data())[i]);
    }
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
    std::vector<ov::Tensor> images = apiHandler->getImages();
    ASSERT_EQ(images.size(), 1);
    ov::Tensor image = images[0];
    EXPECT_EQ(image.get_element_type(), ov::element::u8);
    EXPECT_EQ(image.get_size(), 3);
    std::vector<uint8_t> expectedBytes = {54, 245, 241};
    for (size_t i = 0; i < image.get_size(); i++) {
        EXPECT_EQ(expectedBytes[i], ((uint8_t*)image.data())[i]);
    }
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":null}]}"));
}

TEST_F(HttpOpenAIHandlerParsingTest, ParsingMessagesImageStringWithNoPrefixSucceeds) {
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
    EXPECT_EQ(apiHandler->parseMessages(), absl::InvalidArgumentError("Url should contain base64 encoded string followed by \"base64,\" prefix"));
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
        "role": "system",
        "content": "What is Openvino?"
      }
    ]
  })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    std::shared_ptr<ovms::OpenAIChatCompletionsHandler> apiHandler = std::make_shared<ovms::OpenAIChatCompletionsHandler>(doc, ovms::Endpoint::CHAT_COMPLETIONS, std::chrono::system_clock::now(), *tokenizer);
    ASSERT_EQ(apiHandler->parseMessages(), absl::OkStatus());
    std::vector<ov::Tensor> images = apiHandler->getImages();
    ASSERT_EQ(images.size(), 2);
    std::vector<uint8_t> expectedBytes = {110, 181, 160};
    for (auto image : images) {
        EXPECT_EQ(image.get_element_type(), ov::element::u8);
        EXPECT_EQ(image.get_size(), 3);
        for (size_t i = 0; i < image.get_size(); i++) {
            EXPECT_EQ(expectedBytes[i], ((uint8_t*)image.data())[i]);
        }
    }
    json = apiHandler->getProcessedJson();
    EXPECT_EQ(json, std::string("{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"What is in this image?\"},{\"role\":\"user\",\"content\":\"What is in this image?\"},{\"role\":\"system\",\"content\":\"What is Openvino?\"}]}"));
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
    EXPECT_EQ(apiHandler->parseMessages(), absl::InvalidArgumentError("Url should contain base64 encoded string followed by \"base64,\" prefix"));
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
