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
#include <atomic>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../../../http_rest_api_handler.hpp"
#include "../../../http_status_code.hpp"
#include "../../../json_parser.hpp"
#include "../../../llm/apis/openai_completions.hpp"
#include "../../../ov_utils.hpp"
#include "../../../server.hpp"
#include "../../test_http_utils.hpp"
#include "../../test_utils.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

using namespace ovms;

class VLMServableExecutionTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v3/chat/completions";
    const std::string endpointResponses = "/v3/responses";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    rapidjson::Document parsedResponse;
    ovms::HttpResponseComponents responseComponents;
    static void SetUpTestSuite() {
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/visual_language_model/config.json").c_str(), 60);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));  // make the streaming flow sequential
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
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

std::unique_ptr<std::thread> VLMServableExecutionTest::t;

static std::string createRequestBody(const std::string& modelName, const std::vector<std::pair<std::string, std::string>>& fields, bool includeText = true, int numberOfImages = 1, const std::string contentOfTheFirstMessage = "What is in this image?") {
    std::ostringstream oss;
    oss << R"(
        {
            "model": ")"
        << modelName << R"(",
            "messages": [
            {
                "role": "user",
                "content": [)";
    if (includeText) {
        oss << R"(
                    {
                        "type": "text",
                        "text": ")";
        oss << contentOfTheFirstMessage;
        oss << R"("})";
        if (numberOfImages > 0) {
            oss << ",";
        }
    }
    for (int i = 0; i < numberOfImages; i++) {
        oss << R"(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGIy+/oREAAA//8DiQIftNKCRwAAAABJRU5ErkJggg=="
                        }
                    })";
        if (i < numberOfImages - 1) {
            oss << ",";
        }
    }
    oss << R"(
                ]
            }
            ]
        )";
    for (const auto& field : fields) {
        oss << R"(, ")" << field.first << R"(": )" << field.second << R"()"
            << "\n";
    }
    oss << "\n}";
    return oss.str();
}

static std::string createResponsesRequestBody(const std::string& modelName, const std::vector<std::pair<std::string, std::string>>& fields, bool includeText = true, int numberOfImages = 1, const std::string contentOfTheFirstMessage = "What is in this image?") {
    std::ostringstream oss;
    oss << R"(
        {
            "model": ")"
        << modelName << R"(",
            "input": [
            {
                "role": "user",
                "content": [)";
    if (includeText) {
        oss << R"(
                    {
                        "type": "input_text",
                        "text": ")";
        oss << contentOfTheFirstMessage;
        oss << R"("})";
        if (numberOfImages > 0) {
            oss << ",";
        }
    }
    for (int i = 0; i < numberOfImages; i++) {
        oss << R"(
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGIy+/oREAAA//8DiQIftNKCRwAAAABJRU5ErkJggg=="
                    })";
        if (i < numberOfImages - 1) {
            oss << ",";
        }
    }
    oss << R"(
                ]
            }
            ]
        )";
    for (const auto& field : fields) {
        oss << R"(, ")" << field.first << R"(": )" << field.second << R"()"
            << "\n";
    }
    oss << "\n}";
    return oss.str();
}

class VLMServableExecutionTestParameterized : public VLMServableExecutionTest, public ::testing::WithParamInterface<std::string> {};

// Unary flow

TEST_P(VLMServableExecutionTestParameterized, unaryBasic) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "false"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (modelName.find("legacy") == std::string::npos) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_EQ(choice["index"], i++);
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

// Only image input is accepted, but expected output can't be predicted
TEST_P(VLMServableExecutionTestParameterized, unaryBasicOnlyImage) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "false"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, false, 1);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (modelName.find("legacy") == std::string::npos) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_EQ(choice["index"], i++);
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

// Images are accepted, but expected output can't be predicted
TEST_P(VLMServableExecutionTestParameterized, unaryMultipleImageTagOrderPasses) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "false"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, false, 3);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (modelName.find("legacy") == std::string::npos) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_EQ(choice["index"], i++);
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_P(VLMServableExecutionTestParameterized, UnaryRestrictedTagUsed) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "false"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, true, 1, "<ov_genai_image_2>");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(VLMServableExecutionTestParameterized, unaryBasicWithTools) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "false"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"},
        {"tool_choice", R"("auto")"},
        {"tools", R"([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ])"}};
    std::string requestBody = createRequestBody(modelName, fields);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("choices"));
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    ASSERT_TRUE(parsedResponse["choices"][0].HasMember("message"));
    ASSERT_TRUE(parsedResponse["choices"][0]["message"].IsObject());
    ASSERT_TRUE(parsedResponse["choices"][0]["message"]["content"].IsString());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
    EXPECT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
}

TEST_P(VLMServableExecutionTestParameterized, unaryResponsesWithImageInput) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"max_output_tokens", "5"},
        {"temperature", "0.0"}};
    std::string requestBody = createResponsesRequestBody(modelName, fields);

    ovms::HttpRequestComponents responsesComp;
    ASSERT_EQ(handler->parseRequestComponents(responsesComp, "POST", endpointResponses, headers), ovms::StatusCode::OK);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointResponses, requestBody, &response, responsesComp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("object"));
    EXPECT_STREQ(parsedResponse["object"].GetString(), "response");
    ASSERT_TRUE(parsedResponse.HasMember("model"));
    EXPECT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    ASSERT_TRUE(parsedResponse.HasMember("output"));
    ASSERT_TRUE(parsedResponse["output"].IsArray());
    ASSERT_GT(parsedResponse["output"].GetArray().Size(), 0);
    ASSERT_TRUE(parsedResponse["output"][0].IsObject());
    ASSERT_TRUE(parsedResponse["output"][0].HasMember("type"));
    EXPECT_STREQ(parsedResponse["output"][0]["type"].GetString(), "message");
    ASSERT_TRUE(parsedResponse["output"][0].HasMember("content"));
    ASSERT_TRUE(parsedResponse["output"][0]["content"].IsArray());
    ASSERT_GT(parsedResponse["output"][0]["content"].GetArray().Size(), 0);
    ASSERT_TRUE(parsedResponse["output"][0]["content"][0].HasMember("type"));
    EXPECT_STREQ(parsedResponse["output"][0]["content"][0]["type"].GetString(), "output_text");

    ASSERT_TRUE(parsedResponse.HasMember("usage"));
    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].HasMember("input_tokens"));
    ASSERT_TRUE(parsedResponse["usage"].HasMember("output_tokens"));
    ASSERT_TRUE(parsedResponse["usage"].HasMember("total_tokens"));
}

TEST_P(VLMServableExecutionTestParameterized, unaryResponsesOnlyImageInput) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"max_output_tokens", "5"},
        {"temperature", "0.0"}};
    std::string requestBody = createResponsesRequestBody(modelName, fields, false, 1);

    ovms::HttpRequestComponents responsesComp;
    ASSERT_EQ(handler->parseRequestComponents(responsesComp, "POST", endpointResponses, headers), ovms::StatusCode::OK);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointResponses, requestBody, &response, responsesComp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("object"));
    EXPECT_STREQ(parsedResponse["object"].GetString(), "response");
    ASSERT_TRUE(parsedResponse.HasMember("output"));
    ASSERT_TRUE(parsedResponse["output"].IsArray());
    ASSERT_GT(parsedResponse["output"].GetArray().Size(), 0);
}

TEST_P(VLMServableExecutionTestParameterized, unaryResponsesWithTools) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"max_output_tokens", "5"},
        {"temperature", "0.0"},
        {"tool_choice", R"("auto")"},
        {"tools", R"([
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string"
                        }
                    },
                    "required": ["city"]
                }
            }
        ])"}};
    std::string requestBody = createResponsesRequestBody(modelName, fields);

    ovms::HttpRequestComponents responsesComp;
    ASSERT_EQ(handler->parseRequestComponents(responsesComp, "POST", endpointResponses, headers), ovms::StatusCode::OK);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointResponses, requestBody, &response, responsesComp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("object"));
    EXPECT_STREQ(parsedResponse["object"].GetString(), "response");
    ASSERT_TRUE(parsedResponse.HasMember("tools"));
    ASSERT_TRUE(parsedResponse["tools"].IsArray());
    ASSERT_GT(parsedResponse["tools"].GetArray().Size(), 0);
    ASSERT_TRUE(parsedResponse.HasMember("tool_choice"));
    ASSERT_TRUE(parsedResponse["tool_choice"].IsString());
    EXPECT_STREQ(parsedResponse["tool_choice"].GetString(), "auto");
}

TEST_P(VLMServableExecutionTestParameterized, unaryResponsesWithFunctionToolChoiceObject) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"max_output_tokens", "5"},
        {"temperature", "0.0"},
        {"tool_choice", R"({"type":"function","name":"get_weather"})"},
        {"tools", R"([
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string"
                        }
                    },
                    "required": ["city"]
                }
            }
        ])"}};
    std::string requestBody = createResponsesRequestBody(modelName, fields);

    ovms::HttpRequestComponents responsesComp;
    ASSERT_EQ(handler->parseRequestComponents(responsesComp, "POST", endpointResponses, headers), ovms::StatusCode::OK);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointResponses, requestBody, &response, responsesComp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tool_choice"));
    ASSERT_TRUE(parsedResponse["tool_choice"].IsObject());
    ASSERT_TRUE(parsedResponse["tool_choice"].HasMember("type"));
    EXPECT_STREQ(parsedResponse["tool_choice"]["type"].GetString(), "function");
    ASSERT_TRUE(parsedResponse["tool_choice"].HasMember("name"));
    EXPECT_STREQ(parsedResponse["tool_choice"]["name"].GetString(), "get_weather");
}

// Stream flow

TEST_P(VLMServableExecutionTestParameterized, streamBasic) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "true"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields);

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    if (modelName.find("legacy") == std::string::npos) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

// Only image input is accepted, but expected output can't be predicted
TEST_P(VLMServableExecutionTestParameterized, streamBasicOnlyImage) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "true"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, false, 1);

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    if (modelName.find("legacy") == std::string::npos) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

// Images are accepted, but expected output can't be predicted
TEST_P(VLMServableExecutionTestParameterized, streamMultipleImageTagOrderPasses) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "true"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, false, 3);  // 3=number of images

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    if (modelName.find("legacy") == std::string::npos) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

TEST_P(VLMServableExecutionTestParameterized, streamRestrictedTagUsed) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "true"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, true, 1, "<ov_genai_image_2>");

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: Message contains restricted <ov_genai_image> tag\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(VLMServableExecutionTestParameterized, streamBasicWithTools) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "true"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"},
        {"tool_choice", R"("auto")"},
        {"tools", R"([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ])"}};
    std::string requestBody = createRequestBody(modelName, fields);

    std::vector<std::string> responses;
    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([&responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_FALSE(responses.empty());
}

INSTANTIATE_TEST_SUITE_P(
    VLMServableExecutionTests,
    VLMServableExecutionTestParameterized,
    ::testing::Values("vlm_cb_regular", "vlm_legacy_regular"));
