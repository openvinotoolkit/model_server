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
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <continuous_batching_pipeline.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <pybind11/embed.h>

#include "../config.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../http_rest_api_handler.hpp"
#include "../httpservermodule.hpp"
#include "../kfs_frontend/kfs_graph_executor_impl.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../llm/llmnoderesources.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../python/pythoninterpretermodule.hpp"
#include "../python/pythonnoderesources.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_runner.h"
#pragma GCC diagnostic pop

#include "opencv2/opencv.hpp"
#include "test_utils.hpp"

using namespace ovms;

class MockedServerRequestInterface final : public tensorflow::serving::net_http::ServerRequestInterface {
public:
    MOCK_METHOD(absl::string_view, uri_path, (), (const, override));
    MOCK_METHOD(absl::string_view, http_method, (), (const, override));
    MOCK_METHOD(void, WriteResponseBytes, (const char*, int64_t), (override));
    MOCK_METHOD(void, WriteResponseString, (absl::string_view), (override));
    MOCK_METHOD((std::unique_ptr<char[], tensorflow::serving::net_http::ServerRequestInterface::BlockDeleter>), ReadRequestBytes, (int64_t*), (override));
    MOCK_METHOD(absl::string_view, GetRequestHeader, (absl::string_view), (const, override));
    MOCK_METHOD((std::vector<absl::string_view>), request_headers, (), (const, override));
    MOCK_METHOD(void, OverwriteResponseHeader, (absl::string_view, absl::string_view), (override));
    MOCK_METHOD(void, AppendResponseHeader, (absl::string_view, absl::string_view), (override));
    MOCK_METHOD(void, PartialReplyWithStatus, (tensorflow::serving::net_http::HTTPStatusCode), (override));
    MOCK_METHOD(void, PartialReply, (std::string), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::CallbackStatus, PartialReplyWithFlushCallback, ((std::function<void()>)), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, response_body_status, (), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, request_body_status, (), (override));
    MOCK_METHOD(void, ReplyWithStatus, (tensorflow::serving::net_http::HTTPStatusCode), (override));
    MOCK_METHOD(void, Reply, (), (override));
    MOCK_METHOD(void, Abort, (), (override));
    MOCK_METHOD(void, PartialReplyEnd, (), (override));
};

class LLMFlowHttpTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v3/chat/completions";
    const std::string endpointCompletions = "/v3/completions";
    MockedServerRequestInterface writer;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpTestSuite() {
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, "/ovms/src/test/llm/config_llm_dummy_kfs.json");
        auto start = std::chrono::high_resolution_clock::now();
        const int numberOfRetries = 5;
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < numberOfRetries)) {
        }
    }

    void SetUp() {
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
std::unique_ptr<std::thread> LLMFlowHttpTest::t;

// --------------------------------------- OVMS LLM nodes tests

// TODO: Test bad sampling configuration that would cause errors in step() phase. Need to replace hardcoded generation config
// with user defined one to do that.
// TODO: Test bad message or sampling configuration that would cause errors in add_request() phase. Need to replace hardcoded generation config
// with user defined one to do that.
// TODO: Consider stress testing - exisitng model server under heavy load to check notifications work us expected.

TEST_F(LLMFlowHttpTest, inferChatCompletionsUnary) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
    // Assertion split in two parts to avoid timestamp missmatch
    const size_t timestampLength = 10;
    std::string expectedResponsePart1 = R"({"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"\nOpenVINO is","role":"assistant"}}],"created":)";
    std::string expectedResponsePart2 = R"(,"model":"llmDummyKFS","object":"chat.completion"})";
    ASSERT_EQ(response.compare(0, expectedResponsePart1.length(), expectedResponsePart1), 0);
    ASSERT_EQ(response.compare(expectedResponsePart1.length() + timestampLength, expectedResponsePart2.length(), expectedResponsePart2), 0);
}

TEST_F(LLMFlowHttpTest, inferCompletionsUnary) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
    // Assertion split in two parts to avoid timestamp missmatch
    const size_t timestampLength = 10;
    std::string expectedResponsePart1 = R"({"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"text":"\nOpenVINO is"}],"created":)";
    std::string expectedResponsePart2 = R"(,"model":"llmDummyKFS","object":"text_completion"})";
    ASSERT_EQ(response.compare(0, expectedResponsePart1.length(), expectedResponsePart1), 0);
    ASSERT_EQ(response.compare(expectedResponsePart1.length() + timestampLength, expectedResponsePart2.length(), expectedResponsePart2), 0);
}

TEST_F(LLMFlowHttpTest, inferChatCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(3);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_EQ(response, "");
}

TEST_F(LLMFlowHttpTest, inferCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(3);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_EQ(response, "");
}

const std::string validRequestBodyWithParameter(const std::string& parameter, const std::string& value) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "max_tokens": 1,
            ")" + parameter + R"(": )" +
                              value + R"(,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    return requestBody;
}

class LLMHttpParametersValidationTest : public LLMFlowHttpTest {};

TEST_F(LLMHttpParametersValidationTest, maxTokensInvalid) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": "INVALID",
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, streamInvalid) {
    std::string requestBody = validRequestBodyWithParameter("stream", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::JSON_INVALID);
}

TEST_F(LLMHttpParametersValidationTest, messagesInvalid) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "messages": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, messagesMissing) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, messageNotAnObject) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "messages": [
                "What is OpenVINO?"
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, messageNotAString) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": 1
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, roleNotAString) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": false,
                "content": "What is OpenVino?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, promptInvalid) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "prompt": 5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, promptMissing) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, modelMissing) {
    std::string requestBody = R"(
        {
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::JSON_INVALID);
}

TEST_F(LLMHttpParametersValidationTest, modelInvalid) {
    std::string requestBody = R"(
        {
            "model": 0,
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::JSON_INVALID);
}

TEST_F(LLMHttpParametersValidationTest, ignoreEosValid) {
    std::string requestBody = validRequestBodyWithParameter("ignore_eos", "false");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, ignoreEosInvalid) {
    std::string requestBody = validRequestBodyWithParameter("ignore_eos", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, repetitionPenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("repetition_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, repetitionPenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("repetition_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, diversityPenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("diversity_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, diversityPenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("diversity_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, lengthPenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("length_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, lengthPenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("length_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, temperatureValid) {
    std::string requestBody = validRequestBodyWithParameter("temperature", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, temperatureInvalid) {
    std::string requestBody = validRequestBodyWithParameter("temperature", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, temperatureOutOfRange) {
    std::string requestBody = validRequestBodyWithParameter("temperature", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, topPValid) {
    std::string requestBody = validRequestBodyWithParameter("top_p", "0.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, topPInvalid) {
    std::string requestBody = validRequestBodyWithParameter("top_p", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, topPOutOfRange) {
    std::string requestBody = validRequestBodyWithParameter("top_p", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, topKValid) {
    std::string requestBody = validRequestBodyWithParameter("top_k", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, topKInvalid) {
    std::string requestBody = validRequestBodyWithParameter("top_k", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, seedValid) {
    std::string requestBody = validRequestBodyWithParameter("seed", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, seedInvalid) {
    std::string requestBody = validRequestBodyWithParameter("seed", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, bestOfValid) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, bestOfInvalid) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, bestOfNegative) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, nValid) {
    std::string requestBody = validRequestBodyWithParameter("n", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, nInvalid) {
    std::string requestBody = validRequestBodyWithParameter("n", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, nNegative) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, nGreaterThanBestOf) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "best_of" : 1,
            "n" : 2,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

class LLMConfigHttpTest : public ::testing::Test {
public:
    void SetUp() { py::initialize_interpreter(); }
    void TearDown() { py::finalize_interpreter(); }
};

TEST_F(LLMConfigHttpTest, LLMNodeNameMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "./"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_MISSING_NAME);
}

TEST_F(LLMConfigHttpTest, LLMNodeOptionsMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "LLMExecutor"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_MISSING_OPTIONS);
}

TEST_F(LLMConfigHttpTest, LLMNodeNameExists) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD_1:input"
        input_stream: "HTTP_REQUEST_PAYLOAD_2:input2"
        output_stream: "HTTP_RESPONSE_PAYLOAD_1:output"
        output_stream: "HTTP_RESPONSE_PAYLOAD_2:output2"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/llm_testing/facebook/opt-125m"
                cache_size: 1
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback2"
        input_stream: "HTTP_REQUEST_PAYLOAD:input2"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback2"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output2"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/llm_testing/facebook/opt-125m"
                cache_size: 1
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_NAME_ALREADY_EXISTS);
}

TEST_F(LLMConfigHttpTest, LLMNodeNonExistantModelsPath) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/models_path"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
}

TEST_F(LLMConfigHttpTest, LLMNodeBadWorkspacePathEmpty) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ""
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
}

TEST_F(LLMConfigHttpTest, LLMNodeWorkspacePathToFileNotDir) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/llm_testing/facebook/opt-125m/config.json"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
}

TEST_F(LLMConfigHttpTest, LLMNodeResourceInitFailed) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
    ASSERT_EQ(mediapipeDummy.getLLMNodeResources("llmNode"), nullptr);
}

class LLMOptionsHttpTest : public ::testing::Test {
public:
    void SetUp() { py::initialize_interpreter(); }
    void TearDown() { py::finalize_interpreter(); }
};

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckDefault) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/llm_testing/facebook/opt-125m"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    std::cout << "------------------------A--------------------\n";
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<LLMNodeResources> nodeResources = nullptr;
    ASSERT_EQ(LLMNodeResources::createLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::OK);
    std::cout << "------------------------B--------------------\n";
    ASSERT_EQ(nodeResources->schedulerConfig.max_num_batched_tokens, 256);
    ASSERT_EQ(nodeResources->schedulerConfig.cache_size, 8);
    ASSERT_EQ(nodeResources->schedulerConfig.block_size, 32);
    ASSERT_EQ(nodeResources->schedulerConfig.dynamic_split_fuse, true);
    ASSERT_EQ(nodeResources->schedulerConfig.max_num_seqs, 256);
    ASSERT_EQ(nodeResources->device, "CPU");
    ASSERT_EQ(nodeResources->pluginConfig.size(), 0);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckHalfDefault) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/llm_testing/facebook/opt-125m"
                max_num_batched_tokens: 98
                cache_size: 1
                block_size: 16
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<LLMNodeResources> nodeResources = nullptr;
    ASSERT_EQ(LLMNodeResources::createLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::OK);

    ASSERT_EQ(nodeResources->schedulerConfig.max_num_batched_tokens, 98);
    ASSERT_EQ(nodeResources->schedulerConfig.cache_size, 1);
    ASSERT_EQ(nodeResources->schedulerConfig.block_size, 16);
    ASSERT_EQ(nodeResources->schedulerConfig.dynamic_split_fuse, true);
    ASSERT_EQ(nodeResources->schedulerConfig.max_num_seqs, 256);
    // TODO: Check plugin config
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsWrongPluginFormat) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/llm_testing/facebook/opt-125m"
                cache_size: 1
                plugin_config: "[PERF_COUNT=TRUE]"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<LLMNodeResources> nodeResources = nullptr;
    ASSERT_EQ(LLMNodeResources::createLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::PLUGIN_CONFIG_WRONG_FORMAT);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckNonDefault) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/llm_testing/facebook/opt-125m"
                max_num_batched_tokens: 1024
                cache_size: 1
                block_size: 8
                max_num_seqs: 95
                dynamic_split_fuse: false
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<LLMNodeResources> nodeResources = nullptr;
    ASSERT_EQ(LLMNodeResources::createLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::OK);

    ASSERT_EQ(nodeResources->schedulerConfig.max_num_batched_tokens, 1024);
    ASSERT_EQ(nodeResources->schedulerConfig.cache_size, 1);
    ASSERT_EQ(nodeResources->schedulerConfig.block_size, 8);
    ASSERT_EQ(nodeResources->schedulerConfig.dynamic_split_fuse, false);
    ASSERT_EQ(nodeResources->schedulerConfig.max_num_seqs, 95);
}
