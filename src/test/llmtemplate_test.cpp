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
#include <fstream>
#include <memory>
#include <string>

#include <continuous_batching_pipeline.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <pybind11/embed.h>

#include "../filesystem.hpp"
#include "../llm/http_payload.hpp"
#include "../llm/llm_executor.hpp"
#include "../llm/llmnoderesources.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../http_rest_api_handler.hpp"
#include "../httpservermodule.hpp"
#include "../server.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_runner.h"
#pragma GCC diagnostic pop

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

class LLMChatTemplateTest : public TestWithTempDir {
private:
    bool CreateConfig(std::string& fileContents, std::string& filePath) {
        std::ofstream configFile{filePath};
        if (!configFile.is_open()) {
            std::cout << "Failed to open " << filePath << std::endl;
            return false;
        }
        SPDLOG_INFO("Creating config file: {}\n with content:\n{}", filePath, fileContents);
        configFile << fileContents << std::endl;
        configFile.close();
        if (configFile.fail()) {
            SPDLOG_INFO("Closing configFile failed");
            return false;
        } else {
            SPDLOG_INFO("Closing configFile succeed");
        }

        return true;
    }

protected:
    std::string tokenizerConfigFilePath;
    std::string jinjaConfigFilePath;
    void SetUp() {
        py::initialize_interpreter();
        TestWithTempDir::SetUp();
        tokenizerConfigFilePath = directoryPath + "/tokenizer_config.json";
        jinjaConfigFilePath = directoryPath + "/template.jinja";
    }
    void TearDown() { 
        TestWithTempDir::TearDown();
        py::finalize_interpreter(); 
    }

public:
    bool CreateTokenizerConfig(std::string& fileContents) {
        return CreateConfig(fileContents, tokenizerConfigFilePath);
    }
    bool CreateJinjaConfig(std::string& fileContents) {
        return CreateConfig(fileContents, jinjaConfigFilePath);
    }
};

TEST_F(LLMChatTemplateTest, ChatTemplateEmptyBody) {
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length > 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = "";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), false);
    std::string errorOutput = "Expecting value: line 1 column 1 (char 0)";
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateEmptyMessage) {
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length > 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";
    std::string errorOutput = "list object has no element 0";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateDefault) {
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length > 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "messages": [{ "content": "How can I help you?" }]
        }
    )";
    std::string expectedOutput = "How can I help you?";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateMultiMessage) {
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length > 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "messages": [{ "content": "How can I help you?" }, { "content": "2How can I help you?" }]
        }
    )";
    std::string errorOutput = "This servable accepts only single message requests";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateComplexMessage) {
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length > 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "hello";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateJinjaUppercase) {
    std::string jinjaTemplate = R"( {{ "Hi, " + messages[0]['content'] | upper }} )";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = " Hi, HELLO ";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateJinjaException) {
    std::string jinjaTemplate = R"( {{ "Hi, " + messages[3]['content'] | upper }} )";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string errorOutput = "list object has no element 3";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerDefault) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "hello";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerException) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "Error: Chat template not loaded correctly, so it cannot be applied";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerUpperCase) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    "chat_template": "{{ \"Hi, \" + messages[0]['content'] | upper }}"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "Hi, HELLO";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerTemplateException) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    "chat_template": "{{ \"Hi, \" + messages[3]['content'] | upper }}"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "list object has no element 3";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerTemplateBadVariable) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    "chat_template": {}
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedError = "Error: Chat template not loaded correctly, so it cannot be applied";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedError);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTwoConfigs) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    "chat_template": "{{ \"Hi, \" + messages[0]['content'] | lower }}"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::string jinjaTemplate = R"( {{ "Hi, " + messages[0]['content'] | upper }} )";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);

    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<LLMNodeResources>();
    nodeResources->modelsPath = directoryPath;
    LLMNodeResources::loadTextProcessor(nodeResources, nodeResources->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = " Hi, HELLO ";
    ASSERT_EQ(applyChatTemplate(nodeResources->textProcessor, nodeResources->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

std::string configTemplate = R"(
        {
            "model_config_list": [],
            "mediapipe_config_list": [
            {
                "name":"llmDummyKFS",
                "graph_path":"GRAPH_PATTERN"
            }
            ]
        }
    )";

std::string graphTemplate = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node {
            name: "llmNode1"
            calculator: "HttpLLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
            }
            node_options: {
                [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {
                models_path: "MODELS_PATTERN",
                plugin_config: "{\"INFERENCE_PRECISION_HINT\":\"f32\"}"
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
    })";

class LLMChatTemplateHttpTest : public TestWithTempDir {
private:
    bool CreateConfig(std::string& fileContents, std::string& filePath) {
        std::ofstream configFile{filePath};
        if (!configFile.is_open()) {
            std::cout << "Failed to open " << filePath << std::endl;
            return false;
        }
        SPDLOG_INFO("Creating config file: {}\n with content:\n{}", filePath, fileContents);
        configFile << fileContents << std::endl;
        configFile.close();
        if (configFile.fail()) {
            SPDLOG_INFO("Closing configFile failed");
            return false;
        } else {
            SPDLOG_INFO("Closing configFile succeed");
        }

        return true;
    }

protected:
    static std::unique_ptr<std::thread> t;

    const std::string GRAPH_PATTERN = "GRAPH_PATTERN";
    const std::string WORKSPACE_PATTERN = "MODELS_PATTERN";
    const std::string ONE_MODEL_PATH = "/ovms/llm_testing/facebook/opt-125m";

    std::string tokenizerConfigFilePath;
    std::string jinjaConfigFilePath;
    std::string ovmsConfigFilePath;
    std::string graphConfigFilePath;

    std::string GetFileNameFromPath(const std::string& parentDir, const std::string& fullPath) {
        std::string fileName = fullPath;
        fileName.replace(fileName.find(parentDir), std::string(parentDir).size(), "");
        return fileName;
    }

    bool CreateConfigFile(const std::string& graphPath){
        std::string configContents = configTemplate;
        configContents.replace(configContents.find(GRAPH_PATTERN), std::string(GRAPH_PATTERN).size(), graphPath);
        return CreateConfig(configContents, ovmsConfigFilePath);
    }

    bool CreatePipelineGraph(const std::string& workspacePath) {
        std::string configContents = graphTemplate;
        configContents.replace(configContents.find(WORKSPACE_PATTERN), std::string(WORKSPACE_PATTERN).size(), workspacePath);
        return CreateConfig(configContents, graphConfigFilePath);
    }

    void CreateSymbolicLinks() {
        for (const auto& entry : fs::directory_iterator(ONE_MODEL_PATH)) {
            std::filesystem::path outfilename = entry.path();
            std::string outfilename_str = outfilename.string();
            std::string fileName = GetFileNameFromPath(ONE_MODEL_PATH, outfilename_str);
            SPDLOG_INFO("Filename to link {}\n", fileName);
            std::string symlinkPath = ovms::FileSystem::joinPath({directoryPath, fileName});
            SPDLOG_INFO("Creating symlink from: {}\n to:\n{}", outfilename_str, symlinkPath);
            fs::create_symlink(outfilename_str, symlinkPath);
        }
    }

public:
    bool CreateTokenizerConfig(std::string& fileContents) {
        return CreateConfig(fileContents, tokenizerConfigFilePath);
    }
    bool CreateJinjaConfig(std::string& fileContents) {
        return CreateConfig(fileContents, jinjaConfigFilePath);
    }

    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v3/chat/completions";
    const std::string endpointCompletions = "/v3/completions";
    MockedServerRequestInterface writer;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUp() {
        TestWithTempDir::SetUp();
        tokenizerConfigFilePath = directoryPath + "/tokenizer_config.json";
        jinjaConfigFilePath = directoryPath + "/template.jinja";
        ovmsConfigFilePath = directoryPath + "/ovms_config.json";
        graphConfigFilePath = directoryPath + "/graph_config.pbtxt";

        CreateConfigFile(graphConfigFilePath);
        CreatePipelineGraph(directoryPath);
        CreateSymbolicLinks();

        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, ovmsConfigFilePath.c_str());
        auto start = std::chrono::high_resolution_clock::now();
        const int numberOfRetries = 5;
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < numberOfRetries)) {
        }
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
    }

    void TearDown() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        TestWithTempDir::TearDown();
    }
};
std::unique_ptr<std::thread> LLMChatTemplateHttpTest::t;

TEST_F(LLMChatTemplateHttpTest, inferDefaultChatCompletionsUnary) {
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

class LLMJinjaChatTemplateHttpTest : public LLMChatTemplateHttpTest {
    void SetUp() {
        std::string jinjaTemplate = R"( {{ "JINJA:" + messages[0]['content'] | upper }} )";
        CreateJinjaConfig(jinjaTemplate);
        LLMChatTemplateHttpTest::SetUp();
    }
};

TEST_F(LLMJinjaChatTemplateHttpTest, inferDefaultChatCompletionsUnary) {
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