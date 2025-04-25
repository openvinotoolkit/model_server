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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)
#include <rapidjson/error/en.h>

#include "../../filesystem.hpp"
#include "../../http_payload.hpp"
#include "../../http_rest_api_handler.hpp"
#include "../../httpservermodule.hpp"
#include "../../llm/language_model/continuous_batching/servable.hpp"
#include "../../llm/language_model/continuous_batching/servable_initializer.hpp"
#include "../../llm/py_jinja_template_processor.hpp"
#include "../../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../../server.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_runner.h"
#pragma GCC diagnostic pop

#include "../test_http_utils.hpp"
#include "../test_utils.hpp"

using namespace ovms;

class LLMChatTemplateTest : public TestWithTempDir {
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
        return createConfigFileWithContent(fileContents, tokenizerConfigFilePath);
    }
    bool CreateJinjaConfig(std::string& fileContents) {
        return createConfigFileWithContent(fileContents, jinjaConfigFilePath);
    }
};

TEST_F(LLMChatTemplateTest, ChatTemplateEmptyBody) {
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = "";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
    std::string errorOutput = "Expecting value: line 1 column 1 (char 0)";
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateEmptyMessage) {
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";
    std::string errorOutput = "This servable accepts only single message requests";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateMessageWithEmptyObject) {
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{}]
        }
    )";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, "");
}

TEST_F(LLMChatTemplateTest, ChatTemplateDefault) {
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "messages": [{ "content": "How can I help you?" }]
        }
    )";
    std::string expectedOutput = "How can I help you?";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateMultiMessage) {
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "messages": [{ "content": "How can I help you?" }, { "content": "2How can I help you?" }]
        }
    )";
    std::string errorOutput = "This servable accepts only single message requests";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateComplexMessage) {
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    // default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "hello";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateJinjaUppercase) {
    std::string jinjaTemplate = R"( {{ "Hi, " + messages[0]['content'] | upper }} )";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = " Hi, HELLO ";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateJinjaException) {
    std::string jinjaTemplate = R"( {{ "Hi, " + messages[3]['content'] | upper }} )";
    ASSERT_EQ(CreateJinjaConfig(jinjaTemplate), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string errorOutput = "list object has no element 3";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, errorOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerDefault) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "hello";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerBosNull) {
    std::string tokenizerJson = R"({
    "bos_token": null,
    "eos_token": "</s>"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "hello";
    // Expect no issues with chat template since non string bos token is ignored
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerBosDict) {
    std::string tokenizerJson = R"({
    "bos_token": {"bos" : "INVALID"},
    "eos_token": "</s>"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedError = "Error: Chat template not loaded correctly, so it cannot be applied";
    // Expect no issues with chat template since non string bos token is ignored
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedError);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerEosNull) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": null
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "hello";
    // Expect no issues with chat template since non string eos token is ignored
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerException) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "Error: Chat template not loaded correctly, so it cannot be applied";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerUpperCase) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    "chat_template": "{{ \"Hi, \" + messages[0]['content'] | upper }}"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "Hi, HELLO";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerTemplateException) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    "chat_template": "{{ \"Hi, \" + messages[3]['content'] | upper }}"
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = "list object has no element 3";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

TEST_F(LLMChatTemplateTest, ChatTemplateTokenizerTemplateBadVariable) {
    std::string tokenizerJson = R"({
    "bos_token": "</s>",
    "eos_token": "</s>",
    "chat_template": {}
    })";
    ASSERT_EQ(CreateTokenizerConfig(tokenizerJson), true);
    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedError = "Error: Chat template not loaded correctly, so it cannot be applied";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), false);
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

    std::shared_ptr<GenAiServable> servable = std::make_shared<ContinuousBatchingServable>();

    servable->getProperties()->modelsPath = directoryPath;
    GenAiServableInitializer::loadTemplateProcessor(servable->getProperties(), servable->getProperties()->modelsPath);

    std::string finalPrompt = "";
    std::string payloadBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": [{"role": "user", "content": "hello"}]
        }
    )";
    std::string expectedOutput = " Hi, HELLO ";
    ASSERT_EQ(PyJinjaTemplateProcessor::applyChatTemplate(servable->getProperties()->templateProcessor, servable->getProperties()->modelsPath, payloadBody, finalPrompt), true);
    ASSERT_EQ(finalPrompt, expectedOutput);
}

std::string configTemplate = R"(
        {
            "model_config_list": [],
            "mediapipe_config_list": [
            {
                "name":"lm_cb_regular",
                "graph_path":"<GRAPH_PATTERN>"
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
                models_path: "<MODELS_PATTERN>",
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

class CleanupFilesGuard {
    const std::string& pathToClean;

public:
    CleanupFilesGuard(const std::string& pathToClean) :
        pathToClean(pathToClean) {}
    ~CleanupFilesGuard() {
        std::filesystem::remove_all(pathToClean);
    }
};

const std::string GRAPH_PATTERN = "<GRAPH_PATTERN>";
const std::string WORKSPACE_PATTERN = "<MODELS_PATTERN>";
const std::string MODEL_PATH = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m");

class LLMChatTemplateHttpTest : public TestWithTempDir {
protected:
    static std::unique_ptr<std::thread> t;

    std::string tokenizerConfigFilePath;
    std::string jinjaConfigFilePath;
    std::string ovmsConfigFilePath;
    std::string graphConfigFilePath;

    static std::string GetFileNameFromPath(const std::string& parentDir, const std::string& fullPath) {
        std::string fileName = fullPath;
        fileName.replace(fileName.find(parentDir), std::string(parentDir).size(), "");
        return fileName;
    }

    static bool CreateConfigFile(const std::string& graphPath, const std::string& configFilePath) {
        std::string configContents = configTemplate;
        configContents.replace(configContents.find(GRAPH_PATTERN), std::string(GRAPH_PATTERN).size(), graphPath);
        return createConfigFileWithContent(configContents, configFilePath);
    }

    static bool CreatePipelineGraph(const std::string& workspacePath, const std::string& graphConfigFilePath) {
        std::string configContents = graphTemplate;
        configContents.replace(configContents.find(WORKSPACE_PATTERN), std::string(WORKSPACE_PATTERN).size(), workspacePath);
        return createConfigFileWithContent(configContents, graphConfigFilePath);
    }

    static void CreateSymbolicLinks(const std::string& toDirectory) {
        for (const auto& entry : fs::directory_iterator(MODEL_PATH)) {
            std::filesystem::path outFilename = entry.path();
            std::string outFilenameStr = outFilename.string();
            std::string fileName = GetFileNameFromPath(MODEL_PATH, outFilenameStr);
            SPDLOG_INFO("Filename to link {}\n", fileName);
            std::string symlinkPath = ovms::FileSystem::joinPath({toDirectory, fileName});
            SPDLOG_INFO("Creating symlink from: {}\n to:\n{}", outFilenameStr, symlinkPath);
            fs::create_symlink(outFilenameStr, symlinkPath);
            // TODO: Symlinks are never removed
        }
    }

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v3/chat/completions";
    const std::string endpointCompletions = "/v3/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
        TestWithTempDir::SetUp();
        tokenizerConfigFilePath = directoryPath + "/tokenizer_config.json";
        jinjaConfigFilePath = directoryPath + "/template.jinja";
        ovmsConfigFilePath = directoryPath + "/ovms_config.json";
        graphConfigFilePath = directoryPath + "/graph_config.pbtxt";

        CreateConfigFile(graphConfigFilePath, ovmsConfigFilePath);
        CreatePipelineGraph(directoryPath, graphConfigFilePath);
        CreateSymbolicLinks(directoryPath);

        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, ovmsConfigFilePath.c_str());
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

std::string fullResponse;

// static void ConcatenateResponse(const std::string& partial) {
//     fullResponse += partial;
// }

class LLMJinjaChatTemplateHttpTest : public LLMChatTemplateHttpTest {
public:
    static std::unique_ptr<std::thread> t;

protected:
    static const std::string getDirectoryPath() {
        const std::string directoryName = "LLMJinjaChatTemplateHttpTest";
        std::string directoryPath = std::string{"/tmp/"} + directoryName;
        return getGenericFullPathForTmp(directoryPath);
    }
    static void SetUpTestSuite() {
        const auto directoryPath = getDirectoryPath();
        std::filesystem::remove_all(directoryPath);
        std::filesystem::create_directories(directoryPath);

        const std::string ovmsConfigFilePath = directoryPath + "/ovms_config.json";
        CreateConfigFile(
            directoryPath + "/graph_config.pbtxt",
            ovmsConfigFilePath);
        CreatePipelineGraph(
            directoryPath,
            directoryPath + "/graph_config.pbtxt");
        CreateSymbolicLinks(directoryPath);
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, ovmsConfigFilePath.c_str());
    }

    void SetUp() override {
        writer = std::make_shared<MockedServerRequestInterface>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
    }

    void TearDown() override {
        handler.reset();
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        std::filesystem::remove_all(getDirectoryPath());
    }
};

std::unique_ptr<std::thread> LLMJinjaChatTemplateHttpTest::t;

TEST_F(LLMJinjaChatTemplateHttpTest, inferChatCompletionsUnary) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    // Assertion split in two parts to avoid timestamp mismatch
    // const size_t timestampLength = 10;
    std::string expectedResponsePart1 = R"({"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"\nOpenVINO is","role":"assistant"}}],"created":)";
    std::string expectedResponsePart2 = R"(,"model":"lm_cb_regular","object":"chat.completion"})";
    // TODO: New output ASSERT_EQ(response.compare(0, expectedResponsePart1.length(), expectedResponsePart1), 0);
    // TODO: New output ASSERT_EQ(response.compare(expectedResponsePart1.length() + timestampLength, expectedResponsePart2.length(), expectedResponsePart2), 0);
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferCompletionsUnary) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    // Assertion split in two parts to avoid timestamp mismatch
    // const size_t timestampLength = 10;
    std::string expectedResponsePart1 = R"({"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"text":"\n\nThe first thing"}],"created":)";
    std::string expectedResponsePart2 = R"(,"model":"lm_cb_regular","object":"text_completion"})";
    // TODO: New output ASSERT_EQ(response.compare(0, expectedResponsePart1.length(), expectedResponsePart1), 0);
    // TODO: New output ASSERT_EQ(response.compare(expectedResponsePart1.length() + timestampLength, expectedResponsePart2.length(), expectedResponsePart2), 0);
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferChatCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": true,
            "seed" : 1,
            "max_tokens": 6,
            "prompt": "?"
        }
    )";

    // TODO: New output EXPECT_CALL(writer, PartialReplyEnd()).Times(1);
    /* TODO: New output EXPECT_CALL(writer, PartialReply(::testing::_))
        .WillRepeatedly([](std::string response) {
            rapidjson::Document responseJson;
            const int dataHeaderSize = 6;
            std::string jsonResponse = response.substr(dataHeaderSize);
            rapidjson::ParseResult ok = responseJson.Parse(jsonResponse.c_str());
            if (response.find("[DONE]") == std::string::npos) {
                ASSERT_EQ(ok.Code(), 0);
                auto m = responseJson.FindMember("choices");
                ASSERT_NE(m, responseJson.MemberEnd());
                auto& choices = m->value.GetArray()[0];
                auto modelOutput = choices.GetObject()["text"].GetString();
                ConcatenateResponse(modelOutput);
            }
        });
    */
    // TODO: New output EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);
    // TODO: New output EXPECT_CALL(writer, IsDisconnected()).Times(7);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");

    // TODO: New output ASSERT_EQ(fullResponse, "\n\nThe first thing ");
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
            "stream": true,
            "seed" : 1,
            "max_tokens": 6,
            "prompt": "?"
        }
    )";

    // TODO: New output EXPECT_CALL(writer, PartialReplyEnd()).Times(1);
    /* TODO: New output EXPECT_CALL(writer, PartialReply(::testing::_))
        .WillRepeatedly([](std::string response) {
            rapidjson::Document responseJson;
            const int dataHeaderSize = 6;
            std::string jsonResponse = response.substr(dataHeaderSize);
            rapidjson::ParseResult ok = responseJson.Parse(jsonResponse.c_str());
            if (response.find("[DONE]") == std::string::npos) {
                ASSERT_EQ(ok.Code(), 0);
                auto m = responseJson.FindMember("choices");
                ASSERT_NE(m, responseJson.MemberEnd());
                auto& choices = m->value.GetArray()[0];
                auto modelOutput = choices.GetObject()["text"].GetString();
                ConcatenateResponse(modelOutput);
            }
        });
    */
    // TODO: New output EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);
    // TODO: New output EXPECT_CALL(writer, IsDisconnected()).Times(7);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");

    // ASSERT_EQ(fullResponse, "\n\nThe first thing ");
}

TEST_F(LLMJinjaChatTemplateHttpTest, inferDefaultChatCompletionsUnary) {
    std::unique_ptr<CleanupFilesGuard> cleanupGuard = std::make_unique<CleanupFilesGuard>(directoryPath);
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    // Assertion split in two parts to avoid timestamp mismatch
    // const size_t timestampLength = 10;
    std::string expectedResponsePart1 = R"({"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"\nOpenVINO is","role":"assistant"}}],"created":)";
    std::string expectedResponsePart2 = R"(,"model":"lm_cb_regular","object":"chat.completion"})";
    // TODO: New output ASSERT_EQ(response.compare(0, expectedResponsePart1.length(), expectedResponsePart1), 0);
    // TODO: New output ASSERT_EQ(response.compare(expectedResponsePart1.length() + timestampLength, expectedResponsePart2.length(), expectedResponsePart2), 0);
}
