//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "../../http_rest_api_handler.hpp"
#include "../../servablemanagermodule.hpp"
#include "../../server.hpp"
#include "rapidjson/document.h"
#include "../test_http_utils.hpp"
#include "../test_utils.hpp"
#include "../platform_utils.hpp"
#include "../constructor_enabled_model_manager.hpp"

using namespace ovms;

class Text2SpeechHttpTest : public V3HttpTest {
protected:
    std::string modelName = "text2speech";
    std::string endpoint = "/v3/audio/speech";
    static std::unique_ptr<std::thread> t;

public:
    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/audio/config.json");
        SetUpSuite(port, configPath, t);
    }

    void SetUp() {
        V3HttpTest::SetUp();
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> Text2SpeechHttpTest::t;

TEST_F(Text2SpeechHttpTest, simplePositive) {
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": "The quick brown fox jumped over the lazy dog."
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_F(Text2SpeechHttpTest, positiveWithVoice) {
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": "The quick brown fox jumped over the lazy dog.",
            "voice": "speaker1"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_F(Text2SpeechHttpTest, nonExisitingVoiceRequested) {
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "input": "The quick brown fox jumped over the lazy dog.",
            "voice": "speaker_non_exist"
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

class Text2SpeechConfigTest : public ::testing::Test {};

TEST_F(Text2SpeechConfigTest, simplePositive) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    name: "ttsNode1"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/microsoft/speecht5_tts"
            plugin_config: '{"NUM_STREAMS": "1" }',
            target_device: "CPU"
            voices: [
            {
                name: "speaker1",
                path: "/ovms/src/test/audio/speaker.bin",
            }
        ]
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
}

TEST_F(Text2SpeechConfigTest, NodeNameMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/microsoft/speecht5_tts"
            target_device: "CPU"
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_MISSING_NAME);
}

TEST_F(Text2SpeechConfigTest, SidePacketMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    name: "ttsNode1"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/microsoft/speecht5_tts"
            target_device: "CPU"
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(Text2SpeechConfigTest, MissingModelsPath) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    name: "ttsNode1"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            target_device: "CPU"
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

TEST_F(Text2SpeechConfigTest, InvalidPluginConfig) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    name: "ttsNode1"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/microsoft/speecht5_tts"
            plugin_config: 'INVALID',
            target_device: "CPU"
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

TEST_F(Text2SpeechConfigTest, NonExistingVoicePath) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    name: "ttsNode1"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/microsoft/speecht5_tts"
            plugin_config: '{"NUM_STREAMS": "1" }',
            target_device: "CPU"
            voices: [
            {
                name: "speaker1",
                path: "/ovms/src/test/audio/non_existing.bin",
            }
        ]
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

TEST_F(Text2SpeechConfigTest, VoiceMissingPath) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    name: "ttsNode1"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/microsoft/speecht5_tts"
            plugin_config: '{"NUM_STREAMS": "1" }',
            target_device: "CPU"
            voices: [
            {
                name: "speaker1"
            }
        ]
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

TEST_F(Text2SpeechConfigTest, VoiceInvalidFile) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"

    node {
    name: "ttsNode1"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    calculator: "T2sCalculator"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/microsoft/speecht5_tts"
            plugin_config: '{"NUM_STREAMS": "1" }',
            target_device: "CPU"
            voices: [
            {
                name: "speaker1",
                path: "/ovms/src/test/audio/invalid_speaker.bin",
            }
        ]
        }
        }
    }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}