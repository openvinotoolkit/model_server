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
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>

#include "../../../llm/servable_initializer.hpp"
#include "../../../logging.hpp"
#include "../../constructor_enabled_model_manager.hpp"
#include "../../platform_utils.hpp"
#include "../../test_utils.hpp"
#include "../../test_with_temp_dir.hpp"

using namespace ovms;

Status callDeterminePipelineType(PipelineType& pipelineType, const std::string& testPbtxt) {
    ::mediapipe::CalculatorGraphConfig config;
    ::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config);
    const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig = config.node(0);
    mediapipe::LLMCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    return determinePipelineType(pipelineType, nodeOptions, "");
}

// Initialization tests
class VLMServableInitializationTest : public ::testing::Test {};

class PipelineTypeSelectionTest : public TestWithTempDir {
protected:
    std::ostringstream logs;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> logSink;
    spdlog::level::level_enum originalLogLevel;

    void SetUp() override {
        TestWithTempDir::SetUp();
        originalLogLevel = modelmanager_logger->level();
        modelmanager_logger->set_level(spdlog::level::debug);
        logSink = std::make_shared<spdlog::sinks::ostream_sink_mt>(logs);
        modelmanager_logger->sinks().push_back(logSink);
    }

    void TearDown() override {
        auto& sinks = modelmanager_logger->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), logSink), sinks.end());
        modelmanager_logger->set_level(originalLogLevel);
        TestWithTempDir::TearDown();
    }

    void addModelFile(const std::string& filename) {
        std::ofstream file(std::filesystem::path(directoryPath) / filename);
        ASSERT_TRUE(file.is_open());
    }

    Status selectPipeline(mediapipe::LLMCalculatorOptions::PipelineType requested, PipelineType& selected, const std::string& device = "") {
        mediapipe::LLMCalculatorOptions options;
        options.set_models_path(directoryPath);
        options.set_pipeline_type(requested);
        options.set_device(device);
        return determinePipelineType(selected, options, "");
    }
};

TEST_F(PipelineTypeSelectionTest, VlmRequiresMultimodalCapabilities) {
    for (const auto requested : {mediapipe::LLMCalculatorOptions::VLM, mediapipe::LLMCalculatorOptions::VLM_CB}) {
        PipelineType selected;
        logs.str("");
        logs.clear();
        EXPECT_EQ(selectPipeline(requested, selected), StatusCode::INTERNAL_ERROR);
        EXPECT_NE(logs.str().find("Use an LM or LM_CB pipeline instead."), std::string::npos);
    }
}

TEST_F(PipelineTypeSelectionTest, OmniWithoutTalkerRecommendsLmForTextOnly) {
    PipelineType selected;
    EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::INTERNAL_ERROR);
    EXPECT_NE(logs.str().find("Use an LM or LM_CB pipeline instead."), std::string::npos);
}

TEST_F(PipelineTypeSelectionTest, OmniWithoutTalkerRecommendsVlmForMultimodal) {
    addModelFile("openvino_text_embeddings_model.xml");
    addModelFile("openvino_vision_embeddings_model.bin");
    PipelineType selected;
    EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::INTERNAL_ERROR);
    EXPECT_NE(logs.str().find("Use a VLM or VLM_CB pipeline instead."), std::string::npos);
}

TEST_F(PipelineTypeSelectionTest, ExplicitPipelinesAcceptedForOmniCapableModel) {
    addModelFile("openvino_text_embeddings_model.xml");
    addModelFile("openvino_vision_embeddings_model.bin");
    addModelFile("openvino_talker_model.xml");
    const std::pair<mediapipe::LLMCalculatorOptions::PipelineType, PipelineType> cases[] = {
        {mediapipe::LLMCalculatorOptions::LM, PipelineType::LM},
        {mediapipe::LLMCalculatorOptions::LM_CB, PipelineType::LM_CB},
        {mediapipe::LLMCalculatorOptions::VLM, PipelineType::VLM},
        {mediapipe::LLMCalculatorOptions::VLM_CB, PipelineType::VLM_CB},
        {mediapipe::LLMCalculatorOptions::OMNI, PipelineType::OMNI},
    };
    for (const auto& testCase : cases) {
        PipelineType selected;
        ASSERT_EQ(selectPipeline(testCase.first, selected), StatusCode::OK);
        EXPECT_EQ(selected, testCase.second);
    }
    EXPECT_NE(logs.str().find("talker=true, audio decoder=false, text embeddings=true, vision embeddings=true, audio embeddings=false, audio encoder=false"), std::string::npos);
}

TEST_F(PipelineTypeSelectionTest, AudioDecoderEnablesOmniSelection) {
    addModelFile("openvino_audio_decoder_model.xml");
    PipelineType selected;
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::OMNI);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM_CB);
    addModelFile("openvino_text_embeddings_model.xml");
    addModelFile("openvino_audio_encoder_model.xml");
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::OMNI);
    EXPECT_NE(logs.str().find("talker=false, audio decoder=true"), std::string::npos);
}

TEST_F(PipelineTypeSelectionTest, MultimodalDetectionRequiresTextAndAModality) {
    for (const std::string filename : {"openvino_vision_embeddings_model.bin", "openvino_audio_embeddings_model.xml", "openvino_audio_encoder_model.xml"}) {
        SCOPED_TRACE(filename);
        addModelFile(filename);
        PipelineType selected;
        EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::VLM, selected), StatusCode::INTERNAL_ERROR);
        addModelFile("openvino_text_embeddings_model.xml");
        EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::VLM, selected), StatusCode::OK);
        EXPECT_EQ(selected, PipelineType::VLM);
        std::filesystem::remove(std::filesystem::path(directoryPath) / filename);
        std::filesystem::remove(std::filesystem::path(directoryPath) / "openvino_text_embeddings_model.xml");
    }
}

TEST_F(PipelineTypeSelectionTest, AutoSelectionAndTalkerOnlyOmni) {
    PipelineType selected;
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM_CB);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected, "NPU"), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM);
    addModelFile("openvino_talker_model.xml");
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::OMNI);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM_CB);
    addModelFile("openvino_text_embeddings_model.xml");
    addModelFile("openvino_vision_embeddings_model.bin");
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::OMNI);
    std::filesystem::remove(std::filesystem::path(directoryPath) / "openvino_talker_model.xml");
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::VLM_CB);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected, "NPU"), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::VLM);
}

TEST_F(VLMServableInitializationTest, determinePipelineTypeDefault) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "VLMServable"
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
                models_path: "/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-llava-ov"
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
    adjustConfigForTargetPlatform(testPbtxt);
    PipelineType pipelineType;
    auto status = callDeterminePipelineType(pipelineType, testPbtxt);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(pipelineType, PipelineType::VLM_CB);
}

TEST_F(VLMServableInitializationTest, determinePipelineType_VLM_Specified) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "VLMServable"
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
                pipeline_type: VLM
                models_path: "/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-llava-ov"
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
    adjustConfigForTargetPlatform(testPbtxt);
    PipelineType pipelineType;
    auto status = callDeterminePipelineType(pipelineType, testPbtxt);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(pipelineType, PipelineType::VLM);
}

TEST_F(VLMServableInitializationTest, determinePipelineType_LM_CB_Specified) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "VLMServable"
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
                pipeline_type: LM_CB
                models_path: "/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-llava-ov"
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
    adjustConfigForTargetPlatform(testPbtxt);
    PipelineType pipelineType;
    auto status = callDeterminePipelineType(pipelineType, testPbtxt);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(pipelineType, PipelineType::LM_CB);
}

// Fast Draft (draft_models_path without draft_eagle3_mode) is not supported for VLM pipelines.
TEST_F(VLMServableInitializationTest, fastDraftNotSupportedForVlm) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "VLMServable"
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
                models_path: "/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-llava-ov"
                draft_models_path: "/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-llava-ov"
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
    adjustConfigForTargetPlatform(testPbtxt);
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
}
