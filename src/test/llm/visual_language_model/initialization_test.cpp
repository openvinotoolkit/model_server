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
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../../../llm/servable_initializer.hpp"
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
    inline static constexpr const char* modalityModelFiles[] = {
        "openvino_vision_embeddings_model.bin",
        "openvino_audio_embeddings_model.xml",
        "openvino_audio_encoder_model.xml",
    };

    void addModelFile(const std::string& filename) {
        std::ofstream file(std::filesystem::path(directoryPath) / filename);
        ASSERT_TRUE(file.is_open());
    }

    void removeModelFile(const std::string& filename) {
        ASSERT_TRUE(std::filesystem::remove(std::filesystem::path(directoryPath) / filename));
    }

    void addTextEmbeddingsModel() {
        addModelFile("openvino_text_embeddings_model.xml");
    }

    void removeTextEmbeddingsModel() {
        removeModelFile("openvino_text_embeddings_model.xml");
    }

    void addVisionModelFiles() {
        addTextEmbeddingsModel();
        addModelFile("openvino_vision_embeddings_model.bin");
    }

    void addAudioDecoderModel() {
        addModelFile("openvino_audio_decoder_model.xml");
    }

    void addAudioEncoderModel() {
        addModelFile("openvino_audio_encoder_model.xml");
    }

    void addTalkerModel() {
        addModelFile("openvino_talker_model.xml");
    }

    void removeTalkerModel() {
        removeModelFile("openvino_talker_model.xml");
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
        EXPECT_EQ(selectPipeline(requested, selected), StatusCode::INTERNAL_ERROR);
    }
    addTextEmbeddingsModel();
    for (const auto requested : {mediapipe::LLMCalculatorOptions::VLM, mediapipe::LLMCalculatorOptions::VLM_CB}) {
        PipelineType selected;
        EXPECT_EQ(selectPipeline(requested, selected), StatusCode::INTERNAL_ERROR);
    }
}

TEST_F(PipelineTypeSelectionTest, OmniRejectsEmptyModelDirectory) {
    PipelineType selected;
    EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::INTERNAL_ERROR);
}

TEST_F(PipelineTypeSelectionTest, OmniWithoutSpeechOutputRejectsMultimodalModel) {
    addVisionModelFiles();
    PipelineType selected;
    EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::INTERNAL_ERROR);
}

TEST_F(PipelineTypeSelectionTest, ExplicitPipelinesAcceptedForOmniCapableModel) {
    addVisionModelFiles();
    addTalkerModel();
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
}

TEST_F(PipelineTypeSelectionTest, AudioDecoderEnablesOmniSelection) {
    addAudioDecoderModel();
    PipelineType selected;
    EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::INTERNAL_ERROR);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM_CB);
    addTextEmbeddingsModel();
    addAudioEncoderModel();
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::OMNI);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::OMNI);
}

TEST_F(PipelineTypeSelectionTest, MultimodalDetectionRequiresTextAndAModality) {
    for (const std::string filename : modalityModelFiles) {
        SCOPED_TRACE(filename);
        addModelFile(filename);
        PipelineType selected;
        EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::VLM, selected), StatusCode::INTERNAL_ERROR);
        addTextEmbeddingsModel();
        EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::VLM, selected), StatusCode::OK);
        EXPECT_EQ(selected, PipelineType::VLM);
        removeModelFile(filename);
        removeTextEmbeddingsModel();
    }
}

TEST_F(PipelineTypeSelectionTest, AutoSelectionAndOmniWithTalker) {
    PipelineType selected;
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM_CB);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected, "NPU"), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM);
    addTalkerModel();
    EXPECT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::OMNI, selected), StatusCode::INTERNAL_ERROR);
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::LM_CB);
    addVisionModelFiles();
    ASSERT_EQ(selectPipeline(mediapipe::LLMCalculatorOptions::AUTO, selected), StatusCode::OK);
    EXPECT_EQ(selected, PipelineType::OMNI);
    removeTalkerModel();
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
