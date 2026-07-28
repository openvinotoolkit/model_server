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
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openvino/genai/chat_history.hpp>

#pragma warning(push)
#pragma warning(disable : 6001)
#include "absl/strings/escaping.h"
#pragma warning(pop)

#include "../../../llm/io_processing/input_processors/audio_decoding_processor.hpp"
#include "../../../llm/io_processing/input_request.hpp"
#include "../../wav_test_utils.hpp"

using namespace ovms;
using test_utils::buildWavBuffer;

// Helpers ----------------------------------------------------------------

static InputRequest makeChatRequest(ov::genai::ChatHistory chatHistory) {
    InputRequest req;
    req.input = std::move(chatHistory);
    return req;
}

// Encodes binary data to base64.
static std::string toBase64(const std::string& raw) {
    std::string encoded;
    absl::Base64Escape(raw, &encoded);
    return encoded;
}

// Builds a ChatHistory message with a content array containing an input_audio part.
static InputRequest makeAudioRequest(const std::string& base64Data, const std::string& format = "wav") {
    std::string contentJson = R"([{"type":"input_audio","input_audio":{"data":")" +
                              base64Data + R"(","format":")" + format + R"("}}])";
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(contentJson);
    history.push_back(msg);
    return makeChatRequest(history);
}

// Tests ------------------------------------------------------------------

TEST(AudioDecodingProcessorTest, TextOnlyMessagePassedThrough) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hello, world!"}});

    InputRequest req = makeChatRequest(history);
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(req.inputAudios.empty());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(result[0]["content"].as_string().value_or(""), "Hello, world!");
}

TEST(AudioDecodingProcessorTest, SkipsMessagesWithNonArrayContent) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "system"}, {"content", "You are helpful."}});
    history.push_back({{"role", "user"}, {"content", "What is OpenVINO?"}});

    InputRequest req = makeChatRequest(history);
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(req.inputAudios.empty());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0]["content"].as_string().value_or(""), "You are helpful.");
    EXPECT_EQ(result[1]["content"].as_string().value_or(""), "What is OpenVINO?");
}

TEST(AudioDecodingProcessorTest, ValidWavAudioDecodedSuccessfully) {
    const uint32_t numSamples = 160;
    std::string wavData = buildWavBuffer(numSamples);
    std::string base64 = toBase64(wavData);

    InputRequest req = makeAudioRequest(base64, "wav");
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputAudios.size(), 1u);
    // WAV at 16kHz with no resampling should produce numSamples float values
    EXPECT_EQ(req.inputAudios[0].get_shape()[0], numSamples);
    EXPECT_EQ(req.inputAudios[0].get_element_type(), ov::element::f32);
}

TEST(AudioDecodingProcessorTest, AudioPartRemovedFromContentArray) {
    const std::string wavData = buildWavBuffer(16);
    const std::string base64 = toBase64(wavData);

    // Content: [text, input_audio, text]
    std::string contentJson = R"([{"type":"text","text":"before"},)"
                              R"({"type":"input_audio","input_audio":{"data":")" +
                              base64 + R"(","format":"wav"}},)"
                                       R"({"type":"text","text":"after"}])";
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(contentJson);
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputAudios.size(), 1u);

    // Content array should now only contain the two text parts
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    const auto content = result[0]["content"];
    ASSERT_TRUE(content.is_array());
    ASSERT_EQ(content.size(), 2u);
    EXPECT_EQ(content[0]["type"].as_string().value_or(""), "text");
    EXPECT_EQ(content[0]["text"].as_string().value_or(""), "before");
    EXPECT_EQ(content[1]["type"].as_string().value_or(""), "text");
    EXPECT_EQ(content[1]["text"].as_string().value_or(""), "after");
}

TEST(AudioDecodingProcessorTest, EmptyDataFieldRejected) {
    InputRequest req = makeAudioRequest("", "wav");
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "input_audio data field is empty");
}

TEST(AudioDecodingProcessorTest, InvalidBase64Rejected) {
    InputRequest req = makeAudioRequest("NOT_VALID_BASE64!!!", "wav");
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Invalid base64 string in input_audio data");
}

TEST(AudioDecodingProcessorTest, CorruptWavDataRejected) {
    // Valid base64 but not a valid WAV file
    std::string garbage = "this is not a wav file at all";
    std::string base64 = toBase64(garbage);

    InputRequest req = makeAudioRequest(base64, "wav");
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(status.message().find("Audio decoding failed") != std::string::npos);
}

TEST(AudioDecodingProcessorTest, MultipleAudioPartsProduceMultipleTensors) {
    const std::string wavData1 = buildWavBuffer(16);
    const std::string wavData2 = buildWavBuffer(32);
    const std::string base64_1 = toBase64(wavData1);
    const std::string base64_2 = toBase64(wavData2);

    std::string contentJson = R"([{"type":"input_audio","input_audio":{"data":")" +
                              base64_1 + R"(","format":"wav"}},)"
                                         R"({"type":"input_audio","input_audio":{"data":")" +
                              base64_2 + R"(","format":"wav"}}])";
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(contentJson);
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputAudios.size(), 2u);
    EXPECT_EQ(req.inputAudios[0].get_shape()[0], 16u);
    EXPECT_EQ(req.inputAudios[1].get_shape()[0], 32u);
}

TEST(AudioDecodingProcessorTest, AudioAcrossMultipleMessagesCollected) {
    const std::string wavData1 = buildWavBuffer(16);
    const std::string wavData2 = buildWavBuffer(24);
    const std::string base64_1 = toBase64(wavData1);
    const std::string base64_2 = toBase64(wavData2);

    ov::genai::ChatHistory history;

    // First message with audio
    ov::AnyMap msg1;
    msg1["role"] = std::string("user");
    msg1["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"input_audio","input_audio":{"data":")" + base64_1 + R"(","format":"wav"}}])");
    history.push_back(msg1);

    // Second message with audio
    ov::AnyMap msg2;
    msg2["role"] = std::string("user");
    msg2["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"input_audio","input_audio":{"data":")" + base64_2 + R"(","format":"wav"}}])");
    history.push_back(msg2);

    InputRequest req = makeChatRequest(history);
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputAudios.size(), 2u);
    EXPECT_EQ(req.inputAudios[0].get_shape()[0], 16u);
    EXPECT_EQ(req.inputAudios[1].get_shape()[0], 24u);
}

TEST(AudioDecodingProcessorTest, NonChatHistoryInputRejected) {
    InputRequest req;
    req.input = std::string("raw prompt text");  // Not a ChatHistory

    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
}

TEST(AudioDecodingProcessorTest, OtherContentTypesPreserved) {
    // Content array with text + image_url + input_audio: only input_audio removed
    const std::string wavData = buildWavBuffer(16);
    const std::string base64 = toBase64(wavData);

    std::string contentJson = R"([{"type":"text","text":"describe"},)"
                              R"({"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}},)"
                              R"({"type":"input_audio","input_audio":{"data":")" +
                              base64 + R"(","format":"wav"}}])";
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(contentJson);
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    AudioDecodingProcessor processor;
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputAudios.size(), 1u);

    // Content array should keep text and image_url, remove input_audio
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    const auto content = result[0]["content"];
    ASSERT_TRUE(content.is_array());
    ASSERT_EQ(content.size(), 2u);
    EXPECT_EQ(content[0]["type"].as_string().value_or(""), "text");
    EXPECT_EQ(content[1]["type"].as_string().value_or(""), "image_url");
}
