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

#include <gtest/gtest.h>
#include <openvino/genai/chat_history.hpp>

#include "../../../llm/io_processing/input_processors/video_frames_processor.hpp"
#include "../../../llm/io_processing/input_request.hpp"

using namespace ovms;

// Helpers ----------------------------------------------------------------

static InputRequest makeChatRequest(ov::genai::ChatHistory chatHistory) {
    InputRequest req;
    req.input = std::move(chatHistory);
    return req;
}

// Minimal 1x1 PNG reused as a decoded video frame.
static const std::string FRAME_BASE64 =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
    "AAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg==";

// A 2x2 PNG, used to verify frames of mismatching resolution are rejected.
static const std::string FRAME_BASE64_2X2 =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91Jpz"
    "AAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

// Tests ------------------------------------------------------------------

TEST(VideoFramesProcessorTest, NoVideoInTextOnlyMessage) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hello, world!"}});

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(req.inputVideos.empty());
    const auto& resultHistory = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(resultHistory[0]["content"].as_string().value_or(""), "Hello, world!");
}

TEST(VideoFramesProcessorTest, InjectionGuardBlocksPreexistingTag) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "<ov_genai_video_0>\nsome text"}});

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(VideoFramesProcessorTest, InjectionGuardBlocksTagInArrayTextPart) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"look at this <ov_genai_video_0> tag"}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(VideoFramesProcessorTest, SkipsMessagesWithNonArrayContent) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "system"}, {"content", "You are helpful."}});

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(req.inputVideos.empty());
}

TEST(VideoFramesProcessorTest, SingleFrameVideoDecodedAndTagged) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"describe the video"},)"
        R"( {"type":"video_url","video_url":{"url":[")" + FRAME_BASE64 + R"("]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputVideos.size(), 1u);
    // Video tensor must be [N, H, W, C] with N == number of frames.
    const auto shape = req.inputVideos[0].get_shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[0], 1u);
    // The video_url part was rewritten into a text part carrying the video tag.
    const auto& resultHistory = std::get<ov::genai::ChatHistory>(req.input);
    const auto content = resultHistory[0]["content"];
    ASSERT_TRUE(content.is_array());
    bool foundTag = false;
    for (size_t j = 0; j < content.size(); j++) {
        if (content[j]["text"].as_string().value_or("").find("<ov_genai_video_0>") != std::string::npos) {
            foundTag = true;
        }
    }
    EXPECT_TRUE(foundTag);
}

TEST(VideoFramesProcessorTest, MultiFrameVideoStacksFrames) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":[")" + FRAME_BASE64 + R"(",")" +
        FRAME_BASE64 + R"(",")" + FRAME_BASE64 + R"("]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputVideos.size(), 1u);
    EXPECT_EQ(req.inputVideos[0].get_shape()[0], 3u);
}

TEST(VideoFramesProcessorTest, EmptyFrameArrayRejected) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":[]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(VideoFramesProcessorTest, InvalidBase64FrameRejected) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":["data:image/png;base64,NOT_VALID!!!"]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(req.inputVideos.empty());
}

TEST(VideoFramesProcessorTest, MismatchingFrameResolutionRejected) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    // First frame is 1x1, second frame is 2x2: frames must share the same H, W, C.
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":[")" + FRAME_BASE64 + R"(",")" +
        FRAME_BASE64_2X2 + R"("]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(req.inputVideos.empty());
}
