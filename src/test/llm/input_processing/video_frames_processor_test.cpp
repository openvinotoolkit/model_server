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
#include "../../../llm/io_processing/video_utils.hpp"
#include "src/config.hpp"

using namespace ovms;

// Helpers ----------------------------------------------------------------

// Temporarily overrides the global per-request decoded-pixel budget so the video
// path's cross-frame accumulation can be exercised, restoring the previous
// configuration on destruction.
class ScopedImageDecodeBudget {
public:
    ScopedImageDecodeBudget(uint64_t maxImageDecodePixels, bool allowUnestimatableImageFormats) :
        savedServerSettings(ovms::Config::instance().getServerSettings()),
        savedModelsSettings(ovms::Config::instance().getModelSettings()) {
        ovms::ServerSettingsImpl serverSettings = savedServerSettings;
        serverSettings.maxImageDecodePixels = maxImageDecodePixels;
        serverSettings.allowUnestimatableImageFormats = allowUnestimatableImageFormats;
        ovms::ModelsSettingsImpl modelsSettings = savedModelsSettings;
        ovms::Config::instance().parse(&serverSettings, &modelsSettings);
    }
    ScopedImageDecodeBudget(const ScopedImageDecodeBudget&) = delete;
    ScopedImageDecodeBudget& operator=(const ScopedImageDecodeBudget&) = delete;
    ~ScopedImageDecodeBudget() {
        ovms::Config::instance().parse(&savedServerSettings, &savedModelsSettings);
    }

private:
    ovms::ServerSettingsImpl savedServerSettings;
    ovms::ModelsSettingsImpl savedModelsSettings;
};

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

TEST(VideoFramesProcessorTest, InjectionGuardBlocksTagInToolCallArguments) {
    // The whole message object reaches the chat template, so the reserved tag
    // must be rejected even when hidden in client-controlled fields other than
    // content, such as tool_calls[].function.arguments.
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("assistant");
    msg["content"] = std::string("");
    msg["tool_calls"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"function","function":{"name":"f","arguments":"{\"x\":\"<ov_genai_video_0>\"}"}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(VideoFramesProcessorTest, InjectionGuardBlocksTagInReasoningContent) {
    // reasoning_content is another client-controlled field rendered by some
    // chat templates, so a reserved tag placed there must also be rejected.
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("assistant");
    msg["content"] = std::string("hello");
    msg["reasoning_content"] = std::string("thinking about <ov_genai_video_0>");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(VideoFramesProcessorTest, InjectionGuardBlocksTagInTools) {
    // ChatTemplateProcessor also serializes the tools array into the prompt, so
    // a reserved tag placed in a tool definition must be rejected too.
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "hi"}});
    history.set_tools(ov::genai::JsonContainer::from_json_string(
        R"([{"type":"function","function":{"name":"f","description":"<ov_genai_video_0>"}}])"));

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(VideoFramesProcessorTest, InjectionGuardBlocksTagInExtraContext) {
    // chat_template_kwargs (extra context) is the third client-controlled
    // container rendered by the template, so a reserved tag there is rejected.
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "hi"}});
    history.set_extra_context(ov::genai::JsonContainer::from_json_string(
        R"({"custom_instruction":"<ov_genai_video_0>"})"));

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
        R"( {"type":"video_url","video_url":{"url":[")" +
        FRAME_BASE64 + R"("]}}])");
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

TEST(VideoFramesProcessorTest, PerRequestPixelBudgetRejectsAccumulatedFrames) {
    // The decoded-pixel budget is shared across all frames of a video. With a
    // budget of a single pixel, the first 1x1 frame consumes it and the second
    // frame must be rejected before its pixel buffer is allocated.
    ScopedImageDecodeBudget budgetGuard(1, /*allowUnestimatableImageFormats=*/false);

    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":[")" + FRAME_BASE64 + R"(",")" +
        FRAME_BASE64 + R"("]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Image exceeds maximum decoded size");
    EXPECT_TRUE(req.inputVideos.empty());
}

TEST(VideoFramesProcessorTest, PerRequestPixelBudgetAcceptsFramesWithinTotal) {
    // A budget large enough for both 1x1 frames lets the same two-frame video
    // through, confirming the rejection above is driven by the shared budget.
    ScopedImageDecodeBudget budgetGuard(2, /*allowUnestimatableImageFormats=*/false);

    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":[")" + FRAME_BASE64 + R"(",")" +
        FRAME_BASE64 + R"("]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_EQ(req.inputVideos.size(), 1u);
    EXPECT_EQ(req.inputVideos[0].get_shape()[0], 2u);
}

TEST(VideoFramesProcessorTest, PerRequestPixelBudgetRejectsManyFramesBeforeAllocation) {
    // Aggregate budget check: with many frame references the full
    // {numFrames, H, W, C} tensor must not be allocated when the combined frame
    // pixels exceed the budget. Here the first 1x1 frame consumes 1 pixel of the
    // budget of 4, but the remaining frames (7 pixels) cannot fit the remaining 3,
    // so the request is rejected before the stacked tensor is allocated rather
    // than after a potentially multi-GB allocation.
    ScopedImageDecodeBudget budgetGuard(4, /*allowUnestimatableImageFormats=*/false);

    std::string urls;
    for (int i = 0; i < 8; i++) {
        urls += (i ? ",\"" : "\"") + FRAME_BASE64 + "\"";
    }
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":[)" + urls + R"(]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Image exceeds maximum decoded size");
    EXPECT_TRUE(req.inputVideos.empty());
}

TEST(VideoFramesProcessorTest, OversizedFrameArrayRejected) {
    // A frame array larger than MAX_VIDEO_FRAMES must be rejected before any
    // frame is copied or decoded. The url entries can be short placeholders
    // since the size check triggers before decoding.
    std::string urls;
    for (int64_t i = 0; i < MAX_VIDEO_FRAMES + 1; i++) {
        urls += (i ? ",\"x\"" : "\"x\"");
    }
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"video_url","video_url":{"url":[)" + urls + R"(]}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    VideoFramesProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(req.inputVideos.empty());
}
