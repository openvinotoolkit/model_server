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

#include "../../../llm/io_processing/input_processors/image_decoding_processor.hpp"
#include "../../../llm/io_processing/input_request.hpp"

using namespace ovms;

// Helpers ----------------------------------------------------------------

static InputRequest makeChatRequest(ov::genai::ChatHistory chatHistory) {
    InputRequest req;
    req.input = std::move(chatHistory);
    return req;
}

// Tests ------------------------------------------------------------------

TEST(ImageDecodingProcessorTest, NoImagesInTextOnlyMessage) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hello, world!"}});

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(req.inputImages.empty());
    // Content unchanged
    const auto& resultHistory = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(resultHistory[0]["content"].as_string().value_or(""), "Hello, world!");
}

TEST(ImageDecodingProcessorTest, InjectionGuardBlocksPreexistingTag) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "<ov_genai_image_0>\nsome text"}});

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ImageDecodingProcessorTest, InjectionGuardBlocksTagInMiddleOfContent) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "prefix <ov_genai_image_2> suffix"}});

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ImageDecodingProcessorTest, SkipsMessagesWithNonArrayContent) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "system"}, {"content", "You are helpful."}});
    history.push_back({{"role", "user"}, {"content", "What is OpenVINO?"}});

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(req.inputImages.empty());
}

TEST(ImageDecodingProcessorTest, InjectionGuardBlocksTagInArrayTextPart) {
    // A multimodal message where a text part embeds the restricted tag.
    // Without the array-aware guard this would bypass the check.
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    ov::genai::JsonContainer contentArray = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"look at this <ov_genai_image_0> tag"}])");
    msg["content"] = contentArray;
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ImageDecodingProcessorTest, MultipleTextPartsJoinedWithNewline) {
    // Two text parts in a single message's content array should be joined with \n.
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    ov::genai::JsonContainer contentArray = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"Before image."},{"type":"text","text":"After image."}])");
    msg["content"] = contentArray;
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    const auto& resultHistory = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(resultHistory[0]["content"].as_string().value_or(""), "Before image.\nAfter image.");
}

// --- URL / path validation tests -----------------------------------------

// Helper: build a chat request whose single message has a content array with
// one image_url part pointing at the given URL.
static InputRequest makeImageUrlRequest(const std::string& url) {
    std::string contentJson =
        R"([{"type":"image_url","image_url":{"url":")" + url + R"("}}])";
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(contentJson);
    history.push_back(msg);
    return makeChatRequest(history);
}

TEST(ImageDecodingProcessorTest, Base64InvalidDataRejected) {
    InputRequest req = makeImageUrlRequest("data:image/jpeg;base64,NOT_VALID_BASE64!!!");
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Invalid base64 string in request");
}

TEST(ImageDecodingProcessorTest, HttpUrlDomainNotInAllowList) {
    InputRequest req = makeImageUrlRequest("http://evil.com/image.jpg");
    ImageDecodingProcessor processor(std::nullopt, std::vector<std::string>{"safe.com"});
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Given url does not match any allowed domain from allowed_media_domains");
}

TEST(ImageDecodingProcessorTest, HttpUrlWithNoAllowedDomainsConfiguredRejected) {
    InputRequest req = makeImageUrlRequest("http://any.com/image.jpg");
    // No allowed domains configured — all HTTP URLs must be rejected.
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Given url does not match any allowed domain from allowed_media_domains");
}

TEST(ImageDecodingProcessorTest, LocalFilesystemDisabledRejected) {
    InputRequest req = makeImageUrlRequest("/some/image.png");
    // No allowedLocalMediaPath configured — local filesystem access is disabled.
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Loading images from local filesystem is disabled.");
}

TEST(ImageDecodingProcessorTest, LocalPathTraversalWithDotDotRejected) {
    InputRequest req = makeImageUrlRequest("../escape/image.png");
    ImageDecodingProcessor processor(std::string("/allowed/path"), std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ImageDecodingProcessorTest, LocalPathOutsideAllowedDirectoryRejected) {
    InputRequest req = makeImageUrlRequest("/outside/image.png");
    ImageDecodingProcessor processor(std::string("/allowed/path"), std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Given filepath is not subpath of allowed_local_media_path");
}

TEST(ImageDecodingProcessorTest, LocalPathInsideAllowedDirectoryButFileNotFound) {
    InputRequest req = makeImageUrlRequest("/allowed/path/nonexistent.png");
    ImageDecodingProcessor processor(std::string("/allowed/path"), std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Image file parsing failed");
}
