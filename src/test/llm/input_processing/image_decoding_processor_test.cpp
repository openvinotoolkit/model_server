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
#include <filesystem>
#include <string>
#include <system_error>

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

TEST(ImageDecodingProcessorTest, TextOnlyArrayLeftUntouched) {
    // An array with only text parts and no images should be left as-is.
    // Flattening is deferred to TextContentNormalizationProcessor.
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
    EXPECT_TRUE(req.inputImages.empty());
    const auto& resultHistory = std::get<ov::genai::ChatHistory>(req.input);
    // Content remains an array (not flattened to string)
    EXPECT_TRUE(resultHistory[0]["content"].is_array());
    EXPECT_EQ(resultHistory[0]["content"].size(), 2u);
    EXPECT_EQ(resultHistory[0]["content"][0]["text"].as_string().value_or(""), "Before image.");
    EXPECT_EQ(resultHistory[0]["content"][1]["text"].as_string().value_or(""), "After image.");
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

TEST(ImageDecodingProcessorTest, Base64ImageDecodedAndStoredInInputImages) {
    // Minimal 1x1 PNG — verifies the success path: tensor is populated, image tag
    // replaces image_url in content array, and inputImages receives exactly one entry.
    const std::string base64Url =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
        "AAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg==";

    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"image_url","image_url":{"url":")" + base64Url +
        R"("}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputImages.size(), 1u);
    // Content is an array with image_url replaced by text tag part
    const auto& resultHistory = std::get<ov::genai::ChatHistory>(req.input);
    const auto content = resultHistory[0]["content"];
    ASSERT_TRUE(content.is_array());
    ASSERT_EQ(content.size(), 1u);
    EXPECT_EQ(content[0]["type"].as_string().value_or(""), "text");
    EXPECT_NE(content[0]["text"].as_string().value_or("").find("<ov_genai_image_0>"), std::string::npos);
}

TEST(ImageDecodingProcessorTest, ImageReplacedWhileOtherPartsPreserved) {
    // Content: [text, image_url, input_audio] — after processing:
    // [text, text(tag), input_audio] — image replaced, others untouched.
    const std::string base64Url =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
        "AAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg==";

    std::string contentJson =
        R"([{"type":"text","text":"describe"},)"
        R"({"type":"image_url","image_url":{"url":")" +
        base64Url + R"("}},)"
                    R"({"type":"input_audio","input_audio":{"data":"dGVzdA==","format":"wav"}}])";

    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(contentJson);
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(req.inputImages.size(), 1u);
    const auto& resultHistory = std::get<ov::genai::ChatHistory>(req.input);
    const auto content = resultHistory[0]["content"];
    ASSERT_TRUE(content.is_array());
    ASSERT_EQ(content.size(), 3u);
    // First part: original text preserved
    EXPECT_EQ(content[0]["type"].as_string().value_or(""), "text");
    EXPECT_EQ(content[0]["text"].as_string().value_or(""), "describe");
    // Second part: image_url replaced with text tag
    EXPECT_EQ(content[1]["type"].as_string().value_or(""), "text");
    EXPECT_NE(content[1]["text"].as_string().value_or("").find("<ov_genai_image_0>"), std::string::npos);
    // Third part: input_audio preserved as-is
    EXPECT_EQ(content[2]["type"].as_string().value_or(""), "input_audio");
}

TEST(ImageDecodingProcessorTest, MultipleImageUrlsProduceSeparateInputImageSlots) {
    // A content array with two image_url entries should produce two entries in
    // req.inputImages and two <ov_genai_image_N> tags — error path tested because
    // the URLs are invalid (not-a-real-file), but the processor must attempt both
    // and fail on the first, confirming iteration order rather than early exit
    // on array detection.
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    ov::genai::JsonContainer contentArray = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"image_url","image_url":{"url":"/no/such/first.png"}},)"
        R"( {"type":"image_url","image_url":{"url":"/no/such/second.png"}}])");
    msg["content"] = contentArray;
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    // With no allowedLocalMediaPath the first image_url triggers the
    // "disabled" error immediately. The important thing is that a multi-image
    // content array is parsed without crashing and returns a deterministic error.
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Loading images from local filesystem is disabled.");
    EXPECT_TRUE(req.inputImages.empty());
}

TEST(ImageDecodingProcessorTest, InterleavedTextAndImageUrlInContentArray) {
    // Real-life multimodal request: [text, image_url, text].
    // The processor must reach the image_url entry even when preceded by text parts.
    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    ov::genai::JsonContainer contentArray = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"What is in this image?"},)"
        R"( {"type":"image_url","image_url":{"url":"/no/image.png"}},)"
        R"( {"type":"text","text":"Describe it in detail."}])");
    msg["content"] = contentArray;
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    // Filesystem disabled — error must originate from the image_url entry.
    ImageDecodingProcessor processor(std::nullopt, std::nullopt);
    const auto status = processor.process(req);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Loading images from local filesystem is disabled.");
}

TEST(ImageDecodingProcessorTest, SymlinkEscapeInsideAllowedPathRejected) {
#ifdef _WIN32
    GTEST_SKIP() << "Symlink creation requires elevated privileges on Windows.";
#else
    // The allowed directory contains a symlink pointing to a sibling directory where
    // the real image lives. Accessing the image through the symlink makes the path
    // appear to be inside the allowed root, but weakly_canonical() resolves the
    // symlink and reveals the true location is outside the allowlist.
    const std::filesystem::path realImageDir =
        std::filesystem::path("/ovms/src/test/binaryutils");
    const std::filesystem::path allowedRoot =
        std::filesystem::temp_directory_path() / "ovms_iproc_symlink_test";
    std::error_code ec;
    std::filesystem::remove_all(allowedRoot, ec);
    ASSERT_TRUE(std::filesystem::create_directories(allowedRoot, ec)) << ec.message();

    const std::filesystem::path symlinkInside = allowedRoot / "linked";
    std::filesystem::create_directory_symlink(realImageDir, symlinkInside, ec);
    if (ec) {
        std::filesystem::remove_all(allowedRoot);
        GTEST_SKIP() << "Cannot create symlink: " << ec.message();
    }

    const std::string imageUrl = (symlinkInside / "rgb.jpg").string();

    ov::genai::ChatHistory history;
    ov::AnyMap msg;
    msg["role"] = std::string("user");
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"image_url","image_url":{"url":")" + imageUrl + R"("}}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    ImageDecodingProcessor processor(allowedRoot.string(), std::nullopt);
    const auto status = processor.process(req);
    std::filesystem::remove_all(allowedRoot, ec);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "Given filepath is not subpath of allowed_local_media_path");
#endif
}
