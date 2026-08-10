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

#include "../../../llm/io_processing/input_processors/text_content_normalization_processor.hpp"
#include "../../../llm/io_processing/input_request.hpp"

using namespace ovms;

// Helpers ----------------------------------------------------------------

static InputRequest makeChatRequest(ov::genai::ChatHistory chatHistory) {
    InputRequest req;
    req.input = std::move(chatHistory);
    return req;
}

// Tests ------------------------------------------------------------------

TEST(TextContentNormalizationProcessorTest, StringContentPassedThrough) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hello, world!"}});

    InputRequest req = makeChatRequest(history);
    TextContentNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(result[0]["content"].as_string().value_or(""), "Hello, world!");
}

TEST(TextContentNormalizationProcessorTest, SingleTextPartFlattened) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg = {{"role", std::string("user")}};
    ov::genai::JsonContainer parts = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"hello"}])");
    msg["content"] = parts;
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    TextContentNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(result[0]["content"].as_string().value_or(""), "hello");
}

TEST(TextContentNormalizationProcessorTest, MultipleTextPartsJoinedWithNewline) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg = {{"role", std::string("user")}};
    ov::genai::JsonContainer parts = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"first"},{"type":"text","text":"second"}])");
    msg["content"] = parts;
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    TextContentNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(result[0]["content"].as_string().value_or(""), "first\nsecond");
}

TEST(TextContentNormalizationProcessorTest, MixedContentArrayLeftUntouched) {
    // image_url entries alongside text: the array is NOT text-only, so it must be
    // left untouched for ImageDecodingProcessor to handle downstream.
    ov::genai::ChatHistory history;
    ov::AnyMap msg = {{"role", std::string("user")}};
    ov::genai::JsonContainer parts = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"image_url","image_url":{"url":"http://example.com/img.png"}},{"type":"text","text":"describe this"}])");
    msg["content"] = parts;
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    TextContentNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    ASSERT_TRUE(result[0]["content"].is_array());
    EXPECT_EQ(result[0]["content"].size(), 2u);
}
