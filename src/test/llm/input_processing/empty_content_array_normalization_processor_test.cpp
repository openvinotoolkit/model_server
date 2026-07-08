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

#include "../../../llm/io_processing/input_processors/empty_content_array_normalization_processor.hpp"
#include "../../../llm/io_processing/input_request.hpp"

using namespace ovms;

// Helpers ----------------------------------------------------------------

static InputRequest makeChatRequest(ov::genai::ChatHistory chatHistory) {
    InputRequest req;
    req.input = std::move(chatHistory);
    return req;
}

// Tests ------------------------------------------------------------------

TEST(EmptyContentArrayNormalizationProcessorTest, EmptyArrayConvertedToNull) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg = {{"role", std::string("user")}};
    msg["content"] = ov::genai::JsonContainer::from_json_string("[]");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    EmptyContentArrayNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_FALSE(result[0]["content"].is_array());
    EXPECT_TRUE(result[0]["content"].is_null());
}

TEST(EmptyContentArrayNormalizationProcessorTest, NonEmptyArrayPreserved) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg = {{"role", std::string("user")}};
    msg["content"] = ov::genai::JsonContainer::from_json_string(
        R"([{"type":"text","text":"hello"}])");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    EmptyContentArrayNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    ASSERT_TRUE(result[0]["content"].is_array());
    EXPECT_EQ(result[0]["content"].size(), 1u);
}

TEST(EmptyContentArrayNormalizationProcessorTest, StringContentPassedThrough) {
    ov::genai::ChatHistory history;
    history.push_back({{"role", "user"}, {"content", "Hello, world!"}});

    InputRequest req = makeChatRequest(history);
    EmptyContentArrayNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_EQ(result[0]["content"].as_string().value_or(""), "Hello, world!");
}

TEST(EmptyContentArrayNormalizationProcessorTest, RawPromptInputRejected) {
    InputRequest req;
    req.input = std::string("raw prompt");
    EmptyContentArrayNormalizationProcessor processor;
    const auto status = processor.process(req);

    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
}
