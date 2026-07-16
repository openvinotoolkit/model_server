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

#include "../../../llm/io_processing/input_processors/empty_tool_calls_array_removing_processor.hpp"
#include "../../../llm/io_processing/input_request.hpp"

using namespace ovms;

// Helpers ----------------------------------------------------------------

static InputRequest makeChatRequest(ov::genai::ChatHistory chatHistory) {
    InputRequest req;
    req.input = std::move(chatHistory);
    return req;
}

// Tests ------------------------------------------------------------------

TEST(EmptyToolCallsArrayRemovingProcessorTest, EmptyToolCallsArrayRemoved) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg = {{"role", std::string("user")}};
    msg["content"] = ov::genai::JsonContainer::from_json_string("\"What is the weather in Szczecin?\"");
    msg["tool_calls"] = ov::genai::JsonContainer::from_json_string("[]");
    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    EmptyToolCallsArrayRemovingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    EXPECT_TRUE(result[0]["content"].is_string());
    EXPECT_EQ(result[0]["content"].as_string().value_or(""), "What is the weather in Szczecin?");
    
}

TEST(EmptyToolCallsArrayRemovingProcessorTest, NonEmptyArrayPreserved) {
    ov::genai::ChatHistory history;
    ov::AnyMap msg = {{"role", std::string("assistant")}};
    msg["tool_calls"] = ov::genai::JsonContainer::from_json_string(
        R"([{"name":"get_weather","parameters":{"city":"Szczecin"}}])");


    history.push_back(msg);

    InputRequest req = makeChatRequest(history);
    EmptyToolCallsArrayRemovingProcessor processor;
    const auto status = processor.process(req);

    EXPECT_TRUE(status.ok());
    const auto& result = std::get<ov::genai::ChatHistory>(req.input);
    ASSERT_TRUE(result[0]["tool_calls"].is_array());
    EXPECT_EQ(result[0]["tool_calls"].size(), 1u);
    EXPECT_EQ(result[0]["tool_calls"][0]["name"].as_string().value_or(""), "get_weather");
    ASSERT_TRUE(result[0]["tool_calls"][0].contains("parameters"));
    EXPECT_EQ(result[0]["tool_calls"][0]["parameters"]["city"].as_string().value_or(""), "Szczecin");
}
