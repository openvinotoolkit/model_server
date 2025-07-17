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
#include <gtest/gtest.h>
#include <string>
#include "../../../llm/response_parsers/base_response_parser.hpp"

using namespace ovms;

class BaseResponseParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No specific setup needed for this test class
    }
};

TEST_F(BaseResponseParserTest, wrapFirstDelta) {
    std::string functionName = "example_function";
    rapidjson::Document obj = BaseResponseParser::wrapFirstDelta(functionName, 0);
    const auto& wrappedDelta = obj["delta"];
    ASSERT_TRUE(wrappedDelta.IsObject());
    ASSERT_TRUE(wrappedDelta.HasMember("tool_calls"));
    ASSERT_TRUE(wrappedDelta["tool_calls"].IsArray());
    ASSERT_EQ(wrappedDelta["tool_calls"].Size(), 1);
    const auto& toolCall = wrappedDelta["tool_calls"][0];
    ASSERT_TRUE(toolCall.IsObject());
    ASSERT_TRUE(toolCall.HasMember("id"));
    ASSERT_TRUE(toolCall["id"].IsString());
    std::string idStr = toolCall["id"].GetString();
    // Assuming ID is a random alphanumeric string of length 9 (see src/llm/response_parsers/utils.cpp)
    ASSERT_EQ(idStr.size(), 9);
    ASSERT_TRUE(std::all_of(idStr.begin(), idStr.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c));
    }));
    ASSERT_TRUE(toolCall.HasMember("type"));
    ASSERT_EQ(toolCall["type"].GetString(), std::string("function"));
    ASSERT_TRUE(toolCall.HasMember("index"));
    ASSERT_EQ(toolCall["index"].GetInt(), 0);
    ASSERT_TRUE(toolCall.HasMember("function"));
    const auto& function = toolCall["function"];
    ASSERT_TRUE(function.IsObject());
    ASSERT_TRUE(function.HasMember("name"));
    ASSERT_EQ(function["name"].GetString(), functionName);
}

TEST_F(BaseResponseParserTest, wrapDelta) {
    std::string deltaStr = R"({
        "arguments": "location"
    })";
    rapidjson::Document delta;
    delta.Parse(deltaStr.c_str());

    rapidjson::Document obj = BaseResponseParser::wrapDelta(delta, 0);
    const auto& wrappedDelta = obj["delta"];
    ASSERT_TRUE(wrappedDelta.IsObject());
    ASSERT_TRUE(wrappedDelta.HasMember("tool_calls"));
    ASSERT_TRUE(wrappedDelta["tool_calls"].IsArray());
    ASSERT_EQ(wrappedDelta["tool_calls"].Size(), 1);
    const auto& toolCall = wrappedDelta["tool_calls"][0];
    ASSERT_TRUE(toolCall.IsObject());
    ASSERT_TRUE(toolCall.HasMember("index"));
    ASSERT_EQ(toolCall["index"].GetInt(), 0);
    ASSERT_TRUE(toolCall.HasMember("function"));
    const auto& function = toolCall["function"];
    ASSERT_TRUE(function.IsObject());
    ASSERT_TRUE(function.HasMember("arguments"));
    ASSERT_TRUE(function["arguments"].IsString());
    ASSERT_EQ(function["arguments"].GetString(), std::string("location"));
}
