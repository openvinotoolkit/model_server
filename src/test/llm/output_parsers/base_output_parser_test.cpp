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
#include "../../../llm/io_processing/delta.hpp"
#include "../../../llm/apis/openai_rapidjson_delta_serializer.hpp"
#include "src/port/rapidjson_document.hpp"

using namespace ovms;

class BaseOutputParserTest : public ::testing::Test {};

// Verifies that ToolCallDelta{id, name} serializes to the OpenAI first-delta shape.
TEST_F(BaseOutputParserTest, wrapFirstDelta) {
    std::string id = "abc123XYZ";
    std::string name = "example_function";
    ToolCallDelta d{0, id, name, ""};
    RapidJsonDeltaSerializer s;
    std::string json = s.serialize(d);

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    const auto& wrappedDelta = doc["delta"];
    ASSERT_TRUE(wrappedDelta.HasMember("tool_calls"));
    const auto& tc = wrappedDelta["tool_calls"][0];
    ASSERT_EQ(std::string(tc["id"].GetString()), id);
    ASSERT_EQ(std::string(tc["type"].GetString()), "function");
    ASSERT_EQ(tc["index"].GetInt(), 0);
    ASSERT_EQ(std::string(tc["function"]["name"].GetString()), name);
}

// Verifies that ToolCallDelta{nullopt, nullopt, args} serializes to the OpenAI args-delta shape.
TEST_F(BaseOutputParserTest, wrapDelta) {
    ToolCallDelta d{0, std::nullopt, std::nullopt, "location"};
    RapidJsonDeltaSerializer s;
    std::string json = s.serialize(d);

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());
    const auto& wrappedDelta = doc["delta"];
    ASSERT_TRUE(wrappedDelta.HasMember("tool_calls"));
    const auto& tc = wrappedDelta["tool_calls"][0];
    ASSERT_EQ(tc["index"].GetInt(), 0);
    ASSERT_FALSE(tc.HasMember("id"));
    ASSERT_EQ(std::string(tc["function"]["arguments"].GetString()), "location");
}
