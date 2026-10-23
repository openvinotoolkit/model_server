// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0.

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>

#include "src/llm/apis/openai_completions.hpp"
#include "src/llm/apis/openai_responses.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

namespace {

std::string requestJson(const std::string& parallelField) {
    return std::string(R"({
        "model": "gemma4",
        "messages": [{"role": "user", "content": "Use tools if needed"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "first",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
            }
        }])") + parallelField + "}";
}

struct ParsedParallelPolicy {
    absl::Status status;
    bool parallelToolCalls{true};
};

ov::genai::Tokenizer makeTokenizer() {
    return ov::genai::Tokenizer(getGenericFullPathForSrcTest(
        "/ovms/src/test/llm_testing/facebook/opt-125m"));
}

ParsedParallelPolicy parseParallelPolicy(const std::string& parallelField) {
    rapidjson::Document doc;
    const std::string json = requestJson(parallelField);
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        return {absl::InvalidArgumentError("test JSON failed to parse"), true};
    }

    OpenAIChatCompletionsHandler handler(
        doc,
        Endpoint::CHAT_COMPLETIONS,
        std::chrono::system_clock::now(),
        makeTokenizer());

    auto status = handler.parseRequest(
        /*maxTokensLimit=*/std::nullopt,
        /*bestOfLimit=*/0,
        /*maxModelLength=*/std::nullopt);
    return {status, handler.getRequest().parallelToolCalls};
}

absl::Status parseChatRequestWithoutTools(const std::string& toolChoiceJson) {
    rapidjson::Document doc;
    const std::string json = std::string(R"({
        "model": "gemma4",
        "messages": [{"role": "user", "content": "You must use a tool"}],
        "tool_choice": )") + toolChoiceJson + "}";
    doc.Parse(json.c_str());
    if (doc.HasParseError())
        return absl::InvalidArgumentError("test JSON failed to parse");

    OpenAIChatCompletionsHandler handler(
        doc,
        Endpoint::CHAT_COMPLETIONS,
        std::chrono::system_clock::now(),
        makeTokenizer());
    return handler.parseRequest(
        /*maxTokensLimit=*/std::nullopt,
        /*bestOfLimit=*/0,
        /*maxModelLength=*/std::nullopt);
}

absl::Status parseResponsesRequestWithoutTools(const std::string& toolChoiceJson) {
    rapidjson::Document doc;
    const std::string json = std::string(R"({
        "model": "gemma4",
        "input": "You must use a tool",
        "tool_choice": )") + toolChoiceJson + "}";
    doc.Parse(json.c_str());
    if (doc.HasParseError())
        return absl::InvalidArgumentError("test JSON failed to parse");

    OpenAIResponsesHandler handler(
        doc,
        Endpoint::RESPONSES,
        std::chrono::system_clock::now(),
        makeTokenizer());
    return handler.parseRequest(
        /*maxTokensLimit=*/std::nullopt,
        /*bestOfLimit=*/0,
        /*maxModelLength=*/std::nullopt);
}

void expectInvalidArgument(const absl::Status& status) {
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace

TEST(OpenAIParallelToolCallsContractTest, DefaultsToEnabledWhenFieldIsAbsent) {
    const auto result = parseParallelPolicy("");
    ASSERT_TRUE(result.status.ok()) << result.status;
    EXPECT_TRUE(result.parallelToolCalls);
}

TEST(OpenAIParallelToolCallsContractTest, ParsesExplicitBooleanPolicy) {
    const auto enabled = parseParallelPolicy(", \"parallel_tool_calls\": true");
    ASSERT_TRUE(enabled.status.ok()) << enabled.status;
    EXPECT_TRUE(enabled.parallelToolCalls);

    const auto disabled = parseParallelPolicy(", \"parallel_tool_calls\": false");
    ASSERT_TRUE(disabled.status.ok()) << disabled.status;
    EXPECT_FALSE(disabled.parallelToolCalls);
}

TEST(OpenAIParallelToolCallsContractTest, RejectsNonBooleanValues) {
    for (const std::string value : {"1", "\"no\"", "[]", "{}"}) {
        SCOPED_TRACE(value);
        const auto result = parseParallelPolicy(", \"parallel_tool_calls\": " + value);
        EXPECT_FALSE(result.status.ok());
        EXPECT_EQ(result.status.code(), absl::StatusCode::kInvalidArgument);
    }
}

TEST(OpenAIParallelToolCallsContractTest, HardToolChoiceWithoutToolsFailsClosedForChatCompletions) {
    expectInvalidArgument(parseChatRequestWithoutTools("\"required\""));
    expectInvalidArgument(parseChatRequestWithoutTools(
        R"({"type":"function","function":{"name":"first"}})"));
}

TEST(OpenAIParallelToolCallsContractTest, HardToolChoiceWithoutToolsFailsClosedForResponses) {
    expectInvalidArgument(parseResponsesRequestWithoutTools("\"required\""));
    expectInvalidArgument(parseResponsesRequestWithoutTools(
        R"({"type":"function","name":"first"})"));
}

TEST(OpenAIParallelToolCallsContractTest, ResponsesPreservesPolicyInRequestAndResponseObject) {
    rapidjson::Document doc;
    const std::string json = R"({
        "model": "gemma4",
        "input": "Use tools if needed",
        "parallel_tool_calls": false,
        "tools": [{
            "type": "function",
            "name": "first",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
        }]
    })";
    doc.Parse(json.c_str());
    ASSERT_FALSE(doc.HasParseError());

    OpenAIResponsesHandler handler(
        doc,
        Endpoint::RESPONSES,
        std::chrono::system_clock::now(),
        makeTokenizer());

    ASSERT_TRUE(handler.parseRequest(
        /*maxTokensLimit=*/std::nullopt,
        /*bestOfLimit=*/0,
        /*maxModelLength=*/std::nullopt).ok());
    EXPECT_FALSE(handler.getRequest().parallelToolCalls);

    const std::vector<Delta> deltas;
    const std::string response = handler.serializeUnaryResponse(
        deltas, ov::genai::GenerationFinishReason::STOP);
    rapidjson::Document responseDoc;
    responseDoc.Parse(response.c_str());
    ASSERT_FALSE(responseDoc.HasParseError()) << response;
    ASSERT_TRUE(responseDoc.HasMember("parallel_tool_calls"));
    ASSERT_TRUE(responseDoc["parallel_tool_calls"].IsBool());
    EXPECT_FALSE(responseDoc["parallel_tool_calls"].GetBool());
}
