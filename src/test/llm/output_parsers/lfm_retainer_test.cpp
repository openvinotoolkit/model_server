// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <memory>
#include <string>
#include <vector>
#include "src/llm/io_processing/lfm2/lfm2_tool_parser.hpp"
#include "src/test/platform_utils.hpp"

namespace {
class LfmRetainerTest : public ::testing::Test {
protected:
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    ovms::ToolsSchemas_t schemas{{"f", {nullptr, R"({"type":"object","properties":{"a":{"type":"number"},"text":{"type":"string"},"nested":{"type":"object"}},"required":["a"],"additionalProperties":false})"}},
        {"g", {nullptr, R"({"type":"object","properties":{},"additionalProperties":false})"}}};
    void SetUp() override {
#ifdef _WIN32
        const auto root = getWindowsRepoRootPath();
#else
        const std::string root = "/ovms";
#endif
        tokenizer = std::make_unique<ov::genai::Tokenizer>(root + "/src/test/llm_testing/LiquidAI/LFM2.5-8B-A1B");
    }
    static std::vector<ovms::ToolCallDelta> drain(ovms::Lfm2ToolParser& parser, std::optional<ovms::Delta> first) {
        std::vector<ovms::ToolCallDelta> calls;
        for (size_t i = 0; first && i < 100; ++i) {
            if (const auto* call = std::get_if<ovms::ToolCallDelta>(&*first)) calls.push_back(*call);
            first = parser.drainPendingDelta();
        }
        return calls;
    }
};

TEST_F(LfmRetainerTest, DoesNotPublishSpeculativeIdentity) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    EXPECT_TRUE(drain(parser, parser.parseChunk("<|tool_call_start|>[f(", {}, ov::genai::GenerationFinishReason::NONE)).empty());
}

TEST_F(LfmRetainerTest, RequiresSemanticListClosureBeforePublication) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    EXPECT_TRUE(drain(parser, parser.parseChunk("<|tool_call_start|>[f(a=1)", {}, ov::genai::GenerationFinishReason::NONE)).empty());
    EXPECT_TRUE(drain(parser, parser.parseChunk("", {}, ov::genai::GenerationFinishReason::LENGTH)).empty());
}

TEST_F(LfmRetainerTest, EmptyRegistryRejectsExecutableCalls) {
    ovms::Lfm2ToolParser parser(*tokenizer, ovms::ToolsSchemas_t{});
    EXPECT_THROW(parser.parseChunk("<|tool_call_start|>[g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP), std::invalid_argument);
    EXPECT_TRUE(drain(parser, std::nullopt).empty());
}

TEST_F(LfmRetainerTest, RejectsUnknownNameMissingRequiredWrongTypeAndExtraArgument) {
    for (const auto* body : {"[unknown()]", "[f()]", "[f(a='1')]", "[f(a=1,extra=2)]", "[f(a=1) junk]", "[f(a=1),]", "[]", "f(a=1)"}) {
        SCOPED_TRACE(body);
        ovms::Lfm2ToolParser parser(*tokenizer, schemas);
        EXPECT_THROW(parser.parseChunk(std::string("<|tool_call_start|>") + body + "<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP), std::invalid_argument);
        EXPECT_FALSE(parser.drainPendingDelta());
    }
}

TEST_F(LfmRetainerTest, RejectsWholeEnvelopeWhenLaterCallIsInvalid) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    EXPECT_THROW(parser.parseChunk("<|tool_call_start|>[g(),f(a=)]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::NONE), std::invalid_argument);
    EXPECT_FALSE(parser.drainPendingDelta());
}

TEST_F(LfmRetainerTest, EveryTwoWaySplitPreservesOneCompleteCallAndExactValues) {
    const std::string text = R"(<|tool_call_start|>[f(a=1.10,text='C:\\test\\ O\'Brien <|tool_call_end|>',nested={'n':9007199254740993,'x':[1e3,True,None]})]<|tool_call_end|>)";
    for (size_t split = 0; split <= text.size(); ++split) {
        SCOPED_TRACE(split);
        ovms::Lfm2ToolParser parser(*tokenizer, schemas);
        auto calls = drain(parser, parser.parseChunk(text.substr(0, split), {}, ov::genai::GenerationFinishReason::NONE));
        auto tail = drain(parser, parser.parseChunk(text.substr(split), {}, ov::genai::GenerationFinishReason::STOP));
        calls.insert(calls.end(), tail.begin(), tail.end());
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0].index, 0);
        ASSERT_TRUE(calls[0].id);
        EXPECT_FALSE(calls[0].id->empty());
        EXPECT_EQ(calls[0].name, "f");
        EXPECT_EQ(calls[0].arguments, R"({"a":1.10,"text":"C:\\test\\ O'Brien <|tool_call_end|>","nested":{"n":9007199254740993,"x":[1e3,true,null]}})");
    }
}

TEST_F(LfmRetainerTest, RepeatedCallsKeepDistinctIdsAndIndexesAndDrainOnce) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    const auto calls = drain(parser, parser.parseChunk("<|tool_call_start|>[g(),g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP));
    ASSERT_EQ(calls.size(), 2);
    EXPECT_EQ(calls[0].index, 0);
    EXPECT_EQ(calls[1].index, 1);
    EXPECT_NE(calls[0].id, calls[1].id);
    EXPECT_EQ(calls[0].arguments, "{}");
    EXPECT_EQ(calls[1].arguments, "{}");
    EXPECT_FALSE(parser.drainPendingDelta());
}

TEST_F(LfmRetainerTest, ParallelFalseRejectsWholeMultiCallEnvelope) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas, false);
    EXPECT_THROW(parser.parseChunk("<|tool_call_start|>[g(),g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP), std::invalid_argument);
    EXPECT_FALSE(parser.drainPendingDelta());
}

TEST_F(LfmRetainerTest, ParallelFalseRejectsSecondEnvelopeInSameTurn) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas, false);
    EXPECT_EQ(drain(parser, parser.parseChunk("<|tool_call_start|>[g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::NONE)).size(), 1);
    EXPECT_THROW(parser.parseChunk("<|tool_call_start|>[g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP), std::invalid_argument);
    EXPECT_FALSE(parser.drainPendingDelta());
}

TEST_F(LfmRetainerTest, NoTerminalModeRepairsMissingEnvelopeClosure) {
    for (auto finish : {ov::genai::GenerationFinishReason::STOP, ov::genai::GenerationFinishReason::LENGTH, ov::genai::GenerationFinishReason::TOOL_CALL}) {
        for (const auto* body : {"[f(", "[f(a=1)", "[f(a=1)]", "[f(a='incomplete"}) {
            ovms::Lfm2ToolParser parser(*tokenizer, schemas);
            EXPECT_TRUE(drain(parser, parser.parseChunk(std::string("<|tool_call_start|>") + body, {}, finish)).empty());
            EXPECT_FALSE(parser.drainPendingDelta());
        }
    }
}

TEST_F(LfmRetainerTest, ResetDiscardsCancelledCandidateAndStartsIndexesAgain) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    EXPECT_FALSE(parser.parseChunk("<|tool_call_start|>[f(a='cancelled", {}, ov::genai::GenerationFinishReason::NONE));
    parser.resetState();
    const auto calls = drain(parser, parser.parseChunk("<|tool_call_start|>[g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP));
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0].index, 0);
    EXPECT_EQ(calls[0].name, "g");
}

TEST_F(LfmRetainerTest, TerminalDiscardCannotBeCompletedByLaterRemainder) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    EXPECT_FALSE(parser.parseChunk("<|tool_call_start|>[g(", {}, ov::genai::GenerationFinishReason::LENGTH));
    EXPECT_THROW(parser.parseChunk(")]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::NONE), std::invalid_argument);
}

TEST_F(LfmRetainerTest, BoundsIncompleteCandidateBeforeLiteralParsing) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    EXPECT_THROW(parser.parseChunk("<|tool_call_start|>[f(text='" + std::string(1024 * 1024, 'x'), {}, ov::genai::GenerationFinishReason::NONE), std::invalid_argument);
    EXPECT_FALSE(parser.drainPendingDelta());
    parser.resetState();
    EXPECT_EQ(drain(parser, parser.parseChunk("<|tool_call_start|>[g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP)).size(), 1);
}

TEST_F(LfmRetainerTest, UnsupportedSchemaAndDefaultCannotFabricateRequiredValues) {
    for (const auto* schema : {R"({"type":"object","properties":{"a":{"type":"number","default":3}},"required":["a"]})",
             R"({"type":"object","unevaluatedProperties":false})"}) {
        ovms::ToolsSchemas_t registry{{"f", {nullptr, schema}}};
        ovms::Lfm2ToolParser parser(*tokenizer, registry);
        EXPECT_THROW(parser.parseChunk("<|tool_call_start|>[f()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP), std::invalid_argument);
        EXPECT_FALSE(parser.drainPendingDelta());
    }
}

TEST_F(LfmRetainerTest, ContentAndCallOrderSurviveEverySingleByteChunk) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    const std::string input = "before<|tool_call_start|>[g()]<|tool_call_end|>after";
    std::string content;
    std::vector<ovms::ToolCallDelta> calls;
    for (char value : input) {
        auto delta = parser.parseChunk(std::string(1, value), {}, ov::genai::GenerationFinishReason::NONE);
        for (size_t count = 0; delta && count < 100; ++count) {
            if (const auto* text = std::get_if<ovms::ContentDelta>(&*delta)) content += text->text;
            if (const auto* call = std::get_if<ovms::ToolCallDelta>(&*delta)) calls.push_back(*call);
            delta = parser.drainPendingDelta();
        }
    }
    EXPECT_FALSE(parser.parseChunk("", {}, ov::genai::GenerationFinishReason::STOP));
    EXPECT_EQ(content, "beforeafter");
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0].name, "g");
}

TEST_F(LfmRetainerTest, BoundsAllPublishedCallsAcrossEnvelopesUntilReset) {
    ovms::Lfm2ToolParser parser(*tokenizer, schemas);
    const std::string envelope = "<|tool_call_start|>[g(),g()]<|tool_call_end|>";
    // Intentionally avoid drain: both published IDs and pending deltas must
    // remain bounded across many individually small native envelopes.
    for (size_t i = 0; i < 2048; ++i) {
        ASSERT_TRUE(parser.parseChunk(envelope, {}, ov::genai::GenerationFinishReason::NONE));
    }
    EXPECT_THROW(parser.parseChunk(envelope, {}, ov::genai::GenerationFinishReason::NONE), std::invalid_argument);
    EXPECT_FALSE(parser.drainPendingDelta());
    parser.resetState();
    const auto calls = drain(parser, parser.parseChunk("<|tool_call_start|>[g()]<|tool_call_end|>", {}, ov::genai::GenerationFinishReason::STOP));
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0].index, 0);
}
}  // namespace
