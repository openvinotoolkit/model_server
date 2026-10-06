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
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <variant>

#include "src/llm/io_processing/output_parser.hpp"
#include "src/test/llm/output_parsers/output_parser_test_utils.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string graniteTokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\ibm-granite\\granite-4.2-8b";
#else
const std::string graniteTokenizerPath = "/ovms/src/test/llm_testing/ibm-granite/granite-4.2-8b";
#endif

namespace {

const ToolsSchemas_t EMPTY_TOOLS_SCHEMA = {};
std::unique_ptr<ov::genai::Tokenizer> graniteTokenizer;

std::vector<int64_t> encode(const std::string& text) {
    const auto tensor = graniteTokenizer->encode(text, ov::genai::add_special_tokens(false)).input_ids;
    return std::vector<int64_t>(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size());
}

void appendDelta(ParsedOutput& output, const Delta& delta) {
    std::visit(overloaded{
                   [&](const ContentDelta& d) { output.content.append(d.text); },
                   [&](const ReasoningDelta& d) { output.reasoning.append(d.text); },
                   [](const ToolCallDelta&) {},
                   [](const FinishDelta&) {},
                   [](const AudioDelta&) {},
               },
        delta);
}

ParsedOutput parseTextChunks(OutputParser& parser,
    const std::vector<std::string>& chunks,
    bool toolsAvailable = false) {
    parser.resetStreamingState();
    std::vector<Delta> deltas;
    for (const auto& chunk : chunks) {
        if (const auto delta = parser.parseChunk(chunk, {}, toolsAvailable,
                ov::genai::GenerationFinishReason::NONE)) {
            deltas.push_back(*delta);
        }
    }
    if (const auto delta = parser.parseChunk("", {}, toolsAvailable,
            ov::genai::GenerationFinishReason::STOP)) {
        deltas.push_back(*delta);
    }
    parser.finalizeUnaryDeltas(deltas);

    ParsedOutput output;
    for (const auto& delta : deltas) {
        appendDelta(output, delta);
    }
    return output;
}

}  // namespace

class GraniteOutputParserTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            graniteTokenizer = std::make_unique<ov::genai::Tokenizer>(graniteTokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize Granite parser test tokenizer: " << e.what();
        }
    }

    static void TearDownTestSuite() {
        graniteTokenizer.reset();
    }

    std::unique_ptr<OutputParser> createParser(bool promoteReasoningToContent = false) {
        return std::make_unique<OutputParser>(*graniteTokenizer, "", "granite42",
            EMPTY_TOOLS_SCHEMA, promoteReasoningToContent);
    }

    ParsedOutput parseEnabledThinkingOutput(const std::string& generatedText,
        bool promoteReasoningToContent = false) {
        auto parser = createParser(promoteReasoningToContent);
        // Granite's true template ends the enabled-thinking prompt with this
        // opening span; the model continuation starts inside reasoning.
        parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");
        return test::parseWithStreamer(*graniteTokenizer, *parser, encode(generatedText), false, true);
    }
};

TEST_F(GraniteOutputParserTest, ThinkingDelimitersAreSingleNonSpecialTokens) {
    const auto startTokens = encode("<think>");
    const auto endTokens = encode("</think>");

    ASSERT_EQ(startTokens.size(), 1U);
    ASSERT_EQ(endTokens.size(), 1U);
    EXPECT_EQ(graniteTokenizer->decode(startTokens, ov::genai::skip_special_tokens(true)), "<think>");
    EXPECT_EQ(graniteTokenizer->decode(endTokens, ov::genai::skip_special_tokens(true)), "</think>");
}

TEST_F(GraniteOutputParserTest, UnaryFinalContentStripsOnlyLeadingNewlines) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("reasoning</think>\n\nanswer");

    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, UnaryFinalContentRetainsLeadingSpacesAndLaterNewlines) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("reasoning</think> \nanswer\nmore");

    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, " \nanswer\nmore");
}

TEST_F(GraniteOutputParserTest, UnaryFinalContentRetainsCarriageReturnBeforeNewline) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("reasoning</think>\r\nanswer");

    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, "\r\nanswer");
}

TEST_F(GraniteOutputParserTest, UnaryReasoningOnlyStaysReasoningWhenPromotionIsDisabled) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("unfinished reasoning");

    EXPECT_EQ(parsed.reasoning, "unfinished reasoning");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, UnaryReasoningOnlyPromotesWhenRequested) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("unfinished reasoning", true);

    EXPECT_TRUE(parsed.reasoning.empty());
    EXPECT_EQ(parsed.content, "unfinished reasoning");
}

TEST_F(GraniteOutputParserTest, EmptyFinalSegmentPromotesWhenRequested) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("reasoning</think>", true);

    // Python's `content or None` makes an exactly empty post-end segment
    // indistinguishable from absent content, so Granite promotes reasoning.
    EXPECT_TRUE(parsed.reasoning.empty());
    EXPECT_EQ(parsed.content, "reasoning");
}

TEST_F(GraniteOutputParserTest, MarkerlessInitialOutputIsReasoning) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser, {"markerless reasoning"});

    EXPECT_EQ(parsed.reasoning, "markerless reasoning");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, MarkerlessInitialOutputPromotesWhenRequested) {
    auto parser = createParser(true);

    const ParsedOutput parsed = parseTextChunks(*parser, {"markerless reasoning"});

    EXPECT_TRUE(parsed.reasoning.empty());
    EXPECT_EQ(parsed.content, "markerless reasoning");
}

TEST_F(GraniteOutputParserTest, MarkerlessReasoningClosesIntoFinalContent) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"markerless reasoning", "</think>", "answer"});

    EXPECT_EQ(parsed.reasoning, "markerless reasoning");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, MarkerlessReasoningAndEndTagInOneChunkRouteRemainder) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"markerless reasoning</think>answer"});

    EXPECT_EQ(parsed.reasoning, "markerless reasoning");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, UnaryPartitionDropsPrefixBeforeStandaloneThinkingMarker) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"markerless reasoning", "<think>", "continues"});

    // The streaming parser suppresses the standalone marker, while IBM's unary
    // parent partitions the complete generated string at its first <think>.
    EXPECT_EQ(parsed.reasoning, "continues");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, UnaryPartitionDropsPrefixWhenStandaloneMarkerEndsGeneration) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"markerless reasoning", "<think>"});

    EXPECT_TRUE(parsed.reasoning.empty());
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, StreamingFusedThinkingMarkerRemainsReasoningText) {
    auto parser = createParser();
    const auto markerTokens = encode("<think>");
    ASSERT_FALSE(markerTokens.empty());
    std::vector<int64_t> fusedTokens = markerTokens;
    fusedTokens.push_back(0);

    auto delta = parser->parseChunk("<think>reasoning", fusedTokens, false,
        ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(delta.has_value());
    const auto* reasoning = std::get_if<ReasoningDelta>(&*delta);
    ASSERT_NE(reasoning, nullptr);
    EXPECT_EQ(reasoning->text, "<think>reasoning");
}

TEST_F(GraniteOutputParserTest, ImplicitPromptGeneratedStartAndEndPairStripsMarkerInStreaming) {
    auto parser = createParser();
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");

    const auto delta = parser->parseChunk("<think>reasoning</think>", {}, false,
        ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(delta.has_value());
    const auto* reasoning = std::get_if<ReasoningDelta>(&*delta);
    ASSERT_NE(reasoning, nullptr);
    EXPECT_EQ(reasoning->text, "reasoning");
}

TEST_F(GraniteOutputParserTest, UnaryFirstMarkerLeavesLaterLiteralMarkerInReasoning) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"<think>first<think>second</think>answer"});

    EXPECT_EQ(parsed.reasoning, "first<think>second");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, LaterStartEndPairRemainsLiteralAfterEarlierFusedMarker) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"<think>first", "<think>second</think>answer"});

    EXPECT_EQ(parsed.reasoning, "first<think>second");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, UnaryRestoresLaterStandaloneMarkerAfterFusedFirstMarker) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"<think>first", "<think>", "second"});

    EXPECT_EQ(parsed.reasoning, "first<think>second");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, UnaryRestoresLaterStandaloneMarkerAfterStandaloneFirstMarker) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"first", "<think>", "second", "<think>", "third"});

    EXPECT_EQ(parsed.reasoning, "second<think>third");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, LiteralLaterThinkingMarkerRemainsReasoningText) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("first <think> second");

    // Streaming preserves the fused marker; unary Granite extraction applies
    // the parent parser's first partition and therefore drops the prefix.
    EXPECT_EQ(parsed.reasoning, " second");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, ImplicitStartKeepsRedundantGeneratedThinkingMarker) {
    auto parser = createParser();
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"prefix<think>inner</think>answer"});

    EXPECT_EQ(parsed.reasoning, "inner");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, MarkersAfterFirstEndRemainFinalContent) {
    auto parser = createParser();
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"reasoning</think>", "<think>final</think>"});

    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, "<think>final</think>");
}

TEST_F(GraniteOutputParserTest, ExplicitStartAndEndInOneChunkRouteRemainderToContent) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"<think>reasoning</think>answer"});

    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, ExplicitStartDiscardsItsPreamble) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"preamble<think>reasoning</think>answer"});

    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, ExplicitTagsSplitAcrossChunksPreserveBothSegments) {
    auto parser = createParser();

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"<thi", "nk>reasoning</thi", "nk>answer"});

    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, DisabledThinkingTemplateStartsTheContinuationInContent) {
    auto parser = createParser(true);
    // Granite's disabled-thinking template closes an empty thought span in the
    // prompt; the next generated text is final content, including tool-call
    // syntax when tools are enabled.
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think></think>");

    const ParsedOutput parsed = test::parseWithStreamer(*graniteTokenizer, *parser,
        encode("\nanswer"), false, true);
    EXPECT_TRUE(parsed.reasoning.empty());
    EXPECT_EQ(parsed.content, "answer");
}

TEST_F(GraniteOutputParserTest, DisabledThinkingTemplateRoutesMarkerlessToolsToQwen3Coder) {
    auto parser = std::make_unique<OutputParser>(*graniteTokenizer, "qwen3coder",
        "granite42", EMPTY_TOOLS_SCHEMA);
    // vLLM sees the closing marker in prompt_token_ids and starts its
    // DelegatingParser in CONTENT, where qwen3_coder can claim <tool_call>.
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think></think>");

    const ParsedOutput parsed = test::parseWithStreamer(*graniteTokenizer, *parser,
        encode("<tool_call><function=get_weather></function></tool_call>"), true, true);

    EXPECT_TRUE(parsed.reasoning.empty());
    ASSERT_EQ(parsed.toolCalls.size(), 1U);
    EXPECT_EQ(parsed.toolCalls.front().name, "get_weather");
}

TEST_F(GraniteOutputParserTest, NewlineOnlyFinalSegmentPreventsUnaryPromotion) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("reasoning</think>\n\n", true);

    // Granite's unary parser first observes a present final string, then turns
    // it into "" through lstrip("\n"). It must not promote the reasoning.
    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(GraniteOutputParserTest, SpaceOnlyFinalSegmentPreventsUnaryPromotion) {
    const ParsedOutput parsed = parseEnabledThinkingOutput("reasoning</think>  ", true);

    // IBM's parser tests only final_content is None. Unlike the Nemotron
    // fallback, it does not promote when present content is whitespace.
    EXPECT_EQ(parsed.reasoning, "reasoning");
    EXPECT_EQ(parsed.content, "  ");
}

TEST_F(GraniteOutputParserTest, FirstNewlineOnlyContentChunkIsSuppressedUntilVisibleContent) {
    auto parser = createParser();
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");

    ASSERT_TRUE(parser->parseChunk("reasoning", {}, false, ov::genai::GenerationFinishReason::NONE).has_value());
    EXPECT_FALSE(parser->parseChunk("</think>", {}, false, ov::genai::GenerationFinishReason::NONE).has_value());

    const auto newlineDelta = parser->parseChunk("\n\n", {}, false, ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(newlineDelta.has_value());

    const auto answerDelta = parser->parseChunk("answer", {}, false, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(answerDelta.has_value());
    const auto* answer = std::get_if<ContentDelta>(&*answerDelta);
    ASSERT_NE(answer, nullptr);
    EXPECT_EQ(answer->text, "answer");
}

TEST_F(GraniteOutputParserTest, MalformedToolShapedFinalTextStillPreventsPromotion) {
    auto parser = std::make_unique<OutputParser>(*graniteTokenizer, "qwen3coder",
        "granite42", EMPTY_TOOLS_SCHEMA, true);
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");
    parser->resetStreamingState();

    std::vector<Delta> deltas;
    const auto reasoning = parser->parseChunk("reasoning</think>", {}, true,
        ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(reasoning.has_value());
    deltas.push_back(*reasoning);

    // The Qwen3-Coder FSM consumes this opening tag but has no usable function
    // name to emit. Granite must nevertheless remember that raw final text was
    // present, exactly as its unary parser does before tool extraction.
    EXPECT_FALSE(parser->parseChunk("<tool_call>", {}, true,
        ov::genai::GenerationFinishReason::NONE).has_value());
    EXPECT_FALSE(parser->parseChunk("", {}, true,
        ov::genai::GenerationFinishReason::STOP).has_value());
    parser->finalizeUnaryDeltas(deltas);

    ParsedOutput output;
    for (const auto& delta : deltas) {
        appendDelta(output, delta);
    }

    EXPECT_EQ(output.reasoning, "reasoning");
    EXPECT_TRUE(output.content.empty());
}

TEST_F(GraniteOutputParserTest, CompletedThinkingHandsToolCallsToExistingQwen3CoderParser) {
    auto parser = std::make_unique<OutputParser>(*graniteTokenizer, "qwen3coder",
        "granite42", EMPTY_TOOLS_SCHEMA);
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");

    const ParsedOutput parsed = test::parseWithStreamer(*graniteTokenizer, *parser,
        encode("reasoning</think><tool_call><function=get_weather><parameter=city>Bratislava</parameter></function></tool_call>"), true, true);

    EXPECT_EQ(parsed.reasoning, "reasoning");
    ASSERT_EQ(parsed.toolCalls.size(), 1U);
    EXPECT_EQ(parsed.toolCalls.front().name, "get_weather");
    EXPECT_EQ(parsed.toolCalls.front().arguments, R"({"city":"Bratislava"})");
}

TEST_F(GraniteOutputParserTest, ToolSyntaxBeforeClosingThinkTagStaysReasoning) {
    auto parser = std::make_unique<OutputParser>(*graniteTokenizer, "qwen3coder",
        "granite42", EMPTY_TOOLS_SCHEMA);
    parser->detectAndSetImplicitReasoningStart("<|im_start|>assistant\n<think>\n");

    const ParsedOutput parsed = parseTextChunks(*parser,
        {"reasoning<tool_call><function=get_weather></function></tool_call>"}, true);

    EXPECT_TRUE(parsed.toolCalls.empty());
    EXPECT_NE(parsed.reasoning.find("<tool_call>"), std::string::npos);
}
