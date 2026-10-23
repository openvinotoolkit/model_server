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

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../../../llm/io_processing/output_parser.hpp"
#include "../../../llm/io_processing/gemma4/gemma4_reasoning_parser.hpp"
#include "../../../llm/io_processing/gemma4/gemma4_tool_parser.hpp"
#include "../../../llm/ovms_text_streamer.hpp"
#include "../../platform_utils.hpp"

using namespace ovms;

namespace ovms {
struct Gemma4ToolParserTestAccess {
    static size_t bufferedBytes(const Gemma4ToolParser& parser) {
        return parser.streamingContent.size();
    }
};
}

namespace {
template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\OpenVINO\\gemma-4-E4B-it-int4-ov";
#else
const std::string tokenizerPath = "/ovms/src/test/llm_testing/OpenVINO/gemma-4-E4B-it-int4-ov";
#endif

const std::string questionSchema = R"({"type":"object","properties":{"questions":{"type":"array","items":{"type":"object","properties":{"question":{"type":"string"},"header":{"type":"string"},"options":{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"},"description":{"type":"string"}}}},"multiple":{"type":"boolean"},"custom":{"type":"boolean"}}}}}})";

ParsedOutput parseWithStreamer(
    const ov::genai::Tokenizer& tokenizer,
    OutputParser& outputParser,
    const std::vector<int64_t>& generatedTokens) {
    outputParser.resetStreamingState();

    ParsedOutput result;
    std::vector<ToolCall> toolCalls;
    auto callback = [&](Delta delta, bool /*isLast*/) {
        std::visit(Overloaded{
                       [&](const ContentDelta& d) { result.content.append(d.text); },
                       [&](const ReasoningDelta& d) { result.reasoning.append(d.text); },
                       [&](const ToolCallDelta& d) {
                           if (d.index < 0)
                               return;
                           const auto idx = static_cast<size_t>(d.index);
                           if (idx >= toolCalls.size())
                               toolCalls.resize(idx + 1);
                           auto& tc = toolCalls[idx];
                           if (d.id)
                               tc.id = *d.id;
                           if (d.name)
                               tc.name = *d.name;
                           tc.arguments.append(d.arguments);
                       },
                       [](const FinishDelta&) {},
                       [](const AudioDelta&) {},
                   },
            delta);
        return ov::genai::StreamingStatus::RUNNING;
    };

    auto parserPtr = std::shared_ptr<OutputParser>(&outputParser, [](OutputParser*) {});
    const ov::AnyMap decodeParams{{ov::genai::skip_special_tokens.name(), false}};
    OVMSTextStreamer streamer(tokenizer, parserPtr, true, std::move(callback), decodeParams);
    for (int64_t token : generatedTokens)
        streamer.write(token);
    streamer.end();

    // Match the API accumulator: never hide header-only or sparse calls.
    result.toolCalls = std::move(toolCalls);
    return result;
}

class Gemma4ParserFastContractTest : public ::testing::Test {
protected:
    static std::unique_ptr<ov::genai::Tokenizer> tokenizer;

    static void SetUpTestSuite() {
        const char* configured = std::getenv("GEMMA4_TOKENIZER_PATH");
        tokenizer = std::make_unique<ov::genai::Tokenizer>(configured ? configured : tokenizerPath);
    }

    static void TearDownTestSuite() {
        tokenizer.reset();
    }

    static ToolsSchemas_t questionTools() {
        ToolsSchemas_t tools;
        tools.emplace("question", ToolSchemaWrapper{nullptr, questionSchema});
        return tools;
    }

    ParsedOutput parse(const std::string& input, const ToolsSchemas_t& tools = questionTools()) {
        OutputParser parser(*tokenizer, "gemma4", "gemma4", tools);
        auto tensor = tokenizer->encode(input).input_ids;
        std::vector<int64_t> tokens(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size());
        return parseWithStreamer(*tokenizer, parser, tokens);
    }
};

std::unique_ptr<ov::genai::Tokenizer> Gemma4ParserFastContractTest::tokenizer;

const std::string nativeQuestionArgs =
    R"(questions:[{question:<|"|>Pick one?<|"|>,header:<|"|>Test<|"|>,options:[{label:<|"|>A<|"|>,description:<|"|>Alpha<|"|>},{label:<|"|>B<|"|>,description:<|"|>Beta<|"|>}],multiple:false,custom:true}])";

const std::string expectedQuestionJson =
    R"({"questions":[{"question":"Pick one?","header":"Test","options":[{"label":"A","description":"Alpha"},{"label":"B","description":"Beta"}],"multiple":false,"custom":true}]})";
}  // namespace

TEST_F(Gemma4ParserFastContractTest, ParsesOpenCodeQuestionArrayOfObjectsRecursively) {
    auto parsed = parse("<|tool_call>call:question{" + nativeQuestionArgs + "}<tool_call|>");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
    EXPECT_EQ(parsed.toolCalls[0].arguments, expectedQuestionJson);
}

TEST_F(Gemma4ParserFastContractTest, PreservesNestedScalarTypes) {
    auto parsed = parse(R"(<|tool_call>call:question{questions:[],meta:{count:2,score:22.8,missing:null,flags:[true,false,null,3]}}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments,
        R"({"questions":[],"meta":{"count":2,"score":22.8,"missing":null,"flags":[true,false,null,3]}})");
}

TEST_F(Gemma4ParserFastContractTest, NativeDelimitedKeysDoNotLeakProtocolMarkers) {
    auto parsed = parse(R"(<|tool_call>call:question{<|"|>questions<|"|>:[],<|"|>label<|"|>:<|"|>safe<|"|>}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments, R"({"questions":[],"label":"safe"})");
    EXPECT_EQ(parsed.toolCalls[0].arguments.find("<|\"|>"), std::string::npos);
}

TEST_F(Gemma4ParserFastContractTest, StringPayloadCannotBreakStructuralScanning) {
    auto parsed = parse(R"(<|tool_call>call:question{questions:[],template:<|"|>Hello {name}, items: [a, b, c], json={"x":1}<|"|>}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments,
        R"({"questions":[],"template":"Hello {name}, items: [a, b, c], json={\"x\":1}"})");
}

TEST_F(Gemma4ParserFastContractTest, AcceptsParenthesizedArgumentsWhenAnchored) {
    auto parsed = parse("<|tool_call>call:question(" + nativeQuestionArgs + ")<tool_call|>");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments, expectedQuestionJson);
}

TEST_F(Gemma4ParserFastContractTest, AcceptsColonNameVariantWhenAnchored) {
    auto parsed = parse("<|tool_call>:question{" + nativeQuestionArgs + "}<tool_call|>");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
}

TEST_F(Gemma4ParserFastContractTest, AcceptsDirectCallAfterReasoning) {
    auto parsed = parse("<|channel>thought\nNeed user input<channel|>call:question{" + nativeQuestionArgs + "}<tool_call|>");
    EXPECT_EQ(parsed.reasoning, "Need user input");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
}

TEST_F(Gemma4ParserFastContractTest, UnknownToolIsNotExecutable) {
    auto parsed = parse(R"(<|tool_call>call:not_in_request{x:1}<tool_call|>)");
    EXPECT_TRUE(parsed.toolCalls.empty());
}

TEST_F(Gemma4ParserFastContractTest, MalformedCallIsBoundedAndLaterValidCallSurvives) {
    auto parsed = parse(R"(<|tool_call>call:question{questions:[{question:<|"|>broken<|"|>}<tool_call|><|tool_call>call:question{questions:[]}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].name, "question");
    EXPECT_EQ(parsed.toolCalls[0].arguments, R"({"questions":[]})");
}

TEST_F(Gemma4ParserFastContractTest, TruncatedArgumentsEmitNoToolCall) {
    auto parsed = parse(R"(<|tool_call>call:question{questions:[)");
    EXPECT_TRUE(parsed.toolCalls.empty());
}

TEST_F(Gemma4ParserFastContractTest, RepeatedSameFunctionCallsKeepCallLocalIdentity) {
    auto parsed = parse(
        R"(<|tool_call>call:question{questions:[]}<tool_call|>)"
        R"(<|tool_call>call:question{questions:[]}<tool_call|>)"
        R"(<|tool_call>call:question{questions:[]}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 3u);
    for (const auto& call : parsed.toolCalls) {
        EXPECT_EQ(call.name, "question");
        EXPECT_EQ(call.arguments, R"({"questions":[]})");
        EXPECT_FALSE(call.id.empty());
    }
    EXPECT_NE(parsed.toolCalls[0].id, parsed.toolCalls[1].id);
    EXPECT_NE(parsed.toolCalls[0].id, parsed.toolCalls[2].id);
    EXPECT_NE(parsed.toolCalls[1].id, parsed.toolCalls[2].id);
}

TEST_F(Gemma4ParserFastContractTest, JsonStringEndMarkerIsPayload) {
    auto parsed = parse(R"(<|tool_call>call:question{"text":"keep <tool_call|> literal","questions":[]}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments, R"({"text":"keep <tool_call|> literal","questions":[]})");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(Gemma4ParserFastContractTest, NativeStringEndMarkerIsPayload) {
    auto parsed = parse(R"(<|tool_call>call:question{text:<|"|>keep <tool_call|> literal<|"|>,questions:[]}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments, R"({"text":"keep <tool_call|> literal","questions":[]})");
    EXPECT_TRUE(parsed.content.empty());
}

TEST_F(Gemma4ParserFastContractTest, PreservesJsonEscapesAndWindowsPaths) {
    auto parsed = parse(R"(<|tool_call>call:question{"questions":[],"path":"C:\\git\\x","text":"quote: \"x\"; slash: \\; braces: {}[]"}<tool_call|>)");
    ASSERT_EQ(parsed.toolCalls.size(), 1u);
    EXPECT_EQ(parsed.toolCalls[0].arguments, R"({"questions":[],"path":"C:\\git\\x","text":"quote: \"x\"; slash: \\; braces: {}[]"})");
}

TEST_F(Gemma4ParserFastContractTest, ReasoningBoundaryChunkKeepsText) {
    Gemma4ReasoningParser parser(*tokenizer);
    auto delta = parser.parseChunk("<|channel>thought\nNeed user input<channel|>", {}, ov::genai::GenerationFinishReason::STOP);
    ASSERT_TRUE(delta.has_value());
    ASSERT_TRUE(std::holds_alternative<ReasoningDelta>(*delta));
    EXPECT_EQ(std::get<ReasoningDelta>(*delta).text, "Need user input");
}

TEST_F(Gemma4ParserFastContractTest, CompleteCallSurvivesEveryByteSplit) {
    const std::string input = R"(<|tool_call>call:question{"text":"keep <tool_call|> literal","questions":[]}<tool_call|>)";
    for (size_t split = 0; split <= input.size(); ++split) {
        SCOPED_TRACE(split);
        OutputParser parser(*tokenizer, "gemma4", "gemma4", questionTools());
        std::vector<ToolCallDelta> calls;
        for (const auto& chunk : {input.substr(0, split), input.substr(split)}) {
            auto delta = parser.parseChunk(chunk, {}, true, ov::genai::GenerationFinishReason::NONE);
            if (delta && std::holds_alternative<ToolCallDelta>(*delta))
                calls.push_back(std::get<ToolCallDelta>(*delta));
        }
        auto final = parser.parseChunk("", {}, true, ov::genai::GenerationFinishReason::STOP);
        if (final && std::holds_alternative<ToolCallDelta>(*final))
            calls.push_back(std::get<ToolCallDelta>(*final));
        ASSERT_EQ(calls.size(), 1u);
        EXPECT_EQ(calls[0].name.value_or(""), "question");
        EXPECT_EQ(calls[0].arguments, R"({"text":"keep <tool_call|> literal","questions":[]})");
    }
}

TEST_F(Gemma4ParserFastContractTest, NonGemmaReasoningDoesNotTreatLiteralToolMarkerAsBoundary) {
    OutputParser parser(*tokenizer, "gemma4", "qwen3", questionTools());
    parser.parseChunk("<think>", {}, true, ov::genai::GenerationFinishReason::NONE);
    auto delta = parser.parseChunk("literal <|tool_call> example", {}, true, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(delta.has_value());
    ASSERT_TRUE(std::holds_alternative<ReasoningDelta>(*delta));
    EXPECT_EQ(std::get<ReasoningDelta>(*delta).text, "literal <|tool_call> example");
}

TEST_F(Gemma4ParserFastContractTest, PreservesNumbersBeyondMachinePrecision) {
    const std::string numbers = "[18446744073709551617,-9223372036854775809,0.123456789012345678901,1e400,-0]";
    for (const std::string key : {"values", "\"values\""}) {
        auto parsed = parse("<|tool_call>call:question{" + key + ":" + numbers + "}<tool_call|>");
        ASSERT_EQ(parsed.toolCalls.size(), 1u);
        EXPECT_EQ(parsed.toolCalls[0].arguments, "{\"values\":" + numbers + "}");
    }
    EXPECT_EQ(Gemma4ToolParser::normalizeArgStr(numbers), numbers);
}

TEST_F(Gemma4ParserFastContractTest, MalformedNumericLexemesNeverBecomeExecutableArguments) {
    for (const std::string value : {"1.", "1e", "-", "01"}) {
        SCOPED_TRACE(value);
        auto parsed = parse("<|tool_call>call:question{value:" + value + "}<tool_call|>");
        EXPECT_TRUE(parsed.toolCalls.empty());
    }
}

TEST_F(Gemma4ParserFastContractTest, ManyCallsReleaseConsumedBufferAndKeepOwnedDeltas) {
    Gemma4ToolParser parser(*tokenizer);
    std::vector<ToolCallDelta> calls;
    for (int i = 0; i < 300; ++i) {
        const std::string input = "<|tool_call>call:question{value:" + std::to_string(i) + "}<tool_call|>";
        // Drive every internal phase, retaining deltas as a caller would.
        for (int phase = 0; phase < 6; ++phase) {
            auto delta = parser.parseChunk(phase == 0 ? input : "", {}, ov::genai::GenerationFinishReason::NONE);
            if (delta && std::holds_alternative<ToolCallDelta>(*delta))
                calls.push_back(std::get<ToolCallDelta>(*delta));
        }
        EXPECT_LT(Gemma4ToolParserTestAccess::bufferedBytes(parser), 8192u);
    }
    ASSERT_EQ(calls.size(), 300u);
    for (int i = 0; i < 300; ++i) {
        EXPECT_EQ(calls[i].index, i);
        EXPECT_EQ(calls[i].arguments, "{\"value\":" + std::to_string(i) + "}");
    }
    parser.resetState();
    EXPECT_EQ(Gemma4ToolParserTestAccess::bufferedBytes(parser), 0u);
    EXPECT_EQ(calls.front().arguments, "{\"value\":0}");
}
