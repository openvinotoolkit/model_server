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
#include <string>
#include <vector>
#include <optional>
#include <tuple>
#include <algorithm>
#include <cctype>

#include "../../../llm/io_processing/base_output_parser.hpp"
#include "../../../llm/io_processing/output_parser.hpp"
#include "output_parser_test_utils.hpp"
#include "../../platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\OpenVINO\\gemma-4-E4B-it-int4-ov";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/OpenVINO/gemma-4-E4B-it-int4-ov";
#endif

static std::unique_ptr<ov::genai::Tokenizer> gemma4Tokenizer;
static const ToolsSchemas_t& EMPTY_TOOLS_SCHEMA = {};  // not used in gemma4

class Gemma4OutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParserWithRegularToolParsing;

    static void SetUpTestSuite() {
        try {
            gemma4Tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize gemma4 tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize gemma4 tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        gemma4Tokenizer.reset();
    }

    void SetUp() override {
        // For Gemma4 model there is only tool parser available
        outputParserWithRegularToolParsing = std::make_unique<OutputParser>(*gemma4Tokenizer, "gemma4", "gemma4", EMPTY_TOOLS_SCHEMA);
    }

    void assertChunkEqual(const std::optional<ovms::Delta>& doc, const std::optional<std::string>& expectedDelta, const std::string& chunk) {
        if (!expectedDelta.has_value() && !doc.has_value()) {
            return;
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            std::string docStr = ovms::test::deltaToJson(*doc);
            std::string expected = expectedDelta.value();
            EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
        } else {
            FAIL() << "Mismatch between expectedDelta and doc for chunk: " << chunk;
        }
    }

    void assertStreamingVec(const std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>>& chunkToDeltaVec) {
        for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
            std::optional<ovms::Delta> doc = outputParserWithRegularToolParsing->parseChunk(chunk, {}, true, finishReason);
            if (!expectedDelta.has_value() && !doc.has_value()) {
                continue;  // Both are nullopt, OK
            }
            if (expectedDelta.has_value() && doc.has_value()) {
                std::string docStr = ovms::test::deltaToJson(*doc);
                // If both strings contain "id":"...", compare id values by length and alphanumeric, else compare whole strings
                std::string expected = expectedDelta.value();
                std::string idKey = "\"id\":\"";
                auto docIdPos = docStr.find(idKey);
                auto expectedIdPos = expected.find(idKey);
                if (docIdPos != std::string::npos && expectedIdPos != std::string::npos) {
                    auto docIdStart = docIdPos + idKey.size();
                    auto docIdEnd = docStr.find("\"", docIdStart);
                    auto expectedIdStart = expectedIdPos + idKey.size();
                    auto expectedIdEnd = expected.find("\"", expectedIdStart);
                    ASSERT_NE(docIdEnd, std::string::npos);
                    ASSERT_NE(expectedIdEnd, std::string::npos);
                    std::string docId = docStr.substr(docIdStart, docIdEnd - docIdStart);
                    std::string expectedId = expected.substr(expectedIdStart, expectedIdEnd - expectedIdStart);
                    EXPECT_EQ(docId.size(), expectedId.size()) << "ID length mismatch for chunk: " << chunk;
                    EXPECT_TRUE(std::all_of(docId.begin(), docId.end(), ::isalnum)) << "ID not alphanumeric for chunk: " << chunk;
                    // Compare everything except the id value
                    std::string docStrNoId = docStr;
                    std::string expectedNoId = expected;
                    docStrNoId.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
                    expectedNoId.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
                    EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk (ignoring id value): " << chunk;
                } else {
                    EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
                }
            } else {
                std::string expectedStr = expectedDelta.has_value() ? expectedDelta.value() : "std::nullopt";
                std::string docStr = doc.has_value() ? [&]() {
                    return ovms::test::deltaToJson(*doc);
                }()
                                                     : "std::nullopt";
                FAIL() << "Mismatch between expectedDelta and doc for chunk: " << chunk
                       << "\nexpectedDelta: " << expectedStr
                       << "\ndoc: " << docStr;
            }
        }
    }

    static std::string compactJsonObject(const std::string& json) {
        rapidjson::Document argsDoc;
        if (argsDoc.Parse(json.c_str()).HasParseError()) {
            return json;
        }
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
        argsDoc.Accept(writer);
        return sb.GetString();
    }

    ToolCalls_t collectLogicalToolCalls(const std::vector<std::pair<std::string, ov::genai::GenerationFinishReason>>& chunks) {
        outputParserWithRegularToolParsing->resetStreamingState();
        std::vector<ToolCall> toolCalls;
        for (const auto& [chunk, finishReason] : chunks) {
            std::optional<ovms::Delta> doc = outputParserWithRegularToolParsing->parseChunk(chunk, {}, true, finishReason);
            if (!doc.has_value()) {
                continue;
            }
            const auto* toolCallDelta = std::get_if<ToolCallDelta>(&*doc);
            if (toolCallDelta == nullptr || toolCallDelta->index < 0) {
                continue;
            }
            const auto idx = static_cast<size_t>(toolCallDelta->index);
            if (idx >= toolCalls.size()) {
                toolCalls.resize(idx + 1);
            }
            auto& toolCall = toolCalls[idx];
            if (toolCallDelta->id) {
                toolCall.id = *toolCallDelta->id;
            }
            if (toolCallDelta->name) {
                toolCall.name = *toolCallDelta->name;
            }
            toolCall.arguments.append(toolCallDelta->arguments);
        }
        ToolCalls_t completed;
        completed.reserve(toolCalls.size());
        for (auto& toolCall : toolCalls) {
            if (toolCall.arguments.empty()) {
                continue;
            }
            toolCall.arguments = compactJsonObject(toolCall.arguments);
            completed.push_back(std::move(toolCall));
        }
        return completed;
    }
};

TEST_F(Gemma4OutputParserTest, ParseToolCallOutputWithSingleToolCall) {
    std::string inputWithProperClosure = "<|tool_call>call:example_tool{arg1:<|\"|>value1<|\"|>,arg2:42}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallOutputWithSingleToolCallAndReasoning) {
    std::string inputWithProperClosure = "<|channel>thought\nSome reasoning content<channel|><|tool_call>call:example_tool{arg1:<|\"|>value1<|\"|>,arg2:42}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "Some reasoning content");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

// One generation that closes a thought, emits a tool call, and repeats that pair.
// Same envelope as ParseToolCallOutputWithSingleToolCallAndReasoning, ten times.
TEST_F(Gemma4OutputParserTest, ParseTenAlternatingThoughtAndToolCalls) {
    std::string input;
    std::string expectedReasoning;
    for (int n = 0; n < 10; ++n) {
        const std::string thought = "short thought " + std::to_string(n);
        expectedReasoning += thought;
        input += "<|channel>thought\n" + thought + "<channel|>";
        input += "<|tool_call>call:example_tool{arg1:<|\"|>v" + std::to_string(n) + "<|\"|>}<tool_call|>";
    }

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, expectedReasoning);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 10);
    for (int n = 0; n < 10; ++n) {
        EXPECT_EQ(parsedOutput.toolCalls[n].name, "example_tool") << "cycle " << n;
        EXPECT_EQ(parsedOutput.toolCalls[n].arguments, "{\"arg1\":\"v" + std::to_string(n) + "\"}") << "cycle " << n;
        EXPECT_EQ(parsedOutput.toolCalls[n].id.empty(), false) << "cycle " << n;
    }
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithoutToolCall) {
    std::string inputWithProperClosure = "<|channel>thought\nSome reasoning content<channel|>SOME CONTENT WITHOUT TOOL CALL";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
        EXPECT_EQ(parsedOutput.reasoning, "Some reasoning content");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithThoughtKeywordEmbeddedInSentence) {
    // NOTE: For the case "<channel|>thought\n ..." in the output we re-enter reasoning due to getting to UNKNOWN phase
    // in the main OutputParser where preable is honoured again. Changing that is a OutputParser level behavioral change.
    std::string input = "<|channel>I thought\nabout this for a while<channel|>Here's my thought\n and SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "Here's my thought\n and SOME CONTENT WITHOUT TOOL CALL");
    EXPECT_EQ(parsedOutput.reasoning, "I thought\nabout this for a while");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithCorruptedKeyword) {
    std::string input = "<|channel>_thought\nSome reasoning content<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    EXPECT_EQ(parsedOutput.reasoning, "Some reasoning content");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithDegenerateKeyword) {
    std::string input = "<|channel>works\n<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithSkippedKeywordLeavesEmptyHeaderLine) {
    std::string input = "<|channel>\nSome reasoning content<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    EXPECT_EQ(parsedOutput.reasoning, "Some reasoning content");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningDoesNotTreatContentNewlineAsHeaderBoundary) {
    std::string input = "<|channel>reasoning<channel|>first content line\nsecond line";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.reasoning, "reasoning");
    EXPECT_EQ(parsedOutput.content, "first content line\nsecond line");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithOpenerFollowedByNoKeyword) {
    std::string input = "<|channel> way of thinking about this is very mature and thorough.<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.reasoning, " way of thinking about this is very mature and thorough.");
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithNoKeywordPreservesMultilineFirstLine) {
    std::string input = "<|channel>first line\nsecond line<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.reasoning, "first line\nsecond line");
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithDuplicatedEndTagDoesNotEmitInterveningTextTwice) {
    std::string chunk = "<|channel>thought\nSome reasoning<channel|>Oops<channel|>Final content";

    auto doc1 = outputParserWithRegularToolParsing->parseChunk(chunk, {}, true, ov::genai::GenerationFinishReason::NONE);
    assertChunkEqual(doc1, R"({"delta":{"reasoning_content":"Some reasoning"}})", chunk);

    // First closer ended reasoning. A repeated closer cannot retract content
    // already sent in another chunk or implicitly open a private channel.
    auto doc2 = outputParserWithRegularToolParsing->parseChunk("", {}, true, ov::genai::GenerationFinishReason::NONE);
    assertChunkEqual(doc2, R"({"delta":{"content":"OopsFinal content"}})", "");

    auto doc3 = outputParserWithRegularToolParsing->parseChunk("", {}, true, ov::genai::GenerationFinishReason::STOP);
    assertChunkEqual(doc3, std::nullopt, "");
}

// Bare preambles belong only to the initial header. A trailing 't' in ordinary
// content must not hold the delta while waiting for a possible "thought\n".
TEST_F(Gemma4OutputParserTest, OrdinaryContentDoesNotWaitForReasoningPreamble) {
    auto delta = outputParserWithRegularToolParsing->parseChunk("content", {}, true, ov::genai::GenerationFinishReason::NONE);
    assertChunkEqual(delta, R"({"delta":{"content":"content"}})", "content");
}

TEST_F(Gemma4OutputParserTest, InitialBareReasoningPreambleSupportsEverySplit) {
    const std::string header = "thought\n";
    for (size_t split = 1; split < header.size(); ++split) {
        SCOPED_TRACE(split);
        outputParserWithRegularToolParsing->resetStreamingState();
        EXPECT_FALSE(outputParserWithRegularToolParsing->parseChunk(header.substr(0, split), {}, true, ov::genai::GenerationFinishReason::NONE));
        EXPECT_FALSE(outputParserWithRegularToolParsing->parseChunk(header.substr(split), {}, true, ov::genai::GenerationFinishReason::NONE));
        auto delta = outputParserWithRegularToolParsing->parseChunk("private<channel|>", {}, true, ov::genai::GenerationFinishReason::NONE);
        assertChunkEqual(delta, R"({"delta":{"reasoning_content":"private"}})", "private<channel|>");
        delta = outputParserWithRegularToolParsing->parseChunk("content", {}, true, ov::genai::GenerationFinishReason::NONE);
        assertChunkEqual(delta, R"({"delta":{"content":"content"}})", "content");
    }
}

TEST_F(Gemma4OutputParserTest, ClosedReasoningDoesNotReenterThroughBarePreamble) {
    auto delta = outputParserWithRegularToolParsing->parseChunk("<|channel>thought\nprivate<channel|>", {}, true, ov::genai::GenerationFinishReason::NONE);
    assertChunkEqual(delta, R"({"delta":{"reasoning_content":"private"}})", "initial reasoning");
    delta = outputParserWithRegularToolParsing->parseChunk("thought\npublic content", {}, true, ov::genai::GenerationFinishReason::NONE);
    assertChunkEqual(delta, R"({"delta":{"content":"thought\npublic content"}})", "post-close text");
}

TEST_F(Gemma4OutputParserTest, DuplicatedReasoningCloserHasPartitionInvariantChannels) {
    const std::string input = "<|channel>thought\nprivate<channel|>Oops<channel|>Final content";
    for (size_t split = 1; split < input.size(); ++split) {
        SCOPED_TRACE(split);
        outputParserWithRegularToolParsing->resetStreamingState();
        std::string content;
        std::string reasoning;
        const auto collect = [&](const std::optional<Delta>& delta) {
            if (!delta)
                return;
            if (const auto* text = std::get_if<ContentDelta>(&*delta))
                content += text->text;
            else if (const auto* text = std::get_if<ReasoningDelta>(&*delta))
                reasoning += text->text;
            else
                ADD_FAILURE() << "Unexpected executable delta";
        };
        for (const auto& chunk : {input.substr(0, split), input.substr(split)}) {
            collect(outputParserWithRegularToolParsing->parseChunk(chunk, {}, true, ov::genai::GenerationFinishReason::NONE));
            while (auto delta = outputParserWithRegularToolParsing->drainPendingDelta())
                collect(delta);
        }
        collect(outputParserWithRegularToolParsing->parseChunk("", {}, true, ov::genai::GenerationFinishReason::STOP));
        while (auto delta = outputParserWithRegularToolParsing->drainPendingDelta())
            collect(delta);
        EXPECT_EQ(reasoning, "private");
        EXPECT_EQ(content, "OopsFinal content");
    }
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithOpenerFollowedBySpaceThenKeywordNewline) {
    std::string input = "<|channel> thought\nSome reasoning content<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.reasoning, "Some reasoning content");
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ImplicitStartStillConsumesGeneratedChannelHeader) {
    outputParserWithRegularToolParsing->detectAndSetImplicitReasoningStart("<|turn>model\n<|channel>");
    std::vector<std::pair<std::string, std::optional<std::string>>> chunks{
        {"thought\n", std::nullopt},
        {"Some reasoning content", R"({"delta":{"reasoning_content":"Some reasoning content"}})"},
        {"<channel|>", std::nullopt},
        {"Final answer", R"({"delta":{"content":"Final answer"}})"},
    };
    for (const auto& [chunk, expected] : chunks) {
        auto doc = outputParserWithRegularToolParsing->parseChunk(chunk, {}, true, ov::genai::GenerationFinishReason::NONE);
        assertChunkEqual(doc, expected, chunk);
    }
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithMissingOpenerTag) {
    std::string input = "thought\nSome reasoning content<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    EXPECT_EQ(parsedOutput.reasoning, "Some reasoning content");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseReasoningWithNoKeywordLineAtAll) {
    std::string input = "<|channel>Acquired information: the door is now unlocked.<channel|>SOME CONTENT WITHOUT TOOL CALL";

    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "SOME CONTENT WITHOUT TOOL CALL");
    EXPECT_EQ(parsedOutput.reasoning, "Acquired information: the door is now unlocked.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseToolCallOutputWithNoToolsInTheRequest) {
    std::string inputWithProperClosure = "<|tool_call>call:example_tool{arg1:<|\"|>value1<|\"|>,arg2:42}<tool_call|>";
    std::string inputWithoutSpecialTokens = "call:example_tool{arg1:value1,arg2:42}";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, false);
        EXPECT_EQ(parsedOutput.content, inputWithoutSpecialTokens);
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    }
}

TEST_F(Gemma4OutputParserTest, NoToolsStreamerHonorsBothContentTokenVisibilityModes) {
    const std::string input = R"(<|tool_call>call:example_tool{arg1:<|"|>value1<|"|>,arg2:42}<tool_call|>)";
    auto tensor = gemma4Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    const std::vector<int64_t> tokens(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size());
    // Reusing one parser also checks that the previous stream's mode does not leak.
    for (const bool keep : {false, true, false}) {
        SCOPED_TRACE(keep);
        const auto parsed = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, tokens, false, keep);
        EXPECT_EQ(parsed.content, keep ? input : "call:example_tool{arg1:value1,arg2:42}");
        EXPECT_TRUE(parsed.reasoning.empty());
        EXPECT_TRUE(parsed.toolCalls.empty());
    }
}

TEST_F(Gemma4OutputParserTest, NoToolsHiddenContentTokensPreserveExplicitAndImplicitReasoning) {
    for (const bool implicit : {false, true}) {
        SCOPED_TRACE(implicit);
        OutputParser parser(*gemma4Tokenizer, "gemma4", "gemma4", EMPTY_TOOLS_SCHEMA);
        if (implicit)
            parser.detectAndSetImplicitReasoningStart("<|turn>model\n<|channel>thought\n");
        const std::string input = std::string(implicit ? "" : "<|channel>thought\n") +
                                  "private<channel|><|tool_call>call:example_tool{arg:<|\"|>value<|\"|>}<tool_call|>";
        auto tensor = gemma4Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        const std::vector<int64_t> tokens(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size());
        const auto parsed = ovms::test::parseWithStreamer(*gemma4Tokenizer, parser, tokens, false, false);
        EXPECT_EQ(parsed.reasoning, "private");
        EXPECT_EQ(parsed.content, "call:example_tool{arg:value}");
        EXPECT_TRUE(parsed.toolCalls.empty());
    }
}

TEST_F(Gemma4OutputParserTest, HiddenNoToolsMarkersAreInvariantToEverySplitAndFinish) {
    const std::string input = R"(before<|tool_call>call:example_tool{arg:<|"|>value<|"|>}<tool_call|>after)";
    for (const auto finish : {ov::genai::GenerationFinishReason::STOP, ov::genai::GenerationFinishReason::LENGTH}) {
        for (size_t split = 1; split < input.size(); ++split) {
            SCOPED_TRACE(split);
            outputParserWithRegularToolParsing->resetStreamingState();
            outputParserWithRegularToolParsing->setContentSpecialTokensVisible(false);
            std::string content;
            const auto collect = [&](const std::optional<Delta>& delta) {
                if (!delta)
                    return;
                const auto* text = std::get_if<ContentDelta>(&*delta);
                ASSERT_NE(text, nullptr);
                content += text->text;
            };
            for (const auto& chunk : {input.substr(0, split), input.substr(split)}) {
                collect(outputParserWithRegularToolParsing->parseChunk(chunk, {}, false, ov::genai::GenerationFinishReason::NONE));
                while (auto delta = outputParserWithRegularToolParsing->drainPendingDelta())
                    collect(delta);
            }
            collect(outputParserWithRegularToolParsing->parseChunk("", {}, false, finish));
            while (auto delta = outputParserWithRegularToolParsing->drainPendingDelta())
                collect(delta);
            EXPECT_EQ(content, "beforecall:example_tool{arg:value}after");
        }
    }
}

TEST_F(Gemma4OutputParserTest, HiddenNoToolsMarkerFragmentsSurviveTerminalReasoningDrain) {
    for (const auto finish : {ov::genai::GenerationFinishReason::STOP, ov::genai::GenerationFinishReason::LENGTH}) {
        for (const std::string marker : {"<|tool_call>", "<tool_call|>", "<|\"|>"}) {
            for (size_t length = 1; length < marker.size(); ++length) {
                const std::string suffix = "literal " + marker.substr(0, length);
                SCOPED_TRACE(suffix);
                outputParserWithRegularToolParsing->resetStreamingState();
                outputParserWithRegularToolParsing->setContentSpecialTokensVisible(false);
                const auto first = outputParserWithRegularToolParsing->parseChunk(
                    "<|channel>thought\nprivate<channel|>" + suffix, {}, false, finish);
                assertChunkEqual(first, R"({"delta":{"reasoning_content":"private"}})", "terminal envelope");
                std::string content;
                while (auto delta = outputParserWithRegularToolParsing->drainPendingDelta()) {
                    const auto* text = std::get_if<ContentDelta>(&*delta);
                    ASSERT_NE(text, nullptr);
                    content += text->text;
                }
                EXPECT_EQ(content, suffix);
                EXPECT_FALSE(outputParserWithRegularToolParsing->drainPendingDelta());
            }
        }
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithObjectArguments) {
    std::string inputWithProperClosure = "<|tool_call>call:dummy{config:{name:<|\"|>astro_config<|\"|>,value:99}}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "dummy");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"config\":{\"name\":\"astro_config\",\"value\":99}}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArguments) {
    std::string inputWithProperClosure = "<|tool_call>call:test1{arg1:<|\"|>data1,data2<|\"|>}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "test1");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"data1,data2\"}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithListOfStringsAsArgument) {
    std::string inputWithProperClosure = "<|tool_call>call:generate_DNA_sequence{length:100,preferences:[<|\"|>G<|\"|>,<|\"|>C<|\"|>]}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "generate_DNA_sequence");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"length\":100,\"preferences\":[\"G\",\"C\"]}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Gemma4OutputParserTest, ParserToolCallWithBooleanArgument) {
    std::string inputWithProperClosure = "<|tool_call>call:check_status{flag:true}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "check_status");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"flag\":true}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Gemma4OutputParserTest, ParseTwoToolCallsAtOnce) {
    std::string inputWithProperClosure = "<|tool_call>call:dummy1{config:{name:<|\"|>astro_config<|\"|>,value:99}}call:dummy2{config:{value:199,name:<|\"|>second_config<|\"|>}}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "dummy1");
        EXPECT_EQ(parsedOutput.toolCalls[1].name, "dummy2");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"config\":{\"name\":\"astro_config\",\"value\":99}}");
        EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"config\":{\"value\":199,\"name\":\"second_config\"}}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithArrayArguments) {
    std::string inputWithProperClosure = "<|tool_call>call:sort{array:[42,17,89,5,33],order:<|\"|>descending<|\"|>}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "sort");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"array\":[42,17,89,5,33],\"order\":\"descending\"}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithArrayOfObjectsArguments) {
    std::string input = R"(<|tool_call>call:read_files{files:[{path:<|"|>c:\opt\demo\hello_world_project\hello.py<|"|>},{path:<|"|>c:\opt\demo\hello_world_project\README.md<|"|>}]}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "read_files");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments,
        R"({"files":[{"path":"c:\\opt\\demo\\hello_world_project\\hello.py"},{"path":"c:\\opt\\demo\\hello_world_project\\README.md"}]})");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithArrayOfObjectsWithMixedValueTypes) {
    std::string input = R"(<|tool_call>call:batch{items:[{name:<|"|>a<|"|>,count:2,tags:[<|"|>x<|"|>,<|"|>y<|"|>]},{name:<|"|>b<|"|>,count:3,tags:[]}],dry_run:false}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "batch");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments,
        R"({"items":[{"name":"a","count":2,"tags":["x","y"]},{"name":"b","count":3,"tags":[]}],"dry_run":false})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithNestedArrayOfArrays) {
    std::string input = R"(<|tool_call>call:matrix{rows:[[1,2],[3,4]],labels:[[<|"|>a<|"|>],[<|"|>b<|"|>]]}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "matrix");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"rows":[[1,2],[3,4]],"labels":[["a"],["b"]]})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithNestedArrayOfStringWithMaskedValues) {
    std::string input = R"(<|tool_call>call:sort{strings:[[<|"|>a,b<|"|>, <|"|>c<|"|>], [<|"|>d,e<|"|>, <|"|>e,f<|"|>]]}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "sort");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"strings":[["a,b","c"],["d,e","e,f"]]})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringWithMaskedValues) {
    std::string input = R"(<|tool_call>call:rename{old_name:<|"|>a,b<|"|>, new_name:<|"|>c,d<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "rename");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"old_name":"a,b","new_name":"c,d"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithThoughtPreamble) {
    std::string input = "thought\n<channel|>"
                        R"(<|tool_call>call:rename{old_name:<|"|>a,b<|"|>, new_name:<|"|>c,d<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "rename");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"old_name":"a,b","new_name":"c,d"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithArrayOfObjectsArgumentsStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"read_files", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{files", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{path", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {R"(c:\opt\demo\hello_world_project)", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {R"(\hello.py)", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"},", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{path:<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {R"(c:\opt\demo\hello_world_project\README.md)", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"}]}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"read_files","arguments":"{\"files\":[{\"path\":\"c:\\\\opt\\\\demo\\\\hello_world_project\\\\hello.py\"},{\"path\":\"c:\\\\opt\\\\demo\\\\hello_world_project\\\\README.md\"}]}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, ParseToolCallOutputWithThreeToolCalls) {
    std::string inputWithProperClosure = "<|tool_call>call:example_tool{arg1:<|\"|>value1<|\"|>,arg2:42}call:another_tool{param1:<|\"|>data<|\"|>,param2:true}call:third_tool{key:<|\"|>value<|\"|>}<tool_call|>";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 3);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        auto firstToolCallId = parsedOutput.toolCalls[0].id;

        EXPECT_EQ(parsedOutput.toolCalls[1].name, "another_tool");
        EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
        EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);
        auto secondToolCallId = parsedOutput.toolCalls[1].id;
        EXPECT_NE(firstToolCallId, secondToolCallId);

        EXPECT_EQ(parsedOutput.toolCalls[2].name, "third_tool");
        EXPECT_EQ(parsedOutput.toolCalls[2].arguments, "{\"key\":\"value\"}");
        EXPECT_EQ(parsedOutput.toolCalls[2].id.empty(), false);
        auto thirdToolCallId = parsedOutput.toolCalls[2].id;
        EXPECT_NE(firstToolCallId, thirdToolCallId);
        EXPECT_NE(secondToolCallId, thirdToolCallId);
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallOutputWithThreeToolCallsWithContentInBetween) {
    std::string inputWithProperClosure = "Before tool calls content. "
                                         "<|tool_call>call:example_tool{arg1:<|\"|>value1<|\"|>,arg2:42}<tool_call|>"
                                         "This is some content between tool calls."
                                         "<|tool_call>call:another_tool{param1:<|\"|>data<|\"|>,param2:true}<tool_call|>"
                                         " This is some content between second and third tool call. "
                                         "<|tool_call>call:third_tool{key:<|\"|>value<|\"|>}<tool_call|>"
                                         "After tool calls content.";

    std::vector<std::string> inputs = {inputWithProperClosure};
    for (auto& input : inputs) {
        auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
        EXPECT_EQ(parsedOutput.content, "Before tool calls content. This is some content between tool calls. This is some content between second and third tool call. After tool calls content.");
        EXPECT_EQ(parsedOutput.reasoning, "");

        ASSERT_EQ(parsedOutput.toolCalls.size(), 3);
        EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
        EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
        EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
        auto firstToolCallId = parsedOutput.toolCalls[0].id;

        EXPECT_EQ(parsedOutput.toolCalls[1].name, "another_tool");
        EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"param1\":\"data\",\"param2\":true}");
        EXPECT_EQ(parsedOutput.toolCalls[1].id.empty(), false);
        auto secondToolCallId = parsedOutput.toolCalls[1].id;
        EXPECT_NE(firstToolCallId, secondToolCallId);

        EXPECT_EQ(parsedOutput.toolCalls[2].name, "third_tool");
        EXPECT_EQ(parsedOutput.toolCalls[2].arguments, "{\"key\":\"value\"}");
        EXPECT_EQ(parsedOutput.toolCalls[2].id.empty(), false);
        auto thirdToolCallId = parsedOutput.toolCalls[2].id;
        EXPECT_NE(firstToolCallId, thirdToolCallId);
        EXPECT_NE(secondToolCallId, thirdToolCallId);
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithEmptyArguments) {
    // Tool call with empty braces (no arguments)
    std::string input = "<|tool_call>call:no_args_tool{}<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "no_args_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{}");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithMultipleUtfChars) {
    // Tool call with empty braces (no arguments) and content around
    std::string input = R"(<|tool_call>call:post_tweet{content:<|"|>Check out the sorted report! 🚀 We've made improvements to the content. Tagging @currenttech and mentioning Julia for our insightful team. #currenttech #trend<|"|>,mentions:[<|"|>@currenttech<|"|>,<|"|>Julia<|"|>],tags:[<|"|>#currenttrend<|"|>]}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "post_tweet");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"content":"Check out the sorted report! 🚀 We've made improvements to the content. Tagging @currenttech and mentioning Julia for our insightful team. #currenttech #trend","mentions":["@currenttech","Julia"],"tags":["#currenttrend"]})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithMultipleUtfCharsStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"post", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_tweet", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{content", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"Check out the sorted report!", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 🚀", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" We've made improvements", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to the content.", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" Tagging @currenttech", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" and mentioning Julia", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" for our insightful team.", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" #currenttech #trend", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"mentions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"@currenttech", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"Julia", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"],", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"tags", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"#currenttrend", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"]}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"post_tweet","arguments":"{\"content\":\"Check out the sorted report! 🚀 We've made improvements to the content. Tagging @currenttech and mentioning Julia for our insightful team. #currenttech #trend\",\"mentions\":[\"@currenttech\",\"Julia\"],\"tags\":[\"#currenttrend\"]}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, ParseToolCallArgumentValueContainingComma) {
    // Comma embedded inside a string value (e.g. generated code) must
    // not be mistaken for the separator between arguments.
    std::string input =
        "<|tool_call>call:editor{new_text:<|\"|>print(\"Hello, World!\")\n"
        "<|\"|>,path:<|\"|>/home/user/demos/hello_world_python/hello.py<|\"|>}<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "editor");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments,
        "{\"new_text\":\"print(\\\"Hello, World!\\\")\\n\",\"path\":\"/home/user/demos/hello_world_python/hello.py\"}");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallArgumentValueContainingBracesAndBrackets) {
    // Literal '{', '}', '[', ']' inside a string value must not be mistaken
    // for nested object/array structure or for the tool call's own top-level closing brace.
    std::string input =
        "<|tool_call>call:editor{new_text:<|\"|>numbers = [1, 2, 3] and config = {a: 1, b: 2}<|\"|>,"
        "path:<|\"|>file.txt<|\"|>}<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "editor");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments,
        "{\"new_text\":\"numbers = [1, 2, 3] and config = {a: 1, b: 2}\",\"path\":\"file.txt\"}");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallArgumentValueWithUnclosedQuoteAndBraceMidStream) {
    // While a string value is still streaming (its closing <|"|> hasn't
    // arrived yet), an internal '"' followed by a '}' inside the already-received partial
    // value must not be mistaken for the tool call's own closing brace.
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"editor", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{new_text", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>say ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\"hi}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" keep going", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"editor","arguments":"{\"new_text\":\"say \\\"hi} keep going\"}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, ParseToolCallOutputWithContentAndNoToolCalls) {
    std::string input = "This is a regular model response without tool calls.";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "This is a regular model response without tool calls.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
    EXPECT_EQ(parsedOutput.reasoning, "");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallOutputWithContentAndSingleToolCall) {
    std::string input = "This is a content part and next will be a tool call.\n\n<|tool_call>call:example_tool{arg1:<|\"|>value1<|\"|>,arg2:42}<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "This is a content part and next will be a tool call.\n\n");
    EXPECT_EQ(parsedOutput.reasoning, "");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "example_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"arg1\":\"value1\",\"arg2\":42}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(Gemma4OutputParserTest, HolisticStreaming) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG"}})"},
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"sort", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{array", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"42", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 17", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 89", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 5", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 33", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"],", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"order", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"desc", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ending", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"sort","arguments":"{\"array\":[42,17,89,5,33],\"order\":\"descending\"}"}}]}})"},
        {"call:d", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ummy", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{config", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":{", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"astro_config", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"value", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"99", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"}}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"dummy","arguments":"{\"config\":{\"name\":\"astro_config\",\"value\":99}}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ANOTHER_CONTENT_AFTER_TOOL_CALL", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"ANOTHER_CONTENT_AFTER_TOOL_CALL"}})"},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, StreamingWithBiggerChunks) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"SOME_CONTENT", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"SOME_CONTENT"}})"},
        {"MORE_CONTENT<|tool_call>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"MORE_CONTENT"}})"},
        {"call:sort", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{array:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"[42, 17, 89, 5, 33],order:<|\"|>descending<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"sort","arguments":"{\"array\":[42,17,89,5,33],\"order\":\"descending\"}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ANOTHER_CONTENT_AFTER_TOOL_CALL", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"ANOTHER_CONTENT_AFTER_TOOL_CALL"}})"},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, StreamingWithWhitespacesBetweenToolCalls) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG"}})"},
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:sort", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{array", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":[", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"42", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 17", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 89", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 5", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 33", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"],", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"order", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"desc", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ending", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"sort","arguments":"{\"array\":[42,17,89,5,33],\"order\":\"descending\"}"}}]}})"},
        {" call:d", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ummy", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{config", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":{", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"name", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"astro_config", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {",", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"value", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"99", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"}}", ov ::genai ::GenerationFinishReason ::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"dummy","arguments":"{\"config\":{\"name\":\"astro_config\",\"value\":99}}"}}]}})"},
        {"call: solve", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{e", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"quation", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"2", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"*", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"(", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"x", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"+", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"5)", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" =", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" 13", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":2,"function":{"name":"solve","arguments":"{\"equation\":\"2*(x+5) = 13\"}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"And some content after second tool call", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"And some content after second tool call"}})"},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, StreamingWithToolCallWithEmptyParams) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"empty_params", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"empty_params","arguments":"{}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, StreamingWithToolResponseTokenAtTheEndOfGeneration) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"dummy", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{arg1:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"value1", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|\"|>}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"dummy","arguments":"{\"arg1\":\"value1\"}"}}]}})"},
        {"<tool_call|><|tool_response>", ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

// Model omits the "<tool_call|>" end tag entirely and jumps straight to a stray
// "<|tool_response>" before stopping. Arguments must be emitted exactly once
// (regression test: the finish-flush fallback used to blindly re-emit them).
TEST_F(Gemma4OutputParserTest, StreamingWithMissingEndTagBeforeStop) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ls", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{a:", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"true}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"ls","arguments":"{\"a\":true}"}}]}})"},
        {"<|tool_response>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"", ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, StreamingPublishesFullToolCallAtClosingBrace) {
    // Do not publish at the opening `{`, and do not wait for the outer end tag.
    // One full ToolCallDelta (id, name, index, arguments) is due at the unmasked `}`.
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"call:get_weather", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{city", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>Warsaw<|\"|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"get_weather","arguments":"{\"city\":\"Warsaw\"}"}}]}})"},
        {"call:get_time", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{zone", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {":<|\"|>UTC<|\"|>}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"get_time","arguments":"{\"zone\":\"UTC\"}"}}]}})"},
        {"<tool_call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);

    const std::vector<std::pair<std::string, ov::genai::GenerationFinishReason>> fineChunks{
        {"<|tool_call>", ov::genai::GenerationFinishReason::NONE},
        {"call:get_weather", ov::genai::GenerationFinishReason::NONE},
        {"{city", ov::genai::GenerationFinishReason::NONE},
        {":<|\"|>Warsaw<|\"|>", ov::genai::GenerationFinishReason::NONE},
        {"}", ov::genai::GenerationFinishReason::NONE},
        {"call:get_time", ov::genai::GenerationFinishReason::NONE},
        {"{zone", ov::genai::GenerationFinishReason::NONE},
        {":<|\"|>UTC<|\"|>}", ov::genai::GenerationFinishReason::NONE},
        {"<tool_call|>", ov::genai::GenerationFinishReason::NONE},
    };
    const std::vector<std::pair<std::string, ov::genai::GenerationFinishReason>> coarseChunks{
        {"<|tool_call>call:get_weather{city:<|\"|>Warsaw<|\"|>}", ov::genai::GenerationFinishReason::NONE},
        {"call:get_time{zone:<|\"|>UTC<|\"|>}<tool_call|>", ov::genai::GenerationFinishReason::NONE},
    };

    const ToolCalls_t fineCalls = collectLogicalToolCalls(fineChunks);
    const ToolCalls_t coarseCalls = collectLogicalToolCalls(coarseChunks);
    ASSERT_EQ(fineCalls.size(), 2);
    ASSERT_EQ(coarseCalls.size(), fineCalls.size());
    EXPECT_EQ(fineCalls[0].name, "get_weather");
    EXPECT_EQ(fineCalls[0].arguments, R"({"city":"Warsaw"})");
    EXPECT_EQ(fineCalls[1].name, "get_time");
    EXPECT_EQ(fineCalls[1].arguments, R"({"zone":"UTC"})");
    EXPECT_EQ(coarseCalls[0].name, fineCalls[0].name);
    EXPECT_EQ(coarseCalls[0].arguments, fineCalls[0].arguments);
    EXPECT_EQ(coarseCalls[1].name, fineCalls[1].name);
    EXPECT_EQ(coarseCalls[1].arguments, fineCalls[1].arguments);
}

TEST_F(Gemma4OutputParserTest, StreamingContentWithTurnTokenAtTheEndOfGeneration) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"This is", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"This is"}})"},
        {" some content", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" some content"}})"},
        {" with a turn token at the end.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" with a turn token at the end."}})"},
        {"<turn|>", ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    assertStreamingVec(chunkToDeltaVec);
}

TEST_F(Gemma4OutputParserTest, ToolCallsWithoutToolsInTheRequestStreaming) {
    std::vector<std::pair<std::string, std::optional<std::string>>> chunkToDeltaVec{
        {"<|tool_call>", "{\"delta\":{\"content\":\"<|tool_call>\"}}"},
        {"call:super", "{\"delta\":{\"content\":\"call:super\"}}"},
        {"_tool_number_two", "{\"delta\":{\"content\":\"_tool_number_two\"}}"},
        {"{arg1", "{\"delta\":{\"content\":\"{arg1\"}}"},
        {":<|\"|>", "{\"delta\":{\"content\":\":<|\\\"|>\"}}"},
        {"val{{{ue1", "{\"delta\":{\"content\":\"val{{{ue1\"}}"},
        {"<|\"|>}", "{\"delta\":{\"content\":\"<|\\\"|>}\"}}"},
        {"<tool_call|>", "{\"delta\":{\"content\":\"<tool_call|>\"}}"},
    };

    for (const auto& [chunk, expectedDelta] : chunkToDeltaVec) {
        std::optional<ovms::Delta> doc = outputParserWithRegularToolParsing->parseChunk(chunk, {}, false, ov::genai::GenerationFinishReason::NONE);
        assertChunkEqual(doc, expectedDelta, chunk);
    }
}

// The parser can receive a buffered decoded chunk before the streamer's final
// STOP. end() must drain every remaining call and signal completion only once.
TEST_F(Gemma4OutputParserTest, StreamerStopDrainsBufferedCalls) {
    for (const int count : {3, 7}) {
        for (const bool closeEnvelope : {false, true}) {
            SCOPED_TRACE(std::to_string(count) + (closeEnvelope ? " closed" : " missing end tag"));
            outputParserWithRegularToolParsing->resetStreamingState();
            std::string chunk = "<|tool_call>";
            for (int i = 0; i < count; ++i)
                chunk += "call:tool" + std::to_string(i) + "{value:" + std::to_string(i) + "}";
            if (closeEnvelope)
                chunk += "<tool_call|>";
            auto first = outputParserWithRegularToolParsing->parseChunk(chunk, {}, true, ov::genai::GenerationFinishReason::NONE);
            ASSERT_TRUE(first.has_value());
            std::vector<ToolCallDelta> calls;
            calls.push_back(std::get<ToolCallDelta>(*first));
            int lastCount = 0;
            auto parser = std::shared_ptr<OutputParser>(outputParserWithRegularToolParsing.get(), [](OutputParser*) {});
            OVMSTextStreamer streamer(*gemma4Tokenizer, parser, true,
                [&](Delta delta, bool last) {
                    EXPECT_EQ(lastCount, 0) << "No callback is allowed after completion";
                    if (auto* call = std::get_if<ToolCallDelta>(&delta))
                        calls.push_back(*call);
                    if (last)
                        ++lastCount;
                    return ov::genai::StreamingStatus::RUNNING;
                },
                {});
            streamer.end();
            EXPECT_FALSE(streamer.hadParserError());
            EXPECT_EQ(lastCount, 1);
            ASSERT_EQ(calls.size(), count);
            for (int i = 0; i < count; ++i) {
                EXPECT_EQ(calls[i].index, i);
                EXPECT_EQ(calls[i].name, "tool" + std::to_string(i));
                EXPECT_EQ(calls[i].arguments, "{\"value\":" + std::to_string(i) + "}");
                ASSERT_TRUE(calls[i].id.has_value());
                EXPECT_FALSE(calls[i].id->empty());
            }
        }
    }
}

TEST_F(Gemma4OutputParserTest, BufferedCallsAreInvariantToDecodedChunkBoundaries) {
    const std::string input = "<|tool_call>call:first{value:1}call:second{value:2}call:third{value:3}call:fourth{value:4}<tool_call|>";
    for (const size_t width : {size_t{1}, size_t{2}, size_t{7}, input.size()}) {
        SCOPED_TRACE(width);
        outputParserWithRegularToolParsing->resetStreamingState();
        std::vector<ToolCallDelta> calls;
        for (size_t pos = 0; pos < input.size(); pos += width) {
            auto delta = outputParserWithRegularToolParsing->parseChunk(input.substr(pos, width), {}, true, ov::genai::GenerationFinishReason::NONE);
            if (delta) {
                ASSERT_TRUE(std::holds_alternative<ToolCallDelta>(*delta));
                calls.push_back(std::get<ToolCallDelta>(*delta));
            }
        }
        int lastCount = 0;
        auto parser = std::shared_ptr<OutputParser>(outputParserWithRegularToolParsing.get(), [](OutputParser*) {});
        OVMSTextStreamer streamer(*gemma4Tokenizer, parser, true,
            [&](Delta delta, bool last) {
                EXPECT_EQ(lastCount, 0);
                if (auto* call = std::get_if<ToolCallDelta>(&delta))
                    calls.push_back(*call);
                if (last)
                    ++lastCount;
                return ov::genai::StreamingStatus::RUNNING;
            },
            {});
        streamer.end();
        EXPECT_EQ(lastCount, 1);
        ASSERT_EQ(calls.size(), 4);
        const std::vector<std::string> names{"first", "second", "third", "fourth"};
        for (size_t i = 0; i < calls.size(); ++i) {
            EXPECT_EQ(calls[i].index, i);
            EXPECT_EQ(calls[i].name, names[i]);
            EXPECT_EQ(calls[i].arguments, "{\"value\":" + std::to_string(i + 1) + "}");
        }
    }
    const auto encoded = gemma4Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    const std::vector<int64_t> tokens(encoded.data<int64_t>(), encoded.data<int64_t>() + encoded.get_size());
    const auto output = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, tokens, true, true);
    ASSERT_EQ(output.toolCalls.size(), 4);
    EXPECT_EQ(output.toolCalls[2].name, "third");
    EXPECT_EQ(output.toolCalls[2].arguments, "{\"value\":3}");
    EXPECT_EQ(output.toolCalls[3].name, "fourth");
    EXPECT_EQ(output.toolCalls[3].arguments, "{\"value\":4}");
}

TEST_F(Gemma4OutputParserTest, ImplicitReasoningToolHandoffPreservesAtomicRegistryCalls) {
    ToolsSchemas_t tools;
    tools.emplace("question", ToolSchemaWrapper{nullptr, R"({"type":"object","properties":{"x":{}}})"});
    for (const std::vector<std::string>& chunks : std::vector<std::vector<std::string>>{
             {"<|channel>thought\nsecret<|tool_call>call:question{x:1}"},
             {"<|channel>thought\nsecret<|tool_", "call>call:question{x:1}"},
             {"<|channel>thought\nsecret<channel|><|tool_call>call:question{x:1}"}}) {
        auto parser = std::make_shared<OutputParser>(*gemma4Tokenizer, "gemma4", "gemma4", tools);
        std::vector<ToolCallDelta> calls;
        std::string reasoning;
        auto collect = [&](Delta delta, bool) {
            if (auto* call = std::get_if<ToolCallDelta>(&delta))
                calls.push_back(*call);
            if (auto* thought = std::get_if<ReasoningDelta>(&delta))
                reasoning += thought->text;
            return ov::genai::StreamingStatus::RUNNING;
        };
        for (const auto& chunk : chunks) {
            auto delta = parser->parseChunk(chunk, {}, true, ov::genai::GenerationFinishReason::NONE);
            if (delta)
                collect(*delta, false);
        }
        OVMSTextStreamer streamer(*gemma4Tokenizer, parser, true, collect, {});
        streamer.end();
        EXPECT_EQ(reasoning, "secret");
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0].name, "question");
        EXPECT_EQ(calls[0].arguments, "{\"x\":1}");
    }
}

TEST_F(Gemma4OutputParserTest, UnavailableToolsAndNonGemmaReasoningRemainNonExecutable) {
    for (const auto& names : std::vector<std::pair<std::string, std::string>>{{"gemma4", "gemma4"}, {"gemma4", "qwen3"}}) {
        auto parser = std::make_shared<OutputParser>(*gemma4Tokenizer, names.first, names.second, EMPTY_TOOLS_SCHEMA);
        const bool toolsAvailable = names.second == "qwen3";
        const std::string input = (toolsAvailable ? "<think>example " : "<|channel>thought\nexample ") + std::string("<|tool_call>call:question{x:1}");
        std::vector<Delta> deltas;
        auto delta = parser->parseChunk(input, {}, toolsAvailable, ov::genai::GenerationFinishReason::NONE);
        if (delta)
            deltas.push_back(*delta);
        OVMSTextStreamer streamer(*gemma4Tokenizer, parser, toolsAvailable,
            [&](Delta value, bool) {
                deltas.push_back(std::move(value));
                return ov::genai::StreamingStatus::RUNNING;
            },
            {});
        streamer.end();
        EXPECT_TRUE(std::none_of(deltas.begin(), deltas.end(), [](const Delta& value) { return std::holds_alternative<ToolCallDelta>(value); }));
    }
}

// Malformed tool calls

TEST_F(Gemma4OutputParserTest, ParseToolCallWithMissingParentheses) {
    std::string input = "<|tool_call>call:broken_tool<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithMissingClosingParenthesis) {
    std::string input = "<|tool_call>call:broken_tool{arg1:<|\"|>value1<|\"|><tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithArgumentMissingEquals) {
    std::string input = "<|tool_call>call:broken{malformed_arg}<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "broken");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsContainingComparison) {
    std::string input = R"x(<|tool_call>call:search{query:<|"|>price >= 100, (sale)<|"|>,limit:5}<tool_call|>)x";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "search");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"x({"query":"price >= 100, (sale)","limit":5})x");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsContainingBracesAndBrackets) {
    std::string input = R"(<|tool_call>call:format{template:<|"|>Hello {name}, items: [a, b, c]<|"|>,count:3}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "format");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"template":"Hello {name}, items: [a, b, c]","count":3})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsContainingSpecialCharacters) {
    std::string impl = "import package\nimport package2\n\ndef func(a, b):\n\td={\"python\": \"dict\"}\n\tl = [\"list \\\"with escaped text\\\"\", 123, []]\n\treturn f\"formatted {a} and {b}\"";
    std::string input = R"(<|tool_call>call:execute{code:<|"|>)" + impl + R"(<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "execute");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"code":"import package\nimport package2\n\ndef func(a, b):\n\td={\"python\": \"dict\"}\n\tl = [\"list \\\"with escaped text\\\"\", 123, []]\n\treturn f\"formatted {a} and {b}\""})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsContainingEscapedQuotes) {
    std::string input = R"x(<|tool_call>call:execute{code:<|"|>print(\"hello world\")<|"|>,verbose:true}<tool_call|>)x";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "execute");
    // Native delimiters carry raw bytes. Unlike a JSON string, \" here
    // denotes two bytes; preserve the backslash before serializing to JSON.
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"x({"code":"print(\\\"hello world\\\")","verbose":true})x");
    rapidjson::Document args;
    ASSERT_FALSE(args.Parse(parsedOutput.toolCalls[0].arguments.c_str()).HasParseError());
    ASSERT_TRUE(args["code"].IsString());
    EXPECT_EQ(std::string(args["code"].GetString(), args["code"].GetStringLength()), R"(print(\"hello world\"))");
}

// These are semantic goldens: JSON representation may contain extra slashes,
// but decoding it must reproduce the native-delimited bytes exactly. Ordinary
// JSON strings, in contrast, interpret their JSON escapes.
TEST_F(Gemma4OutputParserTest, NativeStringValuesAndJsonEscapesHaveDistinctSemantics) {
    const std::vector<std::pair<std::string, std::string>> cases{
        {R"(code:<|"|>print("hello world")<|"|>)", R"(print("hello world"))"},
        {R"(code:<|"|>print(\"hello world\")<|"|>)", R"(print(\"hello world\"))"},
        {R"(code:<|"|>C:\Users\test\file.txt<|"|>)", R"(C:\Users\test\file.txt)"},
        {R"(code:<|"|>literal\ntext<|"|>)", R"(literal\ntext)"},
        {"code:<|\"|>line1\nline2<|\"|>", "line1\nline2"},
        {R"x(code:"print(\"hello world\")")x", R"(print("hello world"))"},
        {R"(code:"line1\nline2")", "line1\nline2"},
    };
    for (const bool registered : {false, true}) {
        ToolsSchemas_t registry;
        if (registered)
            registry.emplace("execute", ToolSchemaWrapper{nullptr, R"({"type":"object","properties":{"code":{"type":"string"}}})"});
        OutputParser parser(*gemma4Tokenizer, "gemma4", "gemma4", registry);
        for (const auto& [body, expected] : cases) {
            SCOPED_TRACE(body + (registered ? " registered" : " legacy"));
            const std::string input = "<|tool_call>call:execute{" + body + "}<tool_call|>";
            auto tensor = gemma4Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
            const std::vector<int64_t> tokens(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size());
            const auto parsed = ovms::test::parseWithStreamer(*gemma4Tokenizer, parser, tokens, true);
            ASSERT_EQ(parsed.toolCalls.size(), 1);
            EXPECT_EQ(parsed.toolCalls[0].name, "execute");
            rapidjson::Document args;
            ASSERT_FALSE(args.Parse(parsed.toolCalls[0].arguments.c_str()).HasParseError());
            ASSERT_TRUE(args.HasMember("code"));
            ASSERT_TRUE(args["code"].IsString());
            EXPECT_EQ(std::string(args["code"].GetString(), args["code"].GetStringLength()), expected);
        }
    }
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsContainingApostrophes) {
    std::string input = R"(<|tool_call>call:log{message:<|"|>it's a test, isn't it?<|"|>,level:<|"|>warn<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "log");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"message":"it's a test, isn't it?","level":"warn"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsContainingBackslashes) {
    std::string input = R"(<|tool_call>call:read_file{path:<|"|>C:\Users\test\file.txt<|"|>,encoding:<|"|>utf-8<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "read_file");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"path":"C:\\Users\\test\\file.txt","encoding":"utf-8"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsArrayWithStringsContainingQuotes) {
    std::string input = R"(<|tool_call>call:save{lines:[<|"|>it's the wonderful day<|"|>,<|"|>He said: "My name's John"<|"|>,<|"|>That's Johns' car.<|"|>]}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "save");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"lines":["it's the wonderful day","He said: \"My name's John\"","That's Johns' car."]})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsObjectWithStringsContainingQuotes) {
    std::string input = R"(<|tool_call>call:save{obj:{name:<|"|>it's the wonderful day<|"|>,greeting:<|"|>Hello, my name's Jan<|"|>,note:<|"|>That's Johns' car.<|"|>}}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "save");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"obj":{"name":"it's the wonderful day","greeting":"Hello, my name's Jan","note":"That's Johns' car."}})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithStringArgumentsContainingNestedJSON) {
    std::string input = R"(<|tool_call>call:send{payload:<|"|>{'key': 'value', 'count': 42}<|"|>,endpoint:<|"|>api<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "send");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"payload":"{'key': 'value', 'count': 42}","endpoint":"api"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithPythonReprArtifactInValue) {
    std::string input = "<|tool_call>call:startEngine{ignitionMode:s'START'}<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "startEngine");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"ignitionMode":"s'START'"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithUnwrappedValueLeadingSpace) {
    std::string input = R"(<|tool_call>call:cp{destination:<|"|>backup_tests<|"|>,source: raw_value<|"|>rest<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "cp");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"destination":"backup_tests","source":"raw_value<|\"|>rest<|\"|>"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithParenthesizedSingleQuotedValueContainingBrace) {
    std::string input = "<|tool_call>call:exec{code:fn('a}b')}<tool_call|>";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "exec");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"x({"code":"fn('a}b')"})x");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithEmptyStringArgument) {
    std::string input = R"(<|tool_call>call:create{name:<|"|><|"|>,value:0}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "create");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"name":"","value":0})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithUnicodeCharactersInArguments) {
    std::string input = R"(<|tool_call>call:translate{text:<|"|>zażółć gęślą jaźń<|"|>,lang:<|"|>pl<|"|>}<tool_call|>)";
    auto generatedTensor = gemma4Tokenizer->encode(input).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "translate");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"text":"zażółć gęślą jaźń","lang":"pl"})");
}

TEST_F(Gemma4OutputParserTest, ParseToolCallWithPythonCodeAsArgument) {
    std::string input = R"x(<|tool_call>call:string_tool{param:<|"|>
    if __name__ == "__main__":
    addresses = {}
    addresses["Hodor"] = """The door"""
    addresses["Arya"] = "Winterfell"
    for name, address in addresses.items():
        print(f'\n\t{name} lives at {address}\n\r')<|"|>}<tool_call|>)x";
    auto generatedTensor = gemma4Tokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
    ParsedOutput parsedOutput = ovms::test::parseWithStreamer(*gemma4Tokenizer, *outputParserWithRegularToolParsing, generatedTokens, true, true);
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"x({"param":"\n    if __name__ == \"__main__\":\n    addresses = {}\n    addresses[\"Hodor\"] = \"\"\"The door\"\"\"\n    addresses[\"Arya\"] = \"Winterfell\"\n    for name, address in addresses.items():\n        print(f'\\n\\t{name} lives at {address}\\n\\r')"})x");
}
