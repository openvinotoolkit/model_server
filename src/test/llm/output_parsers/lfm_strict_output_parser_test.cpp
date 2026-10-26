// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "src/test/platform_utils.hpp"
#include "src/llm/io_processing/output_parser.hpp"

namespace {
class LfmStrictOutputParserTest : public ::testing::TestWithParam<bool> {
protected:
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    void SetUp() override {
#ifdef _WIN32
        const auto root = getWindowsRepoRootPath();
#else
        const std::string root = "/ovms";
#endif
        tokenizer = std::make_unique<ov::genai::Tokenizer>(root + "/src/test/llm_testing/LiquidAI/" + (GetParam() ? "LFM2.5-8B-A1B" : "LFM2-2.6B"));
    }
    ovms::ParsedOutput run(const std::vector<std::string>& chunks,
        ov::genai::GenerationFinishReason finish = ov::genai::GenerationFinishReason::STOP) {
        ovms::OutputParser parser(*tokenizer, "lfm2", GetParam() ? "lfm2" : "", {});
        ovms::ParsedOutput result;
        auto accept = [&](const std::optional<ovms::Delta>& delta) {
            if (!delta) return;
            if (const auto* value = std::get_if<ovms::ContentDelta>(&*delta)) result.content += value->text;
            if (const auto* value = std::get_if<ovms::ReasoningDelta>(&*delta)) result.reasoning += value->text;
            if (const auto* value = std::get_if<ovms::ToolCallDelta>(&*delta)) {
                if (value->index < 0) return;
                if (result.toolCalls.size() <= static_cast<size_t>(value->index)) result.toolCalls.resize(value->index + 1);
                auto& call = result.toolCalls[value->index];
                if (value->name) call.name = *value->name;
                if (value->id) call.id = *value->id;
                call.arguments += value->arguments;
            }
        };
        auto drain = [&]() {
            for (int i = 0; i < 100; ++i) {
                auto delta = parser.drainPendingDelta();
                if (!delta) return;
                accept(delta);
            }
            FAIL() << "Unbounded parser drain";
        };
        for (const auto& chunk : chunks) {
            accept(parser.parseChunk(chunk, {}, true, ov::genai::GenerationFinishReason::NONE));
            drain();
        }
        accept(parser.parseChunk("", {}, true, finish));
        drain();
        return result;
    }
};

TEST_P(LfmStrictOutputParserTest, AllTwoWaySplitsPreserveNestedArgumentsAndNumberSpelling) {
    const std::string text = R"(before<|tool_call_start|>[f(a=[True, {'k': None}],b='O\'Brien',n=1.10,big=9007199254740993)]<|tool_call_end|>after)";
    for (size_t split = 0; split <= text.size(); ++split) {
        SCOPED_TRACE(split);
        const auto result = run({text.substr(0, split), text.substr(split)});
        ASSERT_EQ(result.toolCalls.size(), 1);
        EXPECT_EQ(result.toolCalls[0].name, "f");
        EXPECT_EQ(result.toolCalls[0].arguments, R"({"a":[true,{"k":null}],"b":"O'Brien","n":1.10,"big":9007199254740993})");
        EXPECT_EQ(result.content, "beforeafter");
    }
}

TEST_P(LfmStrictOutputParserTest, FixedSeedMultipartSplitsKeepQuotedMarkersAndEscapedBackslashes) {
    const std::string text = R"(<|tool_call_start|>[f(path='C:\\test\\',marker='x<|tool_call_end|>y'),g(v={'a':[False,None]})]<|tool_call_end|>)";
    std::mt19937 random(1234);
    for (int iteration = 0; iteration < 40; ++iteration) {
        std::vector<std::string> chunks;
        for (size_t pos = 0; pos < text.size();) {
            const size_t length = 1 + random() % 17;
            chunks.push_back(text.substr(pos, length));
            pos += length;
        }
        const auto result = run(chunks);
        ASSERT_EQ(result.toolCalls.size(), 2);
        EXPECT_EQ(result.toolCalls[0].arguments, R"({"path":"C:\\test\\","marker":"x<|tool_call_end|>y"})");
        EXPECT_EQ(result.toolCalls[1].arguments, R"({"v":{"a":[false,null]}})");
        EXPECT_TRUE(result.content.empty());
    }
}

TEST_P(LfmStrictOutputParserTest, CompletedMalformedLiteralsThrowInsteadOfBecomingStrings) {
    for (const auto& args : {"missing", "a=", "a=unknown", "a=None,a=True", "a=__import__('os')"}) {
        EXPECT_THROW(run({std::string("<|tool_call_start|>[f(") + args + ")]<|tool_call_end|>"}), std::invalid_argument) << args;
    }
}

TEST_P(LfmStrictOutputParserTest, LengthDoesNotFabricateIncompleteArguments) {
    const auto result = run({"<|tool_call_start|>[f(a='incomplete"}, ov::genai::GenerationFinishReason::LENGTH);
    for (const auto& call : result.toolCalls) EXPECT_TRUE(call.arguments.empty());
}

TEST_P(LfmStrictOutputParserTest, ReasoningIsSeparatedFromContent) {
    if (!GetParam()) GTEST_SKIP() << "LFM2.5 reasoning only";
    const auto result = run({"<think>private</think>public"});
    EXPECT_EQ(result.reasoning, "private");
    EXPECT_EQ(result.content, "public");
}
INSTANTIATE_TEST_SUITE_P(TargetTokenizers, LfmStrictOutputParserTest, ::testing::Values(false, true));
}  // namespace
