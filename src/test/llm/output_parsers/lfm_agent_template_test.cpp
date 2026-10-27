// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#ifndef LFM_TEMPLATE_STANDALONE
#include "src/test/platform_utils.hpp"
#endif

namespace {
class LfmAgentTemplateTest : public ::testing::TestWithParam<bool> {
protected:
    std::string root;
    std::string source;
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    void SetUp() override {
#if defined(LFM_TEMPLATE_STANDALONE)
        root = std::filesystem::current_path().string();
#elif defined(_WIN32)
        root = getWindowsRepoRootPath();
#else
        root = "/ovms";
#endif
        const std::string fixture = root + "/src/test/llm_testing/LiquidAI/" + (GetParam() ? "LFM2.5-8B-A1B" : "LFM2-2.6B");
        tokenizer = std::make_unique<ov::genai::Tokenizer>(fixture);
        const std::string path = root + "/docs/lfm/templates/" + (GetParam() ? "LFM2.5-8B-A1B-int4-ov.jinja" : "LFM2-24B-A2B-int4-ov.jinja");
        std::ifstream file(path);
        ASSERT_TRUE(file.good()) << path;
        source.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    std::string render(const std::vector<std::string>& messages,
        const std::string& tools = "[]", const std::string& kwargs = "{}") {
        ov::genai::ChatHistory history;
        for (const auto& message : messages)
            history.push_back(ov::genai::JsonContainer::from_json_string(message));
        return tokenizer->apply_chat_template(history, true, source,
            ov::genai::JsonContainer::from_json_string(tools),
            ov::genai::JsonContainer::from_json_string(kwargs));
    }
    std::vector<int64_t> tokens(const std::string& text) {
        const auto encoded = tokenizer->encode(text, ov::AnyMap{{"add_special_tokens", false}});
        const auto* data = encoded.input_ids.data<const int64_t>();
        return {data, data + encoded.input_ids.get_size()};
    }
};

TEST_P(LfmAgentTemplateTest, WrappedAndFlatDefinitionsHaveIdenticalNativeTokens) {
    const std::vector<std::string> messages{R"({"role":"user","content":"Read the directory."})"};
    const std::string flat = R"([{"name":"cwd","description":"Read current directory","parameters":{"type":"object","properties":{}}}])";
    const std::string wrapped = R"([{"type":"function","function":{"name":"cwd","description":"Read current directory","parameters":{"type":"object","properties":{}}}}])";
    EXPECT_EQ(tokens(render(messages, flat)), tokens(render(messages, wrapped)));
}

TEST_P(LfmAgentTemplateTest, ContentPrecedesToolCallAndNullContentIsEmpty) {
    const std::string call = R"("tool_calls":[{"type":"function","function":{"name":"cwd","arguments":{}}}])";
    const auto text = render({R"({"role":"user","content":"Act."})", "{\"role\":\"assistant\",\"content\":\"Checking.\"," + call + "}"});
    EXPECT_NE(text.find("Checking.<|tool_call_start|>[cwd()]<|tool_call_end|>"), std::string::npos) << text;
    const auto empty = render({"{\"role\":\"assistant\",\"content\":null," + call + "}"});
    EXPECT_NE(empty.find("assistant\n<|tool_call_start|>[cwd()]<|tool_call_end|>"), std::string::npos) << empty;
    EXPECT_EQ(empty.find("null"), std::string::npos) << empty;
}

TEST_P(LfmAgentTemplateTest, AdaptedObjectArgumentsUseNativeKeywordSyntax) {
    const auto text = render({R"({"role":"assistant","content":"","tool_calls":[{"function":{"name":"write","arguments":{"path":"cube.py","flags":[true,null,{"name":"O'Brien"}]}}}]})"});
    EXPECT_NE(text.find("write(path=\"cube.py\", flags=[true, null, {\"name\": \"O'Brien\"}])"), std::string::npos) << text;
}

TEST_P(LfmAgentTemplateTest, ReasoningAliasesReplayIdentically) {
    const auto thinking = render({R"({"role":"assistant","thinking":"inspect result","content":"done"})"});
    const auto reasoning = render({R"({"role":"assistant","reasoning":"inspect result","content":"done"})"});
    const auto content = render({R"({"role":"assistant","reasoning_content":"inspect result","content":"done"})"});
    EXPECT_NE(thinking.find("<think>inspect result</think>done"), std::string::npos);
    EXPECT_EQ(tokens(thinking), tokens(reasoning));
    EXPECT_EQ(tokens(thinking), tokens(content));
}

TEST_P(LfmAgentTemplateTest, ToolResultAndGenerationSuffixUseCanonicalTokens) {
    const std::vector<std::string> messages{
        R"({"role":"user","content":"Read cwd."})",
        R"({"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"cwd","arguments":{}}}]})",
        R"({"role":"tool","tool_call_id":"call_1","content":"C:\\work"})"};
    const auto text = render(messages);
    EXPECT_NE(text.find("<|im_start|>tool\nC:\\work<|im_end|>\n<|im_start|>assistant\n"), std::string::npos) << text;
    // Neither approved model has a canonical enable_thinking suffix switch.
    // Passing the setting must not invent a prefix the checkpoint was not trained on.
    EXPECT_EQ(tokens(text), tokens(render(messages, "[]", R"({"enable_thinking":false})")));
    EXPECT_EQ(tokens(text), tokens(render(messages, "[]", R"({"enable_thinking":true})")));
}

TEST_P(LfmAgentTemplateTest, NativeHistoryMatchesGoldenTokenSequence) {
    const std::string actual = render({
        R"({"role":"user","content":"Read cwd."})",
        R"({"role":"assistant","content":"Checking.","tool_calls":[{"function":{"name":"cwd","arguments":{}}}]})",
        R"({"role":"tool","content":"C:\\work"})"});
    const std::string expected = "<|startoftext|><|im_start|>user\nRead cwd.<|im_end|>\n"
        "<|im_start|>assistant\nChecking.<|tool_call_start|>[cwd()]<|tool_call_end|><|im_end|>\n"
        "<|im_start|>tool\nC:\\work<|im_end|>\n<|im_start|>assistant\n";
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(tokens(actual), tokens(expected));
}

TEST_P(LfmAgentTemplateTest, PastReasoningRetentionIsExplicit) {
    const std::vector<std::string> messages{
        R"({"role":"user","content":"First."})",
        R"({"role":"assistant","reasoning_content":"old-private","content":"first-answer"})",
        R"({"role":"user","content":"Next."})",
        R"({"role":"assistant","reasoning_content":"current-private","content":"next-answer"})",
        R"({"role":"tool","content":"result"})"};
    const auto ordinary = render(messages);
    EXPECT_EQ(ordinary.find("old-private"), std::string::npos);
    EXPECT_NE(ordinary.find("<think>current-private</think>next-answer"), std::string::npos);
    const auto preserved = render(messages, "[]", R"({"preserve_thinking":true})");
    EXPECT_NE(preserved.find("<think>old-private</think>first-answer"), std::string::npos);
    EXPECT_NE(preserved.find("<think>current-private</think>next-answer"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(TargetTokenizers, LfmAgentTemplateTest, ::testing::Values(false, true));
}  // namespace
