// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <rapidjson/document.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include "src/llm/io_processing/lfm2/python_literal.hpp"
#ifndef LFM_TEMPLATE_STANDALONE
#include "src/test/platform_utils.hpp"
#endif

namespace {
class LfmTemplateRoundTripTest : public ::testing::TestWithParam<bool> {};
TEST_P(LfmTemplateRoundTripTest, MinjaRealTokenizerPreservesNestedToolValues) {
#if defined(LFM_TEMPLATE_STANDALONE)
    const auto root = std::filesystem::current_path().string();
#elif defined(_WIN32)
    const auto root = getWindowsRepoRootPath();
#else
    const std::string root = "/ovms";
#endif
    const std::string model = GetParam() ? "LFM2.5-8B-A1B" : "LFM2-2.6B";
    const std::string fixture = root + "/src/test/llm_testing/LiquidAI/" + model;
    ov::genai::Tokenizer tokenizer(fixture);
#ifdef LFM_TEMPLATE_ORIGINAL
    const std::string path = fixture + "/chat_template.jinja";
#else
    const std::string path = root + "/docs/lfm/templates/" + (GetParam() ? "LFM2.5-8B-A1B-int4-ov.jinja" : "LFM2-24B-A2B-int4-ov.jinja");
#endif
    std::ifstream file(path);
    ASSERT_TRUE(file.good()) << path;
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    ov::genai::ChatHistory history;
    history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"user","content":"Report a value."})"));
    history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"assistant","content":"","thinking":"private","tool_calls":[{"id":"test_call","type":"function","function":{"name":"report_value","arguments":{"text":"O'Brien at C:\\test\nnext","items":[true,null,{"n":42}]}}}]})"));
    const auto rendered = tokenizer.apply_chat_template(history, true, source);
    const std::string prefix = "report_value(";
    const auto begin = rendered.find(prefix);
    ASSERT_NE(begin, std::string::npos) << rendered;
    const auto end = rendered.rfind(")]<|tool_call_end|>");
    ASSERT_NE(end, std::string::npos) << rendered;
    const auto args = rendered.substr(begin + prefix.size(), end - begin - prefix.size());
    std::string parsed;
    ASSERT_NO_THROW(parsed = ovms::lfm2::parseArguments(args)) << rendered;
    rapidjson::Document actual, expected;
    actual.Parse(parsed.c_str());
    expected.Parse(R"({"text":"O'Brien at C:\\test\nnext","items":[true,null,{"n":42}]})");
    ASSERT_FALSE(actual.HasParseError());
    EXPECT_EQ(actual, expected);
    if (GetParam()) EXPECT_NE(rendered.find("<think>private</think>"), std::string::npos);
    history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"tool","tool_call_id":"test_call","content":"TOOL_OK"})"));
    const auto continuation = tokenizer.apply_chat_template(history, true, source);
    EXPECT_NE(continuation.find("TOOL_OK"), std::string::npos);
    EXPECT_NE(continuation.rfind("<|im_start|>assistant\n"), std::string::npos);
}
INSTANTIATE_TEST_SUITE_P(TargetTokenizers, LfmTemplateRoundTripTest, ::testing::Values(false, true));
}  // namespace
