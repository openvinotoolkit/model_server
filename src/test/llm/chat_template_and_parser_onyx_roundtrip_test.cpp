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

#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <openvino/genai/tokenizer.hpp>
#pragma GCC diagnostic pop

#include "src/llm/io_processing/output_parser.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

// =============================================================================
// Genuine request/response round trip for Onyx: render a request with the real
// chat template (minja, via ov::genai::Tokenizer::apply_chat_template -- same
// engine exercised by ChatTemplateEndToEndMinjaTest), hand-author the exact
// continuation the model would emit for that rendered prompt (grounded in what
// ChatTemplateEndToEndMinjaTest's Onyx_* tests already proved the template
// produces), and feed ONLY that continuation into OutputParser -- exactly what
// OVMS does in production (the parser only ever sees newly generated tokens,
// never the prompt). This is the missing link between:
//   - chat_template_end_to_end_{minja,jinja}_test.cpp (request side only)
//   - output_parsers/onyx_output_parser_test.cpp (response side only, with
//     hand-written segments not derived from an actual rendered prompt)
// =============================================================================
class OnyxChatTemplateAndParserRoundtripTest : public ::testing::Test {
protected:
    // TODO @atobiszei change tokenizer for onyx
    const std::string& tokenizerPath = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m", false);
    const std::string& chatTemplatesPath = getGenericFullPathForSrcTest("/ovms/src/test/llm/chat_templates", false);

    static std::string loadTemplateFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return "";
        }
        return std::string((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
    }

    // Renders chatHistory with the real Onyx template via minja and asserts the
    // generation prompt ends with the bare "<|start|>assistant" the parser tests
    // assume (no trailing "<|message|>", no implicit recipient).
    std::string renderPrompt(ov::genai::ChatHistory& chatHistory) {
        std::string chatTemplate = loadTemplateFile(chatTemplatesPath + "/chat_template_onyx.jinja");
        EXPECT_FALSE(chatTemplate.empty()) << "Failed to load onyx template";

        ov::genai::Tokenizer tokenizer(tokenizerPath);
        tokenizer.set_chat_template(chatTemplate);
        std::string rendered = tokenizer.apply_chat_template(chatHistory, /*add_generation_prompt=*/true);
        static const std::string generationPromptTail = "<|start|>assistant";
        EXPECT_TRUE(rendered.size() >= generationPromptTail.size() &&
            rendered.compare(rendered.size() - generationPromptTail.size(), generationPromptTail.size(), generationPromptTail) == 0)
            << "Generation prompt tail changed, Onyx parser assumptions may be stale: " << rendered;
        return rendered;
    }

    // Simulates generation: appends modelContinuation to the rendered prompt (for
    // documentation / sanity only) and runs OutputParser on modelContinuation alone,
    // matching what OVMS actually hands the parser.
    ParsedOutput parseModelContinuation(const std::string& modelContinuation, bool toolsAvailable = true) {
        ov::genai::Tokenizer tokenizer(tokenizerPath);
        auto generatedTensor = tokenizer.encode(modelContinuation, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());

        static ToolsSchemas_t emptyToolsSchema{};  // Onyx tool parser is not schema-driven, see onyx_tool_parser.hpp
        OutputParser outputParser(tokenizer, "onyx", "onyx", emptyToolsSchema);
        return outputParser.parse(generatedTokens, toolsAvailable);
    }
};

// =============================================================================
// Turn 1 of the muse/README.md "get_weather" example: user asks a question, the
// prompt is rendered, and the model's tool-call continuation is parsed.
// =============================================================================
TEST_F(OnyxChatTemplateAndParserRoundtripTest, UserQuestion_ModelEmitsToolCall) {
    ov::genai::ChatHistory chatHistory;
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"system","content":"You can call get_weather(city)."})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"Weather in SF?"})"));

    std::string prompt = renderPrompt(chatHistory);
    EXPECT_NE(prompt.find("You can call get_weather(city)."), std::string::npos) << prompt;

    // Model continuation for a tool call, exactly as documented in muse/README.md.
    std::string modelContinuation = R"( to=functions.get_weather<|message|>{"city": "SF"}<|eom|>)";
    ParsedOutput parsedOutput = parseModelContinuation(modelContinuation);

    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"city": "SF"})");
}

// =============================================================================
// Turn 2 of the same example: tool result fed back into history (Onyx's own
// "name" + role="tool" shape, NOT OpenAI's tool_call_id), then the model's final
// answer continuation is parsed.
// =============================================================================
TEST_F(OnyxChatTemplateAndParserRoundtripTest, ToolResultFedBack_ModelEmitsFinalAnswer) {
    ov::genai::ChatHistory chatHistory;
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"system","content":"You can call get_weather(city)."})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"Weather in SF?"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"assistant","recipient":"functions.get_weather","content":"{\"city\": \"SF\"}"})"));
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"tool","name":"functions.get_weather","content":"<tool_output name=\"functions.get_weather\">{\"temp\": 65}</tool_output>"})"));

    std::string prompt = renderPrompt(chatHistory);
    EXPECT_NE(prompt.find(R"(<|start|>assistant to=functions.get_weather<|message|>{"city": "SF"}<|eom|>)"), std::string::npos) << prompt;
    // NOTE (important, non-obvious): OpenVINO GenAI's own minja-path history
    // preprocessing -- independent of the raw Jinja template's `elif role ==
    // 'tool'` branch -- rewrites role="tool" messages into role="user" with a
    // generic wrapped "tool_response" JSON object whenever it determines the
    // template lacks native tool-call support (the same probe underlying
    // caps.supportsToolCalls == false, see chat_template_end_to_end_minja_test.cpp's
    // Onyx tests). So Onyx's own `elif role == 'tool'` template branch is
    // effectively DEAD CODE on the minja path today -- it never actually fires.
    EXPECT_NE(prompt.find(R"(<|start|>user<|message|>{
  "tool_response": {
    "tool": "functions.get_weather",
    "content": "<tool_output name=\"functions.get_weather\">{\"temp\": 65}</tool_output>"
  }
}<|eot|>)"),
        std::string::npos)
        << prompt;

    std::string modelContinuation = R"( to=user<|message|>It's 65F in SF.<|eot|>)";
    ParsedOutput parsedOutput = parseModelContinuation(modelContinuation);

    EXPECT_EQ(parsedOutput.content, "It's 65F in SF.");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

// =============================================================================
// Private reasoning ("recipient": "self") round trip: the model reasons first,
// which the OnyxReasoningParser must classify as reasoning, not content.
// =============================================================================
TEST_F(OnyxChatTemplateAndParserRoundtripTest, UserQuestion_ModelEmitsPrivateReasoning) {
    ov::genai::ChatHistory chatHistory;
    chatHistory.push_back(ov::genai::JsonContainer::from_json_string(
        R"({"role":"user","content":"What's 2+2?"})"));

    std::string prompt = renderPrompt(chatHistory);
    EXPECT_NE(prompt.find("What's 2+2?"), std::string::npos) << prompt;

    std::string modelContinuation = R"( to=self<|message|>2+2 is a basic addition.<|eom|>)";
    ParsedOutput parsedOutput = parseModelContinuation(modelContinuation);

    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "2+2 is a basic addition.");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}
