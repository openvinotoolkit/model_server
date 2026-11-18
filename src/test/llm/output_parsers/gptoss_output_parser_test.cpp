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
#include <openvino/genai/tokenizer.hpp>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../../llm/io_processing/base_output_parser.hpp"
#include "../../../llm/apis/openai_completions.hpp"
#include "../../../llm/io_processing/generation_config_builder.hpp"
#include "../../../llm/io_processing/gptoss/generation_context.hpp"
#include "../../../llm/io_processing/gptoss/content_parser.hpp"
#include "../../../llm/io_processing/gptoss/reasoning_parser.hpp"
#include "../../../llm/io_processing/gptoss/tool_parser.hpp"
#include "../../../llm/io_processing/output_parser.hpp"
#include "../../../llm/io_processing/gptoss/harmony.hpp"
#include "output_parser_test_utils.hpp"
#include "../../platform_utils.hpp"

using namespace ovms;
using namespace ovms::openai;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\OpenVINO\\gpt-oss-20b-int4-ov";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/OpenVINO/gpt-oss-20b-int4-ov";
#endif

static std::unique_ptr<ov::genai::Tokenizer> gptOssTokenizer;
static const ToolsSchemas_t& EMPTY_TOOLS_SCHEMA = {};  // not used in gptoss

static std::vector<int64_t> getTokens(const std::string& text) {
    ov::Tensor t = gptOssTokenizer->encode(text).input_ids;
    const int64_t* data_ptr = t.data<const int64_t>();
    size_t length = t.get_shape()[1];  // assuming shape is [1, seq_len]
    return std::vector<int64_t>(data_ptr, data_ptr + length);
}

class TokenBuilder {
public:
    TokenBuilder& add(const std::string& text) {
        auto tokens = getTokens(text);
        tokenIDs.insert(tokenIDs.end(), tokens.begin(), tokens.end());
        return *this;
    }

    TokenBuilder& add(int64_t tokenID) {
        tokenIDs.push_back(tokenID);
        return *this;
    }

    TokenBuilder& add(Harmony::TokenID tokenID) {
        tokenIDs.push_back(static_cast<int64_t>(tokenID));
        return *this;
    }

    std::vector<int64_t> build() {
        return tokenIDs;
    }

    TokenBuilder& clear() {
        tokenIDs.clear();
        return *this;
    }

private:
    std::vector<int64_t> tokenIDs;
};

TEST(GptOssToolGuidanceConfigTest, PromptHeaderDetectionIgnoresCompletedMessages) {
    const auto headerFor = [](const std::string& prompt) {
        const auto modelContext = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), prompt);
        return GptOssGenerationContextAdapter::adapt(modelContext).header;
    };
    using Header = GptOssGenerationContext::Header;
    EXPECT_EQ(headerFor("history<|end|><|start|>assistant"), Header::Assistant);
    EXPECT_EQ(headerFor("<|start|>assistant to=functions."), Header::ToolRecipient);
    EXPECT_EQ(headerFor("<|start|>assistant<|channel|>analysis to=functions."), Header::ToolRecipient);
    EXPECT_EQ(headerFor(""), Header::None);
    EXPECT_EQ(headerFor("<|start|>user<|message|>hi"), Header::None);
    EXPECT_EQ(headerFor("<|start|>assistant<|channel|>analysis<|message|>thinking"), Header::None);
    EXPECT_EQ(headerFor("<|start|>assistant<|message|>answer<|return|>"), Header::None);
    const std::string partialRecipientPrompt = "<|start|>assistant to=functions.ec";
    const auto partialRecipientContext = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), partialRecipientPrompt);
    EXPECT_EQ(GptOssGenerationContextAdapter::adapt(partialRecipientContext).partialToolName, "ec");
    const std::string channelRecipientPrompt = "<|start|>assistant<|channel|>commentary to=functions.";
    const auto channelRecipientContext = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), channelRecipientPrompt);
    const auto channelRecipient = GptOssGenerationContextAdapter::adapt(channelRecipientContext);
    EXPECT_TRUE(channelRecipient.recipientAfterChannel);
    EXPECT_EQ(channelRecipient.header, Header::ToolRecipient);
}

TEST(GptOssToolGuidanceConfigTest, CombinedContextIncludesOnlyApplicablePromptState) {
    const std::string prompt = "<|start|>assistant<|channel|>analysis";
    const auto harmony = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), prompt);
    const auto gptOss = GptOssGenerationContextAdapter::adapt(harmony);
    EXPECT_EQ(gptOss.header, GptOssGenerationContext::Header::Channel);
    EXPECT_EQ(gptOss.channel, "analysis");
    EXPECT_EQ(harmony.reasoning.begin, "<|channel|>analysis<|message|>");
    EXPECT_FALSE(harmony.reasoning.implicitStart);
    const auto qwen = determineModelGenerationContext(getDefaultReasoningParsingConfig("qwen3"), prompt);
    EXPECT_EQ(qwen.reasoning.begin, "<think>");
    EXPECT_FALSE(qwen.reasoning.implicitStart);
}

TEST(GptOssToolGuidanceConfigTest, MessageSequenceMatchesToolChoice) {
    using SOC = ov::genai::StructuredOutputConfig;
    for (const ToolChoiceMode choice : {ToolChoiceMode::None, ToolChoiceMode::Auto, ToolChoiceMode::Required, ToolChoiceMode::NamedFunction}) {
        for (const bool enabled : {false, true}) {
            SCOPED_TRACE(static_cast<int>(choice));
            SCOPED_TRACE(enabled);
            const bool namedChoice = choice == ToolChoiceMode::NamedFunction;
            const bool requiresCall = namedChoice || choice == ToolChoiceMode::Required;
            OpenAIRequest request;
            request.toolChoice = choice;
            if (namedChoice) {
                request.toolChoiceFunctionName = "echo";
            }
            const std::string schema = R"({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false})";
            if (choice != ToolChoiceMode::None) {
                request.toolNameSchemaMap.emplace("echo", ToolSchemaWrapper{nullptr, schema});
                if (!namedChoice) {
                    request.toolNameSchemaMap.emplace("other", ToolSchemaWrapper{nullptr, schema});
                }
            }
            GenerationConfigBuilder configBuilder{{}, "gptoss", enabled, DecodingMethod::STANDARD};
            configBuilder.configureStructuredGeneration(request, {}, "gptoss");
            const auto& config = configBuilder.getConfig();
            ASSERT_EQ(config.structured_output_config.has_value(), choice != ToolChoiceMode::None && (enabled || requiresCall));
            if (!config.structured_output_config) {
                continue;
            }
            const auto& tag = std::get<SOC::StructuralTag>(*config.structured_output_config->structural_tags_config);
            const auto initialMessages = std::get<std::shared_ptr<SOC::Union>>(tag);
            ASSERT_FALSE(initialMessages->elements.empty());
            bool foundTool = false;
            bool foundReasoning = false;
            bool foundFinal = false;
            for (const auto& branch : initialMessages->elements) {
                const auto sequence = std::get<std::shared_ptr<SOC::Concat>>(branch);
                if (const auto* finalAnswer = std::get_if<std::shared_ptr<SOC::TagsWithSeparator>>(&sequence->elements.front())) {
                    foundFinal = true;
                    EXPECT_FALSE(requiresCall);
                    EXPECT_EQ(sequence->elements.size(), 1);
                    EXPECT_TRUE((*finalAnswer)->at_least_one);
                    EXPECT_TRUE((*finalAnswer)->stop_after_first);
                    ASSERT_FALSE((*finalAnswer)->tags.empty());
                    for (const auto& message : (*finalAnswer)->tags) {
                        EXPECT_EQ(message.end, "<|return|>");
                        EXPECT_TRUE(std::holds_alternative<SOC::AnyText>(message.content));
                    }
                    continue;
                }
                const auto firstMessages = std::get<std::shared_ptr<SOC::Union>>(sequence->elements.front());
                ASSERT_FALSE(firstMessages->elements.empty());
                for (const auto& firstMessage : firstMessages->elements) {
                    const auto first = std::get<std::shared_ptr<SOC::Tag>>(firstMessage);
                    EXPECT_EQ(first->begin.find("<|start|>assistant"), 0);
                    if (first->end == "<|call|>") {
                        foundTool = true;
                        EXPECT_NE(first->begin.find("to=functions."), std::string::npos);
                        if (namedChoice) {
                            EXPECT_NE(first->begin.find("functions.echo"), std::string::npos);
                        }
                        const auto body = std::get<std::shared_ptr<SOC::Concat>>(first->content);
                        EXPECT_EQ(std::get<SOC::JSONSchema>(body->elements.back()).value, schema);
                        const auto afterCall = std::get<std::shared_ptr<SOC::TagsWithSeparator>>(sequence->elements.back());
                        bool allowsAnotherCall = false;
                        for (const auto& message : afterCall->tags) {
                            allowsAnotherCall = allowsAnotherCall || message.end == "<|call|>";
                        }
                        EXPECT_EQ(allowsAnotherCall, !namedChoice);
                    } else if (first->end == "<|end|>") {
                        foundReasoning = true;
                        EXPECT_TRUE(std::holds_alternative<SOC::AnyText>(first->content));
                        if (requiresCall) {
                            const auto remaining = std::get<std::shared_ptr<SOC::Concat>>(sequence->elements.back());
                            ASSERT_EQ(remaining->elements.size(), 3);
                            EXPECT_TRUE(std::holds_alternative<SOC::EBNF>(remaining->elements.front()));
                            const auto mandatoryCalls = std::get<std::shared_ptr<SOC::Union>>(remaining->elements[1]);
                            ASSERT_FALSE(mandatoryCalls->elements.empty());
                            for (const auto& call : mandatoryCalls->elements) {
                                EXPECT_EQ(std::get<std::shared_ptr<SOC::Tag>>(call)->end, "<|call|>");
                            }
                        }
                    } else {
                        foundFinal = true;
                        EXPECT_EQ(first->end, "<|return|>");
                        EXPECT_FALSE(requiresCall);
                    }
                }
            }
            EXPECT_TRUE(foundTool);
            EXPECT_TRUE(foundReasoning);
            EXPECT_EQ(foundFinal, !requiresCall);
        }
    }
}

class GptOssOutputUnaryParserTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            gptOssTokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath, ov::AnyMap{ov::genai::skip_special_tokens(false)});
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize gptOss tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize gptOss tokenizer due to unknown error.";
        }
    }

protected:
    void pushTokens(const std::string& text, std::vector<int64_t>& tt) {
        auto tokens = getTokens(text);
        tt.insert(tt.end(), tokens.begin(), tokens.end());
    }

    static void TearDownTestSuite() {
        gptOssTokenizer.reset();
    }

    void SetUp() override {
    }

    void assertParseIgnoredNoResults(Harmony& harmony) {
        ASSERT_TRUE(harmony.parse());
        ASSERT_EQ(harmony.getContent(), "");
        ASSERT_EQ(harmony.getReasoning(), "");
        ASSERT_EQ(harmony.getToolCalls().size(), 0);
    }

    TokenBuilder builder;
};

//
//
// Unary
//
//

TEST_F(GptOssOutputUnaryParserTest, ConstructorsEstablishPromptContinuationState) {
    const std::string prompt = "<|start|>assistant<|channel|>analysis";
    const auto context = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), prompt);
    std::vector<std::unique_ptr<BaseOutputParser>> parsers;
    parsers.emplace_back(std::make_unique<GptOssToolParser>(*gptOssTokenizer, std::nullopt, context));
    parsers.emplace_back(std::make_unique<GptOssReasoningParser>(*gptOssTokenizer, std::nullopt, context));
    parsers.emplace_back(std::make_unique<GptOssContentParser>(*gptOssTokenizer, std::nullopt, context));
    const std::vector<std::string> toolTags{" to=functions."};
    const std::vector<std::string> reasoningTags{"<|message|>"};
    const std::vector<std::string> noTags;
    for (auto& parser : parsers) {
        const auto initialTags = parser->getParsingConfig().promptContinuationStartTags;
        parser->resetState();
        EXPECT_EQ(parser->getParsingConfig().promptContinuationStartTags, initialTags);
    }
    EXPECT_EQ(parsers[0]->getParsingConfig().promptContinuationStartTags, toolTags);
    EXPECT_EQ(parsers[1]->getParsingConfig().promptContinuationStartTags, reasoningTags);
    EXPECT_EQ(parsers[2]->getParsingConfig().promptContinuationStartTags, noTags);
    parsers.clear();
    parsers.emplace_back(std::make_unique<GptOssToolParser>(*gptOssTokenizer, std::nullopt, ModelGenerationContext{}));
    parsers.emplace_back(std::make_unique<GptOssReasoningParser>(*gptOssTokenizer, std::nullopt, ModelGenerationContext{}));
    parsers.emplace_back(std::make_unique<GptOssContentParser>(*gptOssTokenizer, std::nullopt, ModelGenerationContext{}));
    for (auto& parser : parsers) {
        EXPECT_TRUE(parser->getParsingConfig().promptContinuationStartTags.empty());
        EXPECT_FALSE(parser->getParsingConfig().startsInsidePhase);
    }
}

TEST_F(GptOssOutputUnaryParserTest, InitializedToolRecipientSurvivesResetWithoutPromptStorage) {
    std::unique_ptr<BaseOutputParser> parser;
    {
        const std::string prompt = "<|start|>assistant to=functions.ec";
        const auto context = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), prompt);
        parser = std::make_unique<GptOssToolParser>(*gptOssTokenizer, std::nullopt, context);
    }
    BaseOutputParser& baseParser = *parser;
    ASSERT_TRUE(baseParser.getParsingConfig().startsInsidePhase);
    for (const bool reset : {false, true}) {
        if (reset) {
            baseParser.resetState();
        }
        EXPECT_FALSE(baseParser.parseChunk("ho", {}, ov::genai::GenerationFinishReason::NONE).has_value());
        const auto delta = baseParser.parseChunk("<|message|>", {}, ov::genai::GenerationFinishReason::NONE);
        ASSERT_TRUE(delta.has_value());
        const auto* call = std::get_if<ToolCallDelta>(&*delta);
        ASSERT_NE(call, nullptr);
        ASSERT_TRUE(call->name.has_value());
        EXPECT_EQ(*call->name, "echo");
        EXPECT_EQ(call->index, 0);
    }
}

TEST_F(GptOssOutputUnaryParserTest, ToolGuidanceHandlesPrefilledHeadersAndReasoning) {
    for (const std::string header : {"", "<|start|>assistant", "<|start|>assistant<|channel|>analysis",
             "<|start|>assistant to=functions.", "<|start|>assistant<|channel|>commentary to=functions.",
             "<|start|>assistant<|channel|>analysis to=functions."}) {
        for (const ToolChoiceMode choice : {ToolChoiceMode::Auto, ToolChoiceMode::Required, ToolChoiceMode::NamedFunction}) {
            SCOPED_TRACE(header);
            SCOPED_TRACE(static_cast<int>(choice));
            const bool namedChoice = choice == ToolChoiceMode::NamedFunction;
            const std::string choiceJson = namedChoice                          ? R"({"type":"function","function":{"name":"echo"}})"
                                           : choice == ToolChoiceMode::Required ? R"("required")"
                                                                                : R"("auto")";
            rapidjson::Document request;
            const std::string json = R"({"model":"gptoss","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"echo","parameters":{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}}}],"tool_choice":)" + choiceJson + "}";
            request.Parse(json.c_str());
            ASSERT_FALSE(request.HasParseError());
            OpenAIChatCompletionsHandler handler(request, Endpoint::CHAT_COMPLETIONS,
                std::chrono::system_clock::now(), *gptOssTokenizer, "gptoss", "gptoss");
            ASSERT_TRUE(handler.parseRequest(std::nullopt, 0, std::nullopt).ok());
            const auto context = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), header);
            const auto gptOssContext = GptOssGenerationContextAdapter::adapt(context);
            GenerationConfigBuilder configBuilder{{}, "gptoss", choice == ToolChoiceMode::Auto, DecodingMethod::STANDARD};
            configBuilder.build(handler.getRequest(), context, "gptoss", *gptOssTokenizer);
            ASSERT_TRUE(configBuilder.getConfig().structured_output_config.has_value());
            handler.initOutputParser(context);
            handler.getOutputParser()->resetStreamingState();
            const std::string fullCall = "<|start|>assistant to=functions.echo <|channel|>commentary json<|message|>{\"value\":1}<|call|>";
            const bool toolHeader = gptOssContext.header == GptOssGenerationContext::Header::ToolRecipient;
            const bool assistantHeaderToolCall = header == "<|start|>assistant";
            std::string output;
            if (toolHeader) {
                output = header.find("<|channel|>") == std::string::npos
                             ? "echo <|channel|>commentary json<|message|>{\"value\":1}<|call|>"
                             : "echo<|message|>{\"value\":1}<|call|>";
            } else if (assistantHeaderToolCall) {
                output = fullCall.substr(header.size());
            } else {
                const std::string reasoningHeader = "<|start|>assistant<|channel|>analysis<|message|>";
                output = reasoningHeader.substr(header.size()) + "thinking<|end|>" + fullCall;
            }
            if (choice == ToolChoiceMode::Required) {
                output += fullCall;
            }
            const auto encoded = gptOssTokenizer->encode(output, ov::genai::add_special_tokens(false)).input_ids;
            const std::vector<int64_t> tokens(encoded.data<int64_t>(), encoded.data<int64_t>() + encoded.get_size());
            std::vector<Delta> deltas;
            OVMSTextStreamer streamer(*gptOssTokenizer, handler.getOutputParser(), handler.areToolsAvailable(),
                [&deltas](Delta delta, bool) {
                    deltas.push_back(std::move(delta));
                    return ov::genai::StreamingStatus::RUNNING;
                },
                {});
            streamer.write(tokens);
            streamer.end();
            ASSERT_FALSE(streamer.hadParserError()) << streamer.parserErrorMessage();
            rapidjson::Document response;
            response.Parse(handler.serializeUnaryResponse(deltas, ov::genai::GenerationFinishReason::STOP).c_str());
            ASSERT_FALSE(response.HasParseError());
            const auto& message = response["choices"][0]["message"];
            if (!toolHeader && !assistantHeaderToolCall) {
                EXPECT_STREQ(message["reasoning_content"].GetString(), "thinking");
            }
            const auto& calls = message["tool_calls"];
            ASSERT_EQ(calls.Size(), choice == ToolChoiceMode::Required ? 2 : 1);
            for (const auto& call : calls.GetArray()) {
                EXPECT_STREQ(call["function"]["name"].GetString(), "echo");
                EXPECT_STREQ(call["function"]["arguments"].GetString(), R"({"value":1})");
            }
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, InvalidToolGrammarKeepsUnconstrainedFallback) {
    OpenAIRequest request;
    request.toolChoice = ToolChoiceMode::Required;
    request.toolNameSchemaMap.emplace("echo", ToolSchemaWrapper{nullptr,
                                                  R"({"type":"object","properties":{"value":{"type":"invalid_type"}},"required":["value"]})"});
    GenerationConfigBuilder configBuilder{{}, "gptoss", false, DecodingMethod::STANDARD};
    const std::string prompt = "<|start|>assistant";
    const auto context = determineModelGenerationContext(getDefaultReasoningParsingConfig("gptoss"), prompt);
    configBuilder.build(request, context, "gptoss", *gptOssTokenizer);
    EXPECT_FALSE(configBuilder.getConfig().structured_output_config.has_value());
}

TEST_F(GptOssOutputUnaryParserTest, SimpleContent) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)  // <|channel|>
            .add("final")
            .add(Harmony::TokenID::MESSAGE)  // <|message|>
            .add("Hello, world!")
            .add(closureToken);  // <|end|> or <|return|> or <|call|>
        Harmony harmony(*gptOssTokenizer, builder.build());
        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "Hello, world!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

TEST_F(GptOssOutputUnaryParserTest, NegativeFinalChannel) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto wrongChannel : std::vector<std::string>{
                 "finalextra",  // finalextra is not final
                 "Final",       // case sensitive
                 " finale",     // leading space
                 "final ",      // trailing space
                 " final",      // trailing space
                 "fi nal",      // space inside
                 ""}) {         // empty channel
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(wrongChannel)
                .add(Harmony::TokenID::MESSAGE)
                .add("Hello, world!")
                .add(closureToken);
            Harmony harmony(*gptOssTokenizer, builder.build());
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, PreambleOnly) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary")
            .add(Harmony::TokenID::MESSAGE)
            .add("Hello, world!")
            .add(closureToken);
        Harmony harmony(*gptOssTokenizer, builder.build());
        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "Hello, world!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

TEST_F(GptOssOutputUnaryParserTest, NegativePreamble) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto wrongChannel : std::vector<std::string>{
                 "commentary ",
                 " commentary",
                 " commentary ",
                 "comment ary",  // space inside
                 "commenTary",   // case sensitive
                 ""}) {
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(wrongChannel)
                .add(Harmony::TokenID::MESSAGE)
                .add("Hello, world!")
                .add(closureToken);
            Harmony harmony(*gptOssTokenizer, builder.build());
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, ReasoningOnly) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)
            .add("analysis")
            .add(Harmony::TokenID::MESSAGE)
            .add("Hello, world!")
            .add(closureToken);
        Harmony harmony(*gptOssTokenizer, builder.build());
        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "Hello, world!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

TEST_F(GptOssOutputUnaryParserTest, NegativeReasoning) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto wrongChannel : std::vector<std::string>{
                 "analysis ",
                 " analysis ",
                 "analy sis",  // space inside
                 "analYsis",   // case sensitive
                 ""}) {
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(wrongChannel)
                .add(Harmony::TokenID::MESSAGE)
                .add("Hello, world!")
                .add(closureToken);
            Harmony harmony(*gptOssTokenizer, builder.build());
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, SingleToolCallWithConstrain) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto functionDeclaration : std::vector<std::string>{
                 "commentary to=functions.hello",  // valid channel with to=
                 "commentary to=functions.hello ",
                 "commentary   to=functions.hello",
                 "commentary  ANYTHING IN BETWEEN to=functions.hello",
             }) {  // spaces after hello
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(functionDeclaration)
                .add(Harmony::TokenID::MESSAGE)
                .add(R"({"Hello": "world!"})")
                .add(closureToken);
            Harmony harmony(*gptOssTokenizer, builder.build());
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls().size(), 1) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls()[0].name, "hello") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls()[0].arguments, R"({"Hello": "world!"})") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
        }
    }
}

// This does not conform to OpenAI Harmony format, however tests show that sometimes model produces tools in analysis channel
TEST_F(GptOssOutputUnaryParserTest, SingleToolCallWithConstrainInAnalysisChannel) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto functionDeclaration : std::vector<std::string>{
                 "analysis to=functions.hello",  // valid channel with to=
                 "analysis to=functions.hello ",
                 "analysis   to=functions.hello",
                 "analysis  ANYTHING IN BETWEEN to=functions.hello",
             }) {  // spaces after hello
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(functionDeclaration)
                .add(Harmony::TokenID::MESSAGE)
                .add(R"({"Hello": "world!"})")
                .add(closureToken);
            Harmony harmony(*gptOssTokenizer, builder.build());
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls().size(), 1) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls()[0].name, "hello") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls()[0].arguments, R"({"Hello": "world!"})") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, InvalidSingleToolCallWithConstrain) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto functionDeclaration : std::vector<std::string>{
                 "commentary to = functions.hello",
                 "commentary to= functions.hello ",
                 "commentary functions.hello",
                 "commentary to=hello",
                 "commentary hello"}) {
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(functionDeclaration)
                .add(Harmony::TokenID::MESSAGE)
                .add(R"({"Hello": "world!"})")
                .add(closureToken);
            Harmony harmony(*gptOssTokenizer, builder.build());
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls().size(), 0) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, HolisticMultiTurn) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        // In regular scenarios it is never that complicated. But we test the parser, so why not.
        // Usually the order is as follows:
        // - Analysis (reasoning)
        // - (optional) commentary (preamble, counts as final content as well)
        // - (optional, multiple) commentary to=functions.* + constrain json (tool calls)
        // - final (content)
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)
            .add("analysis")
            .add(Harmony::TokenID::MESSAGE)
            .add("I need to call a function.")
            .add(closureToken)
            // With constrain, but ignored anyway
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary to=functions.hello")  // strict
            .add(Harmony::TokenID::CONSTRAIN)
            .add("json")
            .add(Harmony::TokenID::MESSAGE)
            .add(R"({"Hello": "world!"})")
            .add(closureToken)
            .add(Harmony::TokenID::CHANNEL)
            .add("final")
            .add(Harmony::TokenID::MESSAGE)
            .add("Dear User, I called function!")
            .add(closureToken)
            // Without constrain, it is ignored anyway
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary ? to=functions.goodbye ")  // with space and anything in the middle
            .add(Harmony::TokenID::MESSAGE)
            .add("NOT A JSON")
            .add(closureToken)
            // Preamble
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary")
            .add(Harmony::TokenID::MESSAGE)
            .add("I called some functions. Will summarize now.")
            .add(closureToken)
            // Final v2
            .add(Harmony::TokenID::CHANNEL)
            .add("final")
            .add(Harmony::TokenID::MESSAGE)
            .add("Dear User, I called second function!")
            .add(closureToken);
        Harmony harmony(*gptOssTokenizer, builder.build());
        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "Dear User, I called function! I called some functions. Will summarize now. Dear User, I called second function!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "I need to call a function.") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls().size(), 2) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[0].name, "hello") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[0].arguments, R"({"Hello": "world!"})") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[1].name, "goodbye") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[1].arguments, "NOT A JSON") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

// Negative
TEST_F(GptOssOutputUnaryParserTest, MissingChannel) {
    builder
        .clear()
        // .add(Harmony::TokenID::CHANNEL)  // no channel
        .add("commentary to=functions.hello")
        .add(Harmony::TokenID::MESSAGE)
        .add(R"({"Hello": "world!"})")
        .add(Harmony::TokenID::END);
    Harmony harmony(*gptOssTokenizer, builder.build());
    assertParseIgnoredNoResults(harmony);
}

TEST_F(GptOssOutputUnaryParserTest, MissingMessageTag) {
    builder
        .clear()
        .add(Harmony::TokenID::CHANNEL)
        .add("commentary to=functions.hello")
        //  .add(Harmony::TokenID::MESSAGE)  // no message tag
        .add(R"({"Hello": "world!"})")
        .add(Harmony::TokenID::END);
    Harmony harmony(*gptOssTokenizer, builder.build());
    assertParseIgnoredNoResults(harmony);
}

TEST_F(GptOssOutputUnaryParserTest, MissingEndTag) {
    builder
        .clear()
        .add(Harmony::TokenID::CHANNEL)
        .add("commentary to=functions.hello")
        .add(Harmony::TokenID::MESSAGE)
        .add(R"({"Hello": "world!"})");
    // .add(Harmony::TokenID::END);  // no end tag
    Harmony harmony(*gptOssTokenizer, builder.build());
    assertParseIgnoredNoResults(harmony);
}

//
//
// Streaming
//
//

class GptOssOutputStreamParserTest : public GptOssOutputUnaryParserTest {
protected:
    std::unique_ptr<OutputParser> outputParser;

    void SetUp() override {
        GptOssOutputUnaryParserTest::SetUp();
        outputParser = std::make_unique<OutputParser>(*gptOssTokenizer, "gptoss", "gptoss", EMPTY_TOOLS_SCHEMA, ovms::ModelGenerationContext{});
    }

    void test(const std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>>& chunkToDeltaVec) {
        // Need to have new output parser per case to simulate separate request processing
        outputParser = std::make_unique<OutputParser>(*gptOssTokenizer, "gptoss", "gptoss", EMPTY_TOOLS_SCHEMA, ovms::ModelGenerationContext{});
        auto chunkToDeltaVecCopy = chunkToDeltaVec;
        int64_t chunkIteration = -1;
        for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVecCopy) {
            chunkIteration++;
            std::optional<ovms::Delta> doc = outputParser->parseChunk(chunk, {}, true, finishReason);
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
                    EXPECT_EQ(docStr, expected) << "Mismatch for chunk: [" << chunk << "] got [" << docStr << "] but expected [" << expected << "]" << chunkIteration;
                }
            } else if (expectedDelta.has_value()) {
                FAIL() << "Mismatch for chunk: [" << chunk << "] got nothing but expected [" << expectedDelta.value() << "]" << chunkIteration;
            } else if (doc.has_value()) {
                std::string docStr = ovms::test::deltaToJson(*doc);
                FAIL() << "Mismatch for chunk: [" << chunk << "] expected nothing but got [" << docStr << "]" << chunkIteration;
            } else {
                FAIL() << "Mismatch for chunk: [" << chunk << "] " << chunkIteration;
            }
        }
    }
};

TEST_F(GptOssOutputStreamParserTest, HolisticStreamingReasoning) {
    for (const auto& stopToken : std::vector<std::string>{"<|end|>", "<|return|>"}) {
        std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
            // Reasoning
            {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"I"}})"}},
            {" am", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" am"}})"}},
            {" reaso", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" reaso"}})"}},
            {"ning.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"ning."}})"}},
            {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            // Preamble
            {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"I"}})"}},
            {" am", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" am"}})"}},
            {" producing", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" producing"}})"}},
            {" preamble", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" preamble"}})"}},
            {".", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"."}})"}},
            {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            // Final content
            {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"final", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"Dear", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"Dear"}})"}},
            {" User,", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" User,"}})"}},
            {" I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" I"}})"}},
            {" reason!", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" reason!"}})"}},
            {stopToken, ov::genai::GenerationFinishReason::NONE, std::nullopt},
        };
        test(chunkToDeltaVec);
    }
}

TEST_F(GptOssOutputStreamParserTest, RepeatedAnalysisSegmentWithAssistantPrefix) {
    test({
        {"<|channel|>analysis<|message|>first<|end|>", ov::genai::GenerationFinishReason::NONE,
            {R"({"delta":{"reasoning_content":"first"}})"}},
        {"<|start|>assistant<|channel|>analysis<|message|>second<|end|>", ov::genai::GenerationFinishReason::NONE,
            {R"({"delta":{"reasoning_content":"second"}})"}},
    });
}

TEST_F(GptOssOutputStreamParserTest, ToolCallInAssistantRoleHeaderWithoutChannel) {
    test({
        {"<|start|>assistant to=functions.echo ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE,
            {R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"echo"}}]}})"}},
        {"json", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{\"value\":1}", ov::genai::GenerationFinishReason::NONE,
            {R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"value\":1}"}}]}})"}},
        {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    });
}

TEST_F(GptOssOutputStreamParserTest, CoalescedToolCallEmitsNameOnly) {
    for (const auto& metadata : std::vector<std::string>{"", "<|constrain|>json"}) {
        for (const auto& terminator : std::vector<std::string>{"", "<|call|>"}) {
            test({
                {"<|start|>assistant to=functions.echo " + metadata + "<|message|>{\"value\":1}" + terminator,
                    ov::genai::GenerationFinishReason::NONE,
                    {R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"echo"}}]}})"}},
            });
        }
    }
}

TEST_F(GptOssOutputStreamParserTest, ToolCallSurvivesGarbledConstrainMetadata) {
    for (const auto& constrainMetadata : std::vector<std::string>{"wcommentary json", "??commentary json"}) {
        test({
            {"<|start|>assistant to=functions.echo ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"<|constrain|>", ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"echo"}}]}})"}},
            {constrainMetadata, ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"{\"value\":1}", ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"value\":1}"}}]}})"}},
            {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        });
    }
}

TEST_F(GptOssOutputStreamParserTest, ToolResponseIsDiscardedBeforeFinalMessage) {
    for (const auto& callTerminator : std::vector<std::string>{"<|call|>", "<|end|>", "<|return|>"}) {
        test({
            {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"commentary to=functions.echo ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"<|message|>", ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"echo"}}]}})"}},
            {"{}" + callTerminator, ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}})"}},
            {"<|start|>functions.echo to=assistant<|message|>tool result<|end|><|start|>assistant<|channel|>final<|message|>do",
                ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"content":"do"}})"}},
            {"ne<|end|>", ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"content":"ne"}})"}},
        });
    }
}

TEST_F(GptOssOutputStreamParserTest, AssistantCommentaryAndFinalAfterToolResponse) {
    test({
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary to=functions.echo ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE,
            {R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"echo"}}]}})"}},
        {"{}<|call|>", ov::genai::GenerationFinishReason::NONE,
            {R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}})"}},
        {"<|start|>functions.echo to=assistant<|message|>tool result<|end|><|start|>assistant<|channel|>commentary<|message|>Working.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|end|>",
            ov::genai::GenerationFinishReason::NONE,
            {R"({"delta":{"content":"Working.Done."}})"}},
    });
}

TEST_F(GptOssOutputStreamParserTest, ToolTerminatorPreservesPostCallSuffix) {
    for (const auto& callTerminator : std::vector<std::string>{"<|call|>", "<|end|>", "<|return|>"}) {
        test({
            {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"commentary to=functions.echo ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
            {"<|message|>", ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"echo"}}]}})"}},
            {"{}" + callTerminator + "<|start|>functions.echo to=assistant<|message|>tool result<|end|><|start|>assistant<|channel|>final<|message|>done<|end|>",
                ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}})"}},
            {"", ov::genai::GenerationFinishReason::NONE,
                {R"({"delta":{"content":"done"}})"}},
        });
    }
}

TEST_F(GptOssOutputStreamParserTest, HolisticStreamingTools) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Reasoning
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"I"}})"}},
        {" will", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" will"}})"}},
        {" call", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" call"}})"}},
        {" fun", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" fun"}})"}},
        {"ction.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"ction."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Preamble
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"I"}})"}},
        {" have", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" have"}})"}},
        {" to", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" to"}})"}},
        {" call", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" call"}})"}},
        {" fun", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" fun"}})"}},
        {"ction.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"ction."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool 1
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".hello ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"hello\"}}]}}"},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" {\""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \""}}]}})"},
        {"Paris", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris"}}]}})"},
        {"\"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}})"},
        {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool 2 (with ignored constrain)
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".world ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":1,\"function\":{\"name\":\"world\"}}]}}"},
        {"json", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" {\""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" \""}}]}})"},
        {"Warsaw", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"Warsaw"}}]}})"},
        {"\"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"}"}}]}})"},
        {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };
    test(chunkToDeltaVec);
}

TEST_F(GptOssOutputStreamParserTest, ConstrainAndCallGluedWithToolCall) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Reasoning
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"I"}})"}},
        {" will", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" will"}})"}},
        {" call", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" call"}})"}},
        {" fun", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" fun"}})"}},
        {"ction.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"ction."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},

        // Tool 1 (with ignored constrain, but it is glued to tool call)
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".world ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"world\"}}]}}"},
        {R"(json<|message|>{
)",
            ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\n"}}]}})"},
        {"   ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"   "}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \""}}]}})"},
        {"Warsaw", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Warsaw"}}]}})"},
        {"\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\""}}]}})"},
        {"}<|call|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}})"},

        // Tool 2 empty msg
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".world ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":1,\"function\":{\"name\":\"world\"}}]}}"},
        {"json", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{}<|call|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{}"}}]}})"},

        // Tool 3 empty msg
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".world ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":2,\"function\":{\"name\":\"world\"}}]}}"},
        {"json", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>{}<|call|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{}"}}]}})"},
    };
    test(chunkToDeltaVec);
}

// This does not conform to OpenAI Harmony format, however tests show that sometimes model produces tools in analysis channel
TEST_F(GptOssOutputStreamParserTest, HolisticStreamingToolsInAnalysisChannel) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Reasoning
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"I"}})"}},
        {" will", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" will"}})"}},
        {" call", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" call"}})"}},
        {" fun", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" fun"}})"}},
        {"ction.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"ction."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Preamble
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"I"}})"}},
        {" have", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" have"}})"}},
        {" to", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" to"}})"}},
        {" call", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" call"}})"}},
        {" fun", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" fun"}})"}},
        {"ction.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"ction."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool 1 (with ignored constrain)
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".hello ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"hello\"}}]}}"},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" {\""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \""}}]}})"},
        {"Paris", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris"}}]}})"},
        {"\"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}})"},
        {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool 2 (with constrain)
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".world ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":1,\"function\":{\"name\":\"world\"}}]}}"},
        {"json", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" {\""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" \""}}]}})"},
        {"Warsaw", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"Warsaw"}}]}})"},
        {"\"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"}"}}]}})"},
        {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool 3 (glued with call token)
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".world ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":2,\"function\":{\"name\":\"world\"}}]}}"},
        {"json", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"{}<|call|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{}"}}]}})"},
    };
    test(chunkToDeltaVec);
}
