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

#include "generation_config_builder.hpp"
#include "generation_context.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ovms {
namespace {

std::string getConsumedHeader(const GptOssGenerationContext& state) {
    using Header = GptOssGenerationContext::Header;
    switch (state.header) {
    case Header::None:
        return {};
    case Header::Assistant:
        return "<|start|>assistant";
    case Header::Channel:
        return "<|start|>assistant<|channel|>" + state.channel;
    case Header::ToolRecipient:
        return (state.recipientAfterChannel
                       ? "<|start|>assistant<|channel|>" + state.channel
                       : "<|start|>assistant") +
               " to=functions." + state.partialToolName;
    case Header::Unsupported:
        throw std::invalid_argument("Unsupported prefilled GPT-OSS assistant header");
    }
    throw std::invalid_argument("Unsupported GPT-OSS parser state");
}

}  // namespace

void GptOssGenerationConfigBuilder::applyToolGuidance(const OpenAIRequest& request, const ModelGenerationContext& generationContext) {
    using SOC = ov::genai::StructuredOutputConfig;
    const auto gptOssContext = GptOssGenerationContextAdapter::adapt(generationContext);
    const std::string consumedHeader = getConsumedHeader(gptOssContext);
    const bool namedChoice = requiresNamedToolCall(request);
    const bool requiresCall = namedChoice || request.toolChoice == ToolChoiceMode::Required;
    std::vector<SOC::Tag> toolMessages;
    const SOC::Regex recipientFirstSuffix(R"(([ \t]+<\|channel\|>(commentary|analysis)([ \t]+json)?[ \t]*|[ \t]*)(<\|constrain\|>json)?<\|message\|>)");
    const SOC::Regex channelFirstSuffix(R"([ \t]*(<\|constrain\|>json)?<\|message\|>)");
    for (const auto& [name, schema] : request.toolNameSchemaMap) {
        const auto arguments = buildToolArgumentGrammar(schema.stringRepr);
        toolMessages.emplace_back("<|start|>assistant to=functions." + name,
            std::make_shared<SOC::Concat>(std::vector<SOC::StructuralTag>{recipientFirstSuffix, arguments}), "<|call|>");
        for (const std::string channel : {"commentary", "analysis"}) {
            toolMessages.emplace_back("<|start|>assistant<|channel|>" + channel + " to=functions." + name,
                std::make_shared<SOC::Concat>(std::vector<SOC::StructuralTag>{channelFirstSuffix, arguments}), "<|call|>");
        }
    }

    const std::vector<SOC::Tag> analysisMessages{
        SOC::Tag("<|start|>assistant<|channel|>analysis<|message|>", SOC::AnyText{}, "<|end|>")};
    const std::vector<SOC::Tag> commentaryMessages{
        SOC::Tag("<|start|>assistant<|channel|>commentary<|message|>", SOC::AnyText{}, "<|end|>")};
    auto afterCall = std::make_shared<SOC::TagsWithSeparator>();
    afterCall->separator = "";
    afterCall->tags = commentaryMessages;
    if (!namedChoice) {
        afterCall->tags.insert(afterCall->tags.end(), toolMessages.begin(), toolMessages.end());
    }
    const std::vector<SOC::Tag> finalMessages{
        SOC::Tag("<|start|>assistant<|channel|>final<|message|>", SOC::AnyText{}, "<|return|>"),
        SOC::Tag("<|start|>assistant<|message|>", SOC::AnyText{}, "<|return|>")};
    afterCall->tags.insert(afterCall->tags.end(), finalMessages.begin(), finalMessages.end());

    auto toolAlternatives = std::make_shared<SOC::Union>();
    for (const auto& message : toolMessages) {
        toolAlternatives->elements.emplace_back(std::make_shared<SOC::Tag>(message));
    }
    // Before the first action, GPT-OSS may emit analysis messages followed by commentary.
    // It cannot return to analysis after the parser routes to content or a tool call.
    // After these opening messages, auto must produce a tool call or final answer;
    // required and named choices must produce a tool call.

    // Why raw EBNF and not built-in grammar constructs?
    // XGrammar only allows TagsWithSeparator as the last item in a sequence, but this
    // repeated message sequence precedes that required choice, so it is expressed in EBNF.
    // The body rules allow partial <|end|> text as content but stop at the complete delimiter.
    const SOC::EBNF analysisThenCommentaryPrefix(R"(root ::= analysis* commentary*
analysis ::= "<|start|>assistant<|channel|>analysis<|message|>" body "<|end|>"
commentary ::= "<|start|>assistant<|channel|>commentary<|message|>" body "<|end|>"
body ::= "" | [^<] body | "<" after-open
after-open ::= "" | "<" after-open | "|" after-pipe | [^<|] body
after-pipe ::= "" | "<" after-open | "e" after-e | [^<e] body
after-e ::= "" | "<" after-open | "n" after-n | [^<n] body
after-n ::= "" | "<" after-open | "d" after-d | [^<d] body
after-d ::= "" | "<" after-open | "|" after-end-pipe | [^<|] body
after-end-pipe ::= "" | "<" after-open | [^<>] body
)");
    const SOC::EBNF commentaryOnlyPrefix(R"(root ::= commentary*
commentary ::= "<|start|>assistant<|channel|>commentary<|message|>" body "<|end|>"
body ::= "" | [^<] body | "<" after-open
after-open ::= "" | "<" after-open | "|" after-pipe | [^<|] body
after-pipe ::= "" | "<" after-open | "e" after-e | [^<e] body
after-e ::= "" | "<" after-open | "n" after-n | [^<n] body
after-n ::= "" | "<" after-open | "d" after-d | [^<d] body
after-d ::= "" | "<" after-open | "|" after-end-pipe | [^<|] body
after-end-pipe ::= "" | "<" after-open | [^<>] body
)");
    auto initialMessages = std::make_shared<SOC::Union>();
    const auto initialAlternatives = [&](const std::vector<SOC::Tag>& messages) {
        auto alternatives = std::make_shared<SOC::Union>();
        for (const auto& message : messages) {
            if (message.begin.compare(0, consumedHeader.size(), consumedHeader) == 0) {
                alternatives->elements.emplace_back(std::make_shared<SOC::Tag>(
                    message.begin.substr(consumedHeader.size()), message.content, message.end));
            }
        }
        return alternatives;
    };
    const auto firstTools = initialAlternatives(toolMessages);
    if (!firstTools->elements.empty()) {
        initialMessages->elements.emplace_back(std::make_shared<SOC::Concat>(std::vector<SOC::StructuralTag>{firstTools, afterCall}));
    }
    const auto firstAnalysis = initialAlternatives(analysisMessages);
    const auto firstCommentary = initialAlternatives(commentaryMessages);
    const auto addFirstTextMessagePaths = [&](const std::shared_ptr<SOC::Union>& firstMessage, const SOC::EBNF& remainingOpeningMessages) {
        if (firstMessage->elements.empty()) {
            return;
        }
        initialMessages->elements.emplace_back(std::make_shared<SOC::Concat>(
            std::vector<SOC::StructuralTag>{firstMessage, remainingOpeningMessages, toolAlternatives, afterCall}));
        if (!requiresCall) {
            auto finalAnswer = std::make_shared<SOC::TagsWithSeparator>();
            finalAnswer->separator = "";
            finalAnswer->at_least_one = true;
            finalAnswer->stop_after_first = true;
            finalAnswer->tags = finalMessages;
            initialMessages->elements.emplace_back(std::make_shared<SOC::Concat>(
                std::vector<SOC::StructuralTag>{firstMessage, remainingOpeningMessages, finalAnswer}));
        }
    };
    addFirstTextMessagePaths(firstAnalysis, analysisThenCommentaryPrefix);
    addFirstTextMessagePaths(firstCommentary, commentaryOnlyPrefix);
    if (!requiresCall) {
        const auto firstFinal = initialAlternatives(finalMessages);
        if (!firstFinal->elements.empty()) {
            auto finalAnswer = std::make_shared<SOC::TagsWithSeparator>();
            finalAnswer->separator = "";
            finalAnswer->at_least_one = true;
            finalAnswer->stop_after_first = true;
            for (const auto& message : firstFinal->elements) {
                finalAnswer->tags.push_back(*std::get<std::shared_ptr<SOC::Tag>>(message));
            }
            initialMessages->elements.emplace_back(std::make_shared<SOC::Concat>(std::vector<SOC::StructuralTag>{finalAnswer}));
        }
    }
    if (initialMessages->elements.empty()) {
        throw std::invalid_argument("Unsupported prefilled GPT-OSS assistant header");
    }
    config.structured_output_config.emplace();
    setStructuralTagsConfig(initialMessages);
}

}  // namespace ovms
