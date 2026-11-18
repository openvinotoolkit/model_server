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
#include "../output_parsing_config.hpp"
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

    auto beforeCall = std::make_shared<SOC::TagsWithSeparator>();
    beforeCall->separator = "";
    beforeCall->tags = {
        SOC::Tag("<|start|>assistant<|channel|>analysis<|message|>", SOC::AnyText{}, "<|end|>"),
        SOC::Tag("<|start|>assistant<|channel|>commentary<|message|>", SOC::AnyText{}, "<|end|>")};
    auto afterCall = std::make_shared<SOC::TagsWithSeparator>();
    afterCall->separator = "";
    afterCall->tags = beforeCall->tags;
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
    // We need a mandatory tool-call rule after zero or more reasoning messages.
    // XGrammar allows TagsWithSeparator only as the last element of a sequence, so
    // it cannot express this prefix. Use EBNF to allow the tool-call rule to follow.
    // The body rules
    // exclude a complete <|end|> marker while allowing partial marker text as content.
    const SOC::EBNF reasoningPrefix(R"(root ::= message*
message ::= "<|start|>assistant<|channel|>" ("analysis" | "commentary") "<|message|>" body "<|end|>"
body ::= "" | [^<] body | "<" after-open
after-open ::= "" | "<" after-open | "|" after-pipe | [^<|] body
after-pipe ::= "" | "<" after-open | "e" after-e | [^<e] body
after-e ::= "" | "<" after-open | "n" after-n | [^<n] body
after-n ::= "" | "<" after-open | "d" after-d | [^<d] body
after-d ::= "" | "<" after-open | "|" after-end-pipe | [^<|] body
after-end-pipe ::= "" | "<" after-open | [^<>] body
)");
    const SOC::StructuralTag continuation = requiresCall
                                                ? SOC::StructuralTag(std::make_shared<SOC::Concat>(std::vector<SOC::StructuralTag>{reasoningPrefix, toolAlternatives, afterCall}))
                                                : SOC::StructuralTag(afterCall);
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
    const auto firstReasoning = initialAlternatives(beforeCall->tags);
    if (!firstReasoning->elements.empty()) {
        initialMessages->elements.emplace_back(std::make_shared<SOC::Concat>(std::vector<SOC::StructuralTag>{firstReasoning, continuation}));
    }
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
