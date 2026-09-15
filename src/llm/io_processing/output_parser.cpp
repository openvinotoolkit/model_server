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

#include <algorithm>
#include <unordered_set>

#include "src/logging.hpp"
#include "src/stringutils.hpp"
#include "output_parser.hpp"
#include "parser_config_validation.hpp"
#include "llama3/tool_parser.hpp"
#include "hermes3/tool_parser.hpp"
#include "phi4/tool_parser.hpp"
#include "mistral/tool_parser.hpp"
#include "gptoss/tool_parser.hpp"
#include "qwen3/reasoning_parser.hpp"
#include "qwen3coder/qwen3coder_tool_parser.hpp"
#include "devstral/tool_parser.hpp"
#include "gemma4/gemma4_reasoning_parser.hpp"
#include "gptoss/reasoning_parser.hpp"
#include "lfm2/lfm2_tool_parser.hpp"
#include "lfm2/lfm25_reasoning_parser.hpp"
#include "gemma4/gemma4_tool_parser.hpp"
#include "onyx/onyx_tool_parser.hpp"
#include "onyx/onyx_reasoning_parser.hpp"
#include "onyx/onyx_content_parser.hpp"
#include "default_content_parser.hpp"
#include "minicpm5/minicpm5_tool_parser.hpp"
#include "minicpm5/minicpm5_reasoning_parser.hpp"

namespace ovms {
OutputParser::TagLookupStatus OutputParser::StreamOutputCache::lookupTag(const std::string& tag) const {
    if (tag.empty()) {
        return TagLookupStatus::NOT_FOUND;
    }
    if (tag.size() > buffer.size()) {
        if (stringsOverlap(buffer, tag)) {
            return TagLookupStatus::FOUND_INCOMPLETE;
        } else {
            return TagLookupStatus::NOT_FOUND;
        }
    } else if (tag.size() < buffer.size()) {
        if (buffer.find(tag) != std::string::npos) {
            return TagLookupStatus::FOUND_COMPLETE;
        } else if (stringsOverlap(buffer, tag)) {
            return TagLookupStatus::FOUND_INCOMPLETE;
        } else {
            return TagLookupStatus::NOT_FOUND;
        }
    } else {
        if (buffer == tag) {
            return TagLookupStatus::FOUND_COMPLETE;
        } else if (stringsOverlap(buffer, tag)) {
            return TagLookupStatus::FOUND_INCOMPLETE;
        } else {
            return TagLookupStatus::NOT_FOUND;
        }
    }
}

OutputParser::TagLookupStatus OutputParser::StreamOutputCache::lookupTags(const std::vector<std::string>& tags) const {
    TagLookupStatus finalTagLookupStatus = TagLookupStatus::NOT_FOUND;
    for (const auto& tag : tags) {
        auto tagLookupStatus = lookupTag(tag);
        if (tagLookupStatus == TagLookupStatus::FOUND_COMPLETE) {
            return TagLookupStatus::FOUND_COMPLETE;
        }
        if (tagLookupStatus == TagLookupStatus::FOUND_INCOMPLETE) {
            finalTagLookupStatus = TagLookupStatus::FOUND_INCOMPLETE;
        }
    }
    return finalTagLookupStatus;
}

void OutputParser::StreamOutputCache::add(const std::string& chunk) {
    buffer += chunk;
}

void OutputParser::StreamOutputCache::clear() {
    buffer.clear();
}

const std::string& OutputParser::StreamOutputCache::getBuffer() const {
    return buffer;
}

std::optional<Delta> OutputParser::parseContentChunk(ProcessingPhase newPhase) {
    auto result = contentParser->parseChunk(streamOutputCache.getBuffer(), {}, ov::genai::GenerationFinishReason::NONE);
    if (!result.has_value())
        return std::nullopt;
    streamOutputCache.clear();
    processingPhase = newPhase;
    if (const auto* cd = std::get_if<ContentDelta>(&*result)) {
        if (cd->text.empty())
            return std::nullopt;
    }
    return result;
}

std::optional<Delta> OutputParser::parseToolCallChunk(const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase) {
    if (!toolParser) {
        throw std::runtime_error("Tool parser is not available, cannot parse tool call chunk");
    }
    if (toolParser->getParsingConfig().ownsToolCallBoundaries) {
        const bool bareRecovery = streamOutputCache.getBuffer().rfind("call:", 0) == 0 &&
            finishReason == ov::genai::GenerationFinishReason::NONE;
        auto result = toolParser->parseChunk(streamOutputCache.getBuffer(), tokens, finishReason);
        streamOutputCache.clear();
        processingPhase = TOOL_CALLS_PROCESSING_TOOL;
        if (bareRecovery && result.has_value() && std::holds_alternative<ToolCallDelta>(*result)) {
            pendingDelta = std::move(result);
            return std::nullopt;
        }
        return result;
    }
    std::string remainder;
    const std::string& endTag = toolParser->getParsingConfig().endTag;
    if (!endTag.empty()) {
        const std::string& buf = streamOutputCache.getBuffer();
        const size_t pos = buf.find(endTag);
        if (pos != std::string::npos)
            remainder = buf.substr(pos + endTag.size());
    }
    std::optional<Delta> result;
    try {
        result = toolParser->parseChunk(streamOutputCache.getBuffer(), tokens, finishReason);
    } catch (...) {
        streamOutputCache.clear();
        throw;
    }
    streamOutputCache.clear();
    processingPhase = newPhase;
    if (!remainder.empty())
        streamOutputCache.add(remainder);
    return result;
}

std::optional<Delta> OutputParser::parseReasoningChunk(const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase) {
    if (!reasoningParser) {
        throw std::runtime_error("Reasoning parser is not available, cannot parse reasoning chunk");
    }
    std::string remainder;
    const std::string& endTag = reasoningParser->getParsingConfig().endTag;
    if (!endTag.empty()) {
        const std::string& buf = streamOutputCache.getBuffer();
        const size_t pos = buf.find(endTag);
        if (pos != std::string::npos)
            remainder = buf.substr(pos + endTag.size());
    }
    std::optional<Delta> result;
    try {
        result = reasoningParser->parseChunk(streamOutputCache.getBuffer(), tokens, finishReason);
    } catch (...) {
        streamOutputCache.clear();
        throw;
    }
    streamOutputCache.clear();
    processingPhase = newPhase;
    if (!remainder.empty())
        streamOutputCache.add(remainder);
    return result;
}

OutputParser::OutputParser(ov::genai::Tokenizer& tokenizer, const std::string toolParserName, const std::string reasoningParserName, const ToolsSchemas_t& toolNameSchemaMap) :
    tokenizer(tokenizer) {
    if (toolParserName == "llama3") {
        toolParser = std::make_unique<Llama3ToolParser>(tokenizer);
    } else if (toolParserName == "hermes3") {
        toolParser = std::make_unique<Hermes3ToolParser>(tokenizer);
    } else if (toolParserName == "phi4") {
        toolParser = std::make_unique<Phi4ToolParser>(tokenizer);
    } else if (toolParserName == "mistral") {
        toolParser = std::make_unique<MistralToolParser>(tokenizer);
    } else if (toolParserName == "gptoss") {
        toolParser = std::make_unique<GptOssToolParser>(tokenizer);
    } else if (toolParserName == "qwen3coder") {
        toolParser = std::make_unique<Qwen3CoderToolParser>(tokenizer, toolNameSchemaMap);
    } else if (toolParserName == "devstral") {
        toolParser = std::make_unique<DevstralToolParser>(tokenizer, toolNameSchemaMap);
    } else if (toolParserName == "lfm2") {
        toolParser = std::make_unique<Lfm2ToolParser>(tokenizer);
    } else if (toolParserName == "gemma4") {
        toolParser = std::make_unique<Gemma4ToolParser>(tokenizer, toolNameSchemaMap);
    } else if (toolParserName == "onyx") {
        toolParser = std::make_unique<OnyxToolParser>(tokenizer, toolNameSchemaMap);
    } else if (toolParserName == "minicpm5") {
        toolParser = std::make_unique<Minicpm5ToolParser>(tokenizer, toolNameSchemaMap);
    } else if (!toolParserName.empty()) {
        throw std::runtime_error("Unsupported tool parser: \"" + toolParserName +
                                 "\". Supported tool parsers are: " + getSupportedToolParserNamesAsString());
    }

    if (reasoningParserName == "qwen3") {
        reasoningParser = std::make_unique<Qwen3ReasoningParser>(tokenizer);
    } else if (reasoningParserName == "gemma4") {
        reasoningParser = std::make_unique<Gemma4ReasoningParser>(tokenizer);
    } else if (reasoningParserName == "gptoss") {
        reasoningParser = std::make_unique<GptOssReasoningParser>(tokenizer);
    } else if (reasoningParserName == "minicpm5") {
        reasoningParser = std::make_unique<Minicpm5ReasoningParser>(tokenizer);
    } else if (reasoningParserName == "lfm2") {
        reasoningParser = std::make_unique<Lfm25ReasoningParser>(tokenizer);
    } else if (reasoningParserName == "onyx") {
        reasoningParser = std::make_unique<OnyxReasoningParser>(tokenizer);
    } else if (!reasoningParserName.empty()) {
        throw std::runtime_error("Unsupported reasoning parser: \"" + reasoningParserName +
                                 "\". Supported reasoning parsers are: " + getSupportedReasoningParserNamesAsString());
    }

    if (toolParserName == "onyx" || reasoningParserName == "onyx")
        contentParser = std::make_unique<OnyxContentParser>(tokenizer);
    else if (toolParserName == "gptoss" || reasoningParserName == "gptoss")
        contentParser = std::make_unique<DefaultContentParser>(tokenizer, std::vector<std::string>{
                                                                              "<|start|>assistant<|channel|>final<|message|>",
                                                                              "<|channel|>final<|message|>",
                                                                              "<|channel|>commentary<|message|>",
                                                                              "<|end|>",
                                                                              "<|return|>"});
    else if (toolParserName == "gemma4")
        contentParser = std::make_unique<DefaultContentParser>(tokenizer, std::vector<std::string>{"<turn|>", "<|tool_response>", "<|channel>thought\n", "<channel|>"});
    else if (toolParserName == "lfm2")
        contentParser = std::make_unique<DefaultContentParser>(tokenizer, std::vector<std::string>{"<|im_end|>"});
    else if (toolParserName == "minicpm5")
        contentParser = std::make_unique<DefaultContentParser>(tokenizer, std::vector<std::string>{"<s>", "<|im_end|>"});
    else
        contentParser = std::make_unique<DefaultContentParser>(tokenizer);

    defaultDecodingWithSpecialTokens =
        (toolParser && toolParser->getParsingConfig().defaultDecodingWithSpecialTokens) ||
        (reasoningParser && reasoningParser->getParsingConfig().defaultDecodingWithSpecialTokens);

    if (llm_calculator_logger->should_log(spdlog::level::debug)) {
        std::string toolParsingConfigStr = toolParser ? toolParser->buildParsingConfigStringRepresentation() : "N/A";
        std::string reasoningParsingConfigStr = reasoningParser ? reasoningParser->buildParsingConfigStringRepresentation() : "N/A";
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
            "OutputParser initialized with tool parser: \"{}\" (parsing config: {}), reasoning parser: \"{}\" (parsing config: {}), defaultDecodingWithSpecialTokens={}",
            toolParserName, toolParsingConfigStr, reasoningParserName, reasoningParsingConfigStr,
            defaultDecodingWithSpecialTokens);
    }
}

bool OutputParser::isToolParserAvailable() const {
    return toolParser != nullptr;
}

bool OutputParser::isReasoningParserAvailable() const {
    return reasoningParser != nullptr;
}

std::string OutputParser::getToolParserStartTag() const {
    if (toolParser) {
        return toolParser->getParsingConfig().startTags[0];
    } else {
        throw std::runtime_error("Tool parser is not available, cannot get start tag");
    }
}

void OutputParser::resetStreamingState() {
    processingPhase = UNKNOWN;
    streamOutputCache.clear();
    pendingDelta.reset();
    if (toolParser)
        toolParser->resetState();
    if (reasoningParser)
        reasoningParser->resetState();
    if (contentParser)
        contentParser->resetState();
    if (implicitReasoningStart) {
        setImplicitReasoningStart(true);
    }
}

bool OutputParser::needSpecialTokensForCurrentDecode(bool userWantsSpecialTokens) const {
    if (processingPhase == CONTENT || processingPhase == UNKNOWN) {
        return defaultDecodingWithSpecialTokens || userWantsSpecialTokens;
    }
    if (processingPhase == REASONING) {
        return reasoningParser && reasoningParser->getParsingConfig().needsSpecialTokens;
    }
    if (processingPhase == TOOL_CALLS_PROCESSING_TOOL || processingPhase == TOOL_CALLS_WAITING_FOR_TOOL) {
        return toolParser && toolParser->getParsingConfig().needsSpecialTokens;
    }
    return false;
}

std::string OutputParser::getPhaseStartTagForToken(int64_t tokenId, bool toolsAvailable) const {
    if (toolParser && toolsAvailable) {
        const auto& tokenMap = toolParser->getResolvedStartTokenToTag();
        auto it = tokenMap.find(tokenId);
        if (it != tokenMap.end() &&
            processingPhase != TOOL_CALLS_PROCESSING_TOOL &&
            processingPhase != TOOL_CALLS_WAITING_FOR_TOOL) {
            return it->second;
        }
    }
    if (reasoningParser) {
        const auto& tokenMap = reasoningParser->getResolvedStartTokenToTag();
        auto it = tokenMap.find(tokenId);
        if (it != tokenMap.end() && processingPhase != REASONING) {
            return it->second;
        }
    }
    return {};
}

void OutputParser::setImplicitReasoningStart(bool value) {
    implicitReasoningStart = value;
    if (!reasoningParser) {
        return;
    }
    reasoningParser->setImplicitStart(value);
    if (processingPhase == UNKNOWN || processingPhase == REASONING) {
        processingPhase = value ? REASONING : UNKNOWN;
    }
}

void OutputParser::detectAndSetImplicitReasoningStart(const std::string& renderedPrompt) {
    if (!reasoningParser) {
        return;
    }
    std::string trimmed = renderedPrompt;
    rtrim(trimmed);
    const auto& startTags = reasoningParser->getParsingConfig().startTags;
    // Tags may end with whitespace (e.g. Gemma4 "<|channel>thought\n").
    // Since the prompt is rtrimmed, compare against an rtrimmed tag copy so
    // a prompt ending inside the thought channel is still detected.
    bool detected = std::any_of(startTags.begin(), startTags.end(), [&](const std::string& tag) {
        if (tag.empty()) {
            return false;
        }
        std::string trimmedTag = tag;
        rtrim(trimmedTag);
        return !trimmedTag.empty() && endsWith(trimmed, trimmedTag);
    });
    setImplicitReasoningStart(detected);
    return;
}

std::optional<Delta> OutputParser::parseChunk(const std::string& chunkResponse, const std::vector<int64_t>& tokens, const bool toolsAvailable, ov::genai::GenerationFinishReason finishReason) {
    bool reasoningParserExistsAndSupportsStreaming = reasoningParser && !reasoningParser->getParsingConfig().startTags.empty() && !reasoningParser->getParsingConfig().endTag.empty();
    bool toolParserExistsAndSupportsStreaming = toolParser && !toolParser->getParsingConfig().startTags.empty();
    bool applyToolParser = toolParserExistsAndSupportsStreaming && toolsAvailable;

    streamOutputCache.add(chunkResponse);
    if (pendingDelta.has_value()) {
        auto result = std::move(pendingDelta);
        pendingDelta.reset();
        return result;
    }

    if (llm_calculator_logger->should_log(spdlog::level::trace)) {
        std::string tokenIds;
        tokenIds.reserve(tokens.size() * 7);
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0)
                tokenIds += ", ";
            tokenIds += std::to_string(tokens[i]);
        }
        std::string processingPhaseStr;
        switch (processingPhase) {
        case UNKNOWN: processingPhaseStr = "UNKNOWN"; break;
        case CONTENT: processingPhaseStr = "CONTENT"; break;
        case REASONING: processingPhaseStr = "REASONING"; break;
        case TOOL_CALLS_PROCESSING_TOOL: processingPhaseStr = "TOOL_CALLS_PROCESSING_TOOL"; break;
        case TOOL_CALLS_WAITING_FOR_TOOL: processingPhaseStr = "TOOL_CALLS_WAITING_FOR_TOOL"; break;
        default: processingPhaseStr = "UNKNOWN"; break;
        }
        SPDLOG_LOGGER_TRACE(llm_calculator_logger,
            "OutputParser::parseChunk[PROCESSING_PHASE={}] called with {} tokens, text=\"{}\", finish_reason={}, token IDs=[{}]",
            processingPhaseStr, tokens.size(), chunkResponse, static_cast<int>(finishReason), tokenIds);
    }

    if (processingPhase == UNKNOWN) {
        TagLookupStatus anyStartTagStatus = TagLookupStatus::NOT_FOUND;
        if (reasoningParserExistsAndSupportsStreaming) {
            TagLookupStatus reasoningStartTagStatus = streamOutputCache.lookupTags(reasoningParser->getParsingConfig().startTags);
            if (reasoningStartTagStatus == TagLookupStatus::NOT_FOUND) {
                reasoningStartTagStatus = streamOutputCache.lookupTags(reasoningParser->getParsingConfig().preambleStartTags);
            }
            if (reasoningStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseReasoningChunk(tokens, finishReason);
            }
            anyStartTagStatus = reasoningStartTagStatus;
        }

        if (applyToolParser) {
            TagLookupStatus toolCallStartTagStatus = streamOutputCache.lookupTags(toolParser->getParsingConfig().startTags);
            if (toolCallStartTagStatus == TagLookupStatus::NOT_FOUND) {
                toolCallStartTagStatus = streamOutputCache.lookupTags(toolParser->getParsingConfig().preambleStartTags);
            }
            if (toolCallStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseToolCallChunk(tokens, finishReason);
            }
            if (toolCallStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE) {
                anyStartTagStatus = toolCallStartTagStatus;
            }
        }

        if ((!reasoningParserExistsAndSupportsStreaming && !applyToolParser) || finishReason != ov::genai::GenerationFinishReason::NONE || anyStartTagStatus == TagLookupStatus::NOT_FOUND) {
            return parseContentChunk();
        }
        return std::nullopt;
    } else if (processingPhase == REASONING) {
        const auto& reasoningConfig = reasoningParser->getParsingConfig();
        TagLookupStatus endTagStatus = streamOutputCache.lookupTag(reasoningConfig.endTag);

        if (reasoningConfig.toolStartTerminatesReasoning && applyToolParser) {
            const auto& toolStartTags = toolParser->getParsingConfig().startTags;
            TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTags(toolStartTags);

            if (toolStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                const std::string& buf = streamOutputCache.getBuffer();
                size_t toolStartPos = std::string::npos;
                for (const auto& tag : toolStartTags) {
                    const size_t pos = buf.find(tag);
                    if (pos != std::string::npos && (toolStartPos == std::string::npos || pos < toolStartPos)) {
                        toolStartPos = pos;
                    }
                }
                const size_t reasoningEndPos = reasoningConfig.endTag.empty() ? std::string::npos : buf.find(reasoningConfig.endTag);

                // Canonical Google Gemma4 closes the thought channel with <channel|>
                // before <|tool_call>. A parser may opt into toolStartTerminatesReasoning
                // only as a tolerance/recovery boundary for malformed or edge output.
                // Take over only when that recovery opener precedes an explicit closer.
                if (toolStartPos != std::string::npos &&
                    (reasoningEndPos == std::string::npos || toolStartPos < reasoningEndPos)) {
                    const std::string reasoningPrefix = buf.substr(0, toolStartPos);
                    const std::string toolRemainder = buf.substr(toolStartPos);
                    streamOutputCache.clear();
                    processingPhase = TOOL_CALLS_PROCESSING_TOOL;
                    streamOutputCache.add(toolRemainder);

                    if (!reasoningPrefix.empty()) {
                        auto reasoningDelta = reasoningParser->parseChunk(reasoningPrefix, tokens, finishReason);
                        if (reasoningDelta.has_value()) {
                            return reasoningDelta;
                        }
                    }
                    return parseToolCallChunk(tokens, finishReason);
                }
            }

            // Hold back a partial tool opener at the tail instead of leaking its
            // bytes into reasoning. This mirrors the boundary holdback used by
            // the dedicated Gemma4 parsers in other runtimes.
            if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE &&
                endTagStatus != TagLookupStatus::FOUND_COMPLETE &&
                finishReason == ov::genai::GenerationFinishReason::NONE) {
                return std::nullopt;
            }
        }

        if (endTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            return parseReasoningChunk(tokens, finishReason, UNKNOWN);
        } else if (endTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;
        }
        return parseReasoningChunk(tokens, finishReason);
    } else if (processingPhase == CONTENT) {
        if (applyToolParser) {
            // Gemma4 owns tool-call/content boundaries after ordinary prose has
            // started. The generic content parser cannot recover split line-start
            // `call:` or strip Gemma4 turn/tool-response markers without losing
            // byte ownership, so route through the owning parser once requested.
            if (toolParser->getParsingConfig().ownsToolCallBoundaries)
                return parseToolCallChunk(tokens, finishReason);
            TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTags(toolParser->getParsingConfig().startTags);
            if (toolStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseToolCallChunk(tokens, finishReason);
            } else if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
                return std::nullopt;
            }
            return parseContentChunk();
        }
        return parseContentChunk();
    } else if (processingPhase == TOOL_CALLS_PROCESSING_TOOL) {
        if (toolParser->getParsingConfig().ownsToolCallBoundaries)
            return parseToolCallChunk(tokens, finishReason);
        TagLookupStatus toolEndTagStatus = streamOutputCache.lookupTag(toolParser->getParsingConfig().endTag);
        if (toolEndTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;
        }
        if (toolEndTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            return parseToolCallChunk(tokens, finishReason, TOOL_CALLS_WAITING_FOR_TOOL);
        }
        return parseToolCallChunk(tokens, finishReason);
    } else if (processingPhase == TOOL_CALLS_WAITING_FOR_TOOL) {
        TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTags(toolParser->getParsingConfig().startTags);
        if (toolStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            return parseToolCallChunk(tokens, finishReason, TOOL_CALLS_PROCESSING_TOOL);
        }
        const auto& contentTurnStartTags = contentParser->getParsingConfig().startTags;
        if (!contentTurnStartTags.empty()) {
            TagLookupStatus contentTurnStatus = streamOutputCache.lookupTags(contentTurnStartTags);
            if (contentTurnStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseContentChunk();
            }
            if (finishReason != ov::genai::GenerationFinishReason::NONE) {
                return parseContentChunk();
            }
            return std::nullopt;
        }
        if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;
        }
        return parseToolCallChunk(tokens, finishReason, TOOL_CALLS_WAITING_FOR_TOOL);
    } else {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Unexpected processing phase: {}", static_cast<int>(processingPhase));
        throw std::runtime_error("Unexpected error during stream output parsing");
    }
}
}  // namespace ovms
