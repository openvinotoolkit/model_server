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
        /* 
        If the tag is longer than the buffer, we check if the buffer and tag overlap (either partially or fully for exact match)
        They do overlap, we assume that tag may appear in the future, so we return FOUND_INCOMPLETE
        otherwise we return NOT_FOUND
        */
        if (stringsOverlap(buffer, tag)) {
            return TagLookupStatus::FOUND_INCOMPLETE;
        } else {
            return TagLookupStatus::NOT_FOUND;
        }
    } else if (tag.size() < buffer.size()) {
        /* 
        If the tag is shorter than the buffer, we check:
           a) if the tag is a substring of the buffer (tag is fully matched)
           b) if the buffer and tag overlap (part of the tag is matched)
        in the first case we return FOUND_COMPLETE, in the second FOUND_INCOMPLETE
        otherwise we return NOT_FOUND
        */
        if (buffer.find(tag) != std::string::npos) {
            return TagLookupStatus::FOUND_COMPLETE;
        } else if (stringsOverlap(buffer, tag)) {
            return TagLookupStatus::FOUND_INCOMPLETE;
        } else {
            return TagLookupStatus::NOT_FOUND;
        }
    } else {
        /*
        If the tag and buffer are of the same length, we check:
              a) if they are equal (tag is fully matched)
              b) if they overlap (part of the tag is matched)
        in the first case we return FOUND_COMPLETE, in the second FOUND_INCOMPLETE
        otherwise we return NOT_FOUND
        */
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
    // We look for multiple tags and return the status in the following priority: FOUND COMPLETE > FOUND_INCOMPLETE > NOT_FOUND
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

std::optional<rapidjson::Document> OutputParser::parseContentChunk(ProcessingPhase newPhase) {
    auto result = contentParser->parseChunk(streamOutputCache.getBuffer(), {}, ov::genai::GenerationFinishReason::NONE);
    if (!result.has_value())
        return std::nullopt;  // hold — keep buffer
    streamOutputCache.clear();
    processingPhase = newPhase;
    const auto& deltaIt = result->FindMember("delta");
    if (deltaIt != result->MemberEnd()) {
        const auto& contentIt = deltaIt->value.FindMember("content");
        if (contentIt != deltaIt->value.MemberEnd() && contentIt->value.GetStringLength() == 0)
            return std::nullopt;  // consumed preamble, nothing to emit
    }
    return result;
}

std::optional<rapidjson::Document> OutputParser::parseToolCallChunk(const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase) {
    if (!toolParser) {
        throw std::runtime_error("Tool parser is not available, cannot parse tool call chunk");
    }
    // Bytes after the end tag belong to the next phase — preserve them before clearing.
    std::string remainder;
    const std::string& endTag = toolParser->getParsingConfig().endTag;
    if (!endTag.empty()) {
        const std::string& buf = streamOutputCache.getBuffer();
        const size_t pos = buf.find(endTag);
        if (pos != std::string::npos)
            remainder = buf.substr(pos + endTag.size());
    }
    std::optional<rapidjson::Document> result;
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

std::optional<rapidjson::Document> OutputParser::parseReasoningChunk(const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase) {
    if (!reasoningParser) {
        throw std::runtime_error("Reasoning parser is not available, cannot parse reasoning chunk");
    }
    // Bytes after the end tag belong to the next phase — preserve them before clearing.
    std::string remainder;
    const std::string& endTag = reasoningParser->getParsingConfig().endTag;
    if (!endTag.empty()) {
        const std::string& buf = streamOutputCache.getBuffer();
        const size_t pos = buf.find(endTag);
        if (pos != std::string::npos)
            remainder = buf.substr(pos + endTag.size());
    }
    std::optional<rapidjson::Document> result;
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
        toolParser = std::make_unique<Gemma4ToolParser>(tokenizer);
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

    // Model/output formats whose structural tokens must stay visible in the content/unknown phase
    // (e.g. GptOss uses <|channel|>... throughout the stream; devstral's [TOOL_CALLS] tag and
    // minicpm5's <s>/<|im_end|> must be visible before parser-owned phases begin). For all other
    // parser combinations the content phase decodes with skip_special_tokens=true (the default,
    // lower noise). Each parser that requires this sets defaultDecodingWithSpecialTokens in its config.
    if (toolParserName == "onyx" || reasoningParserName == "onyx")
        contentParser = std::make_unique<OnyxContentParser>(tokenizer);
    else if (toolParserName == "gemma4")
        contentParser = std::make_unique<DefaultContentParser>(tokenizer, std::vector<std::string>{"<turn|>", "<|tool_response>"});
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
    // Content / unknown phase: use the computed baseline for this parser combination;
    // also honour user preference here (scoped to content — does not override parser phases).
    if (processingPhase == CONTENT || processingPhase == UNKNOWN) {
        return defaultDecodingWithSpecialTokens || userWantsSpecialTokens;
    }
    // Reasoning phase: the active reasoning parser owns the decision.
    if (processingPhase == REASONING) {
        return reasoningParser && reasoningParser->getParsingConfig().needsSpecialTokens;
    }
    // Tool-call phases: the active tool parser owns the decision.
    if (processingPhase == TOOL_CALLS_PROCESSING_TOOL || processingPhase == TOOL_CALLS_WAITING_FOR_TOOL) {
        return toolParser && toolParser->getParsingConfig().needsSpecialTokens;
    }
    return false;
}

std::string OutputParser::getPhaseStartTagForToken(int64_t tokenId, bool toolsAvailable) const {
    // Returns the tag string for a phase-start token, or empty if not a phase-start token.
    // The guard conditions mirror isPhaseStartToken: don't re-fire for a phase we are
    // already in (the parser's own text-based detection handles re-entry there).
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
    // Bias the streaming state machine: the model output is expected to begin already
    // inside the reasoning segment, so skip the UNKNOWN phase and go straight to REASONING.
    // When value is false, restore the default initial phase.
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
    bool detected = std::any_of(startTags.begin(), startTags.end(),
        [&](const std::string& tag) { return !tag.empty() && endsWith(trimmed, tag); });
    setImplicitReasoningStart(detected);
    return;
}

std::optional<rapidjson::Document> OutputParser::parseChunk(const std::string& chunkResponse, const std::vector<int64_t>& tokens, const bool toolsAvailable, ov::genai::GenerationFinishReason finishReason) {
    /*
    Using appropriate parser based on the current processing phase
    Call to this method should return either result from parserContentChunk, parseToolCallChunk, parseReasoningChunk when we can determine the phase
    or std::nullopt when we are waiting for more chunks to determine if we should switch phase or not.
    Note that mentioned methods do not take chunk as argument, they read it from streamOutputCache and are responsible for clearing the cache,
    so only use those methods or return nullopt.
    */

    bool reasoningParserExistsAndSupportsStreaming = reasoningParser && !reasoningParser->getParsingConfig().startTags.empty() && !reasoningParser->getParsingConfig().endTag.empty();
    bool toolParserExistsAndSupportsStreaming = toolParser && !toolParser->getParsingConfig().startTags.empty();
    bool applyToolParser = toolParserExistsAndSupportsStreaming && toolsAvailable;

    streamOutputCache.add(chunkResponse);

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
        case UNKNOWN:
            processingPhaseStr = "UNKNOWN";
            break;
        case CONTENT:
            processingPhaseStr = "CONTENT";
            break;
        case REASONING:
            processingPhaseStr = "REASONING";
            break;
        case TOOL_CALLS_PROCESSING_TOOL:
            processingPhaseStr = "TOOL_CALLS_PROCESSING_TOOL";
            break;
        case TOOL_CALLS_WAITING_FOR_TOOL:
            processingPhaseStr = "TOOL_CALLS_WAITING_FOR_TOOL";
            break;
        default:
            processingPhaseStr = "UNKNOWN";
            break;
        }

        SPDLOG_LOGGER_TRACE(llm_calculator_logger,
            "OutputParser::parseChunk[PROCESSING_PHASE={}] called with {} tokens, text=\"{}\", finish_reason={}, token IDs=[{}]",
            processingPhaseStr, tokens.size(), chunkResponse, static_cast<int>(finishReason), tokenIds);
    }

    if (processingPhase == UNKNOWN) {
        // If we are in the UNKNOWN phase, we need to determine if we should switch to CONTENT, REASONING, or TOOL_CALLS phase.
        TagLookupStatus anyStartTagStatus = TagLookupStatus::NOT_FOUND;
        if (reasoningParserExistsAndSupportsStreaming) {
            // Check if reasoning start tag has been received
            TagLookupStatus reasoningStartTagStatus = streamOutputCache.lookupTags(reasoningParser->getParsingConfig().startTags);
            if (reasoningStartTagStatus == TagLookupStatus::NOT_FOUND) {
                // If reasoning start tag is not found, check if any of the special start tags are found
                reasoningStartTagStatus = streamOutputCache.lookupTags(reasoningParser->getParsingConfig().preambleStartTags);
            }
            if (reasoningStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseReasoningChunk(tokens, finishReason);
            }  // else startTagStatus is FOUND_INCOMPLETE or NOT_FOUND, we continue processing, so potential tool parser start tag is not missed
            anyStartTagStatus = reasoningStartTagStatus;
        }

        if (applyToolParser) {
            // Check if tool call start tag has been received
            TagLookupStatus toolCallStartTagStatus = streamOutputCache.lookupTags(toolParser->getParsingConfig().startTags);
            if (toolCallStartTagStatus == TagLookupStatus::NOT_FOUND) {
                // If tool call start tag is not found, check if any of the special start tags are found
                toolCallStartTagStatus = streamOutputCache.lookupTags(toolParser->getParsingConfig().preambleStartTags);
            }
            if (toolCallStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseToolCallChunk(tokens, finishReason);
            }  // else startTagStatus is FOUND_INCOMPLETE or NOT_FOUND, we continue processing
            if (toolCallStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE) {
                anyStartTagStatus = toolCallStartTagStatus;  // We have at least one incomplete start tag
            }
        }

        if ((!reasoningParserExistsAndSupportsStreaming && !applyToolParser) || finishReason != ov::genai::GenerationFinishReason::NONE || anyStartTagStatus == TagLookupStatus::NOT_FOUND) {
            // If no special parsers are available, generation has finished or we have no start tags we just return content chunks and switch to CONTENT phase.
            return parseContentChunk();
        }
        // If we are here, it means we have incomplete start tag for either reasoning or tool parser, so we wait for more chunks
        return std::nullopt;
    } else if (processingPhase == REASONING) {
        // If we are in the REASONING phase, we check if parsing end tag is found and if so, switch to UNKNOWN phase.
        TagLookupStatus endTagStatus = streamOutputCache.lookupTag(reasoningParser->getParsingConfig().endTag);
        if (endTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            // Switch back to UNKNOWN phase (we can have either CONTENT or TOOL_CALLS next)
            return parseReasoningChunk(tokens, finishReason, UNKNOWN);
        } else if (endTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;  // Wait for more chunks to determine if end tag is complete
        }
        return parseReasoningChunk(tokens, finishReason);
    } else if (processingPhase == CONTENT) {
        // If we are in the CONTENT phase, we check if tool parser start tag is found and if so, switch to TOOL_CALLS phase.
        // TOOL_CALLS is the only phase that can be processed after CONTENT.
        if (applyToolParser) {
            TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTags(toolParser->getParsingConfig().startTags);
            if (toolStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseToolCallChunk(tokens, finishReason);
            } else if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
                return std::nullopt;  // Wait for more chunks to determine if end tag is complete
            }
            return parseContentChunk();
        }
        return parseContentChunk();
    } else if (processingPhase == TOOL_CALLS_PROCESSING_TOOL) {
        // Active tool call: accumulate until the end tag, then transition to WAITING_FOR_TOOL
        // to determine whether another tool call or a content turn follows.
        TagLookupStatus toolEndTagStatus = streamOutputCache.lookupTag(toolParser->getParsingConfig().endTag);
        if (toolEndTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;  // Wait for more chunks to determine if end tag is complete
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
        // Check for content-turn start signal or generation end.
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
