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

#include "../../logging.hpp"
#include "../../stringutils.hpp"
#include "output_parser.hpp"
#include "llama3/tool_parser.hpp"
#include "hermes3/tool_parser.hpp"
#include "phi4/tool_parser.hpp"
#include "mistral/tool_parser.hpp"
#include "gptoss/tool_parser.hpp"
#include "qwen3/reasoning_parser.hpp"
#include "qwen3coder/qwen3coder_tool_parser.hpp"
#include "gptoss/reasoning_parser.hpp"

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

OutputParser::TagLookupStatus OutputParser::StreamOutputCache::lookupTags(const std::unordered_set<std::string>& tags) const {
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

rapidjson::Document OutputParser::parseContentChunk(ProcessingPhase newPhase) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("delta");
    writer.StartObject();
    writer.String("content");
    writer.String(streamOutputCache.getBuffer().c_str());
    writer.EndObject();
    writer.EndObject();
    rapidjson::Document doc;
    doc.Parse(buffer.GetString());
    streamOutputCache.clear();
    processingPhase = newPhase;
    return doc;
}

std::optional<rapidjson::Document> OutputParser::parseToolCallChunk(ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase) {
    if (!toolParser) {
        throw std::runtime_error("Tool parser is not available, cannot parse tool call chunk");
    }
    auto result = toolParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
    streamOutputCache.clear();
    processingPhase = newPhase;
    return result;
}

std::optional<rapidjson::Document> OutputParser::parseReasoningChunk(ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase) {
    if (!reasoningParser) {
        throw std::runtime_error("Reasoning parser is not available, cannot parse reasoning chunk");
    }
    auto result = reasoningParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
    streamOutputCache.clear();
    processingPhase = newPhase;
    return result;
}

OutputParser::OutputParser(ov::genai::Tokenizer& tokenizer, const std::string toolParserName, const std::string reasoningParserName, const ToolsSchemas_t& toolNameSchemaMap) :
    tokenizer(tokenizer) {
    SPDLOG_TRACE("OutputParser created with toolNameSchemaMap of size: {}", toolNameSchemaMap.size());
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
    } else if (!toolParserName.empty()) {
        throw std::runtime_error("Unsupported tool parser: " + toolParserName);
    }

    if (reasoningParserName == "qwen3") {
        reasoningParser = std::make_unique<Qwen3ReasoningParser>(tokenizer);
    } else if (reasoningParserName == "gptoss") {
        reasoningParser = std::make_unique<GptOssReasoningParser>(tokenizer);
    } else if (!reasoningParserName.empty()) {
        throw std::runtime_error("Unsupported reasoning parser: " + reasoningParserName);
    }

    if (toolParser && reasoningParser) {
        if (toolParser->requiresStreamingWithSpecialTokens() != reasoningParser->requiresStreamingWithSpecialTokens()) {
            throw std::runtime_error("Cannot use tool parser " + toolParserName + " with reasoning parser " + reasoningParserName +
                                     " as they have different requirements for special tokens in streaming mode");
        }
    }
}

bool OutputParser::isToolParserAvailable() const {
    return toolParser != nullptr;
}

bool OutputParser::isReasoningParserAvailable() const {
    return reasoningParser != nullptr;
}

void OutputParser::enableImmediateToolParsing() {
    if (toolParser) {
        toolParser->enableImmediateParsing();
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool parser is not available, cannot enable zero trigger tool parsing");
    }
}

std::string OutputParser::getToolParserStartTag() const {
    if (toolParser) {
        return toolParser->getParsingStartTag();
    } else {
        throw std::runtime_error("Tool parser is not available, cannot get start tag");
    }
}

ParsedOutput OutputParser::parse(const std::vector<int64_t>& generatedTokens, const bool toolsAvailable) {
    // Model output is processed by the chain of parsers. Each parser extracts relevant part of the output and fills the ParsedOutput structure.
    // At the beginning, the content field of ParsedOutput is already filled with decoded content from generatedTokens.
    // When parser extracts relevant information, it should remove it from the content field, so we don't duplicate it in the final output.

    if (spdlog::default_logger_raw()->level() == spdlog::level::trace) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Raw model output: {}", tokenizer.decode(generatedTokens, ov::genai::skip_special_tokens(false)));
    }
    ParsedOutput parsedOutput;
    parsedOutput.content = tokenizer.decode(generatedTokens);
    if (reasoningParser) {
        reasoningParser->parse(parsedOutput, generatedTokens);
    }
    // We run tool parser only if the parser is available and tools have been provided in the request.
    if (toolParser && toolsAvailable) {
        toolParser->parse(parsedOutput, generatedTokens);
    }
    return parsedOutput;
}

std::optional<rapidjson::Document> OutputParser::parseChunk(const std::string& chunkResponse, const bool toolsAvailable, ov::genai::GenerationFinishReason finishReason) {
    /*
    Using appropriate parser based on the current processing phase
    Call to this method should return either result from parserContentChunk, parseToolCallChunk, parseReasoningChunk when we can determine the phase
    or std::nullopt when we are waiting for more chunks to determine if we should switch phase or not.
    Note that mentioned methods do not take chunk as argument, they read it from streamOutputCache and are responsible for clearing the cache,
    so only use those methods or return nullopt.
    */

    bool reasoningParserExistsAndSupportsStreaming = reasoningParser && !reasoningParser->getParsingStartTag().empty() && !reasoningParser->getParsingEndTag().empty();
    bool toolParserExistsAndSupportsStreaming = toolParser && !toolParser->getParsingStartTag().empty();  // FIXME why not check for parsingEntTag not empty?
    bool applyToolParser = toolParserExistsAndSupportsStreaming && toolsAvailable;

    if (applyToolParser && toolParser->isImmediateParsingEnabled() && processingPhase == UNKNOWN) {
        // If zero trigger parsing is enabled, we assume the start tag has been injected to the prompt.
        streamOutputCache.add(toolParser->getParsingStartTag());
    }

    streamOutputCache.add(chunkResponse);

    if (processingPhase == UNKNOWN) {
        // If we are in the UNKNOWN phase, we need to determine if we should switch to CONTENT, REASONING, or TOOL_CALLS phase.
        TagLookupStatus anyStartTagStatus = TagLookupStatus::NOT_FOUND;
        if (reasoningParserExistsAndSupportsStreaming) {
            // Check if reasoning start tag has been received
            TagLookupStatus reasoningStartTagStatus = streamOutputCache.lookupTag(reasoningParser->getParsingStartTag());
            if (reasoningStartTagStatus == TagLookupStatus::NOT_FOUND) {
                // If reasoning start tag is not found, check if any of the special start tags are found
                reasoningStartTagStatus = streamOutputCache.lookupTags(reasoningParser->getSpecialParsingStartTags());
            }
            if (reasoningStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseReasoningChunk(finishReason);
            }  // else startTagStatus is FOUND_INCOMPLETE or NOT_FOUND, we continue processing, so potential tool parser start tag is not missed
            anyStartTagStatus = reasoningStartTagStatus;
        }

        if (applyToolParser) {
            if (toolParser->isImmediateParsingEnabled()) {
                // If zero trigger parsing is enabled, we assume the start tag has been injected to the prompt, but for the unified parsing logic,
                // we still parse it to put parser in a proper state.
                return parseToolCallChunk(finishReason);
            } else {
                // Check if tool call start tag has been received
                TagLookupStatus toolCallStartTagStatus = streamOutputCache.lookupTag(toolParser->getParsingStartTag());
                if (toolCallStartTagStatus == TagLookupStatus::NOT_FOUND) {
                    // If tool call start tag is not found, check if any of the special start tags are found
                    toolCallStartTagStatus = streamOutputCache.lookupTags(toolParser->getSpecialParsingStartTags());
                }
                if (toolCallStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                    return parseToolCallChunk(finishReason);
                }  // else startTagStatus is FOUND_INCOMPLETE or NOT_FOUND, we continue processing
                if (toolCallStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE) {
                    anyStartTagStatus = toolCallStartTagStatus;  // We have at least one incomplete start tag
                }
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
        TagLookupStatus endTagStatus = streamOutputCache.lookupTag(reasoningParser->getParsingEndTag());
        if (endTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            // Switch back to UNKNOWN phase (we can have either CONTENT or TOOL_CALLS next)
            return parseReasoningChunk(finishReason, UNKNOWN);
        } else if (endTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;  // Wait for more chunks to determine if end tag is complete
        }
        return parseReasoningChunk(finishReason);
    } else if (processingPhase == CONTENT) {
        // If we are in the CONTENT phase, we check if tool parser start tag is found and if so, switch to TOOL_CALLS phase.
        // TOOL_CALLS is the only phase that can be processed after CONTENT.
        if (applyToolParser) {
            TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTag(toolParser->getParsingStartTag());
            if (toolStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                return parseToolCallChunk(finishReason);
            } else if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
                return std::nullopt;  // Wait for more chunks to determine if end tag is complete
            }
            return parseContentChunk();
        }
        return parseContentChunk();
    } else if (processingPhase == TOOL_CALLS_PROCESSING_TOOL) {
        // Processing TOOL_CALLS is the last phase, so we always return the result of tool parser.
        TagLookupStatus toolEndTagStatus = streamOutputCache.lookupTag(toolParser->getParsingEndTag());
        if (toolEndTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;  // Wait for more chunks to determine if end tag is complete
        }
        if (toolEndTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            // If tool call has finished, we switch to waiting for next tool call as tool calls in the last phase,
            // so we either get next tool call or finish processing.
            return parseToolCallChunk(finishReason, TOOL_CALLS_WAITING_FOR_TOOL);
        }
        return parseToolCallChunk(finishReason);
    } else if (processingPhase == TOOL_CALLS_WAITING_FOR_TOOL) {
        // In this phase we are waiting for next tool call or finish of generation.
        // If we get next tool call start tag, we switch to TOOL_CALLS phase, otherwise if generation finishes we switch to CONTENT phase to flush any remaining content.
        TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTag(toolParser->getParsingStartTag());
        if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;  // Wait for more chunks to determine if start tag is complete
        }
        if (toolStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            // If tool call has started, we switch back to processing tool phase.
            return parseToolCallChunk(finishReason, TOOL_CALLS_PROCESSING_TOOL);
        }
        return parseToolCallChunk(finishReason);
    } else {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Unexpected processing phase: {}", static_cast<int>(processingPhase));
        throw std::runtime_error("Unexpected error during stream output parsing");
    }
}
}  // namespace ovms
