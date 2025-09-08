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
#include "output_parser.hpp"
#include "llama3/tool_parser.hpp"
#include "hermes3/tool_parser.hpp"
#include "phi4/tool_parser.hpp"
#include "mistral/tool_parser.hpp"
#include "qwen3/reasoning_parser.hpp"

namespace ovms {

bool stringsOverlap(const std::string& lhs, const std::string& rhs) {
    size_t minLength = std::min(lhs.size(), rhs.size());
    for (size_t len = 1; len <= minLength; ++len) {
        if (lhs.compare(lhs.size() - len, len, rhs, 0, len) == 0) {
            return true;
        }
    }
    return false;
}

OutputParser::TagLookupStatus OutputParser::StreamOutputCache::lookupTag(const std::string& tag) const {
    if (tag.size() > buffer.size()) {
        /* 
        If the tag is longer than the buffer, we check:
           a) if the buffer is a prefix of the tag (whole cache is matched)
           b) if the buffer and tag overlap (part of the cache is matched)
        in both cases we assume that tag may appear in the future, so we return FOUND_INCOMPLETE
        otherwise we return NOT_FOUND
        */
        if (tag.compare(0, buffer.size(), buffer) == 0 || stringsOverlap(buffer, tag)) {
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
    TagLookupStatus tagLookupStatus = TagLookupStatus::NOT_FOUND;
    for (const auto& tag : tags) {
        tagLookupStatus = lookupTag(tag);
        if (tagLookupStatus == TagLookupStatus::FOUND_COMPLETE) {
            break;
        }
    }
    return tagLookupStatus;
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

rapidjson::Document OutputParser::parseContentChunk(const std::string& chunk) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("delta");
    writer.StartObject();
    writer.String("content");
    writer.String(chunk.c_str());
    writer.EndObject();
    writer.EndObject();
    rapidjson::Document doc;
    doc.Parse(buffer.GetString());
    return doc;
}

OutputParser::OutputParser(ov::genai::Tokenizer& tokenizer, const std::string toolParserName, const std::string reasoningParserName) :
    tokenizer(tokenizer) {
    if (toolParserName == "llama3") {
        toolParser = std::make_unique<Llama3ToolParser>(tokenizer);
    } else if (toolParserName == "hermes3") {
        toolParser = std::make_unique<Hermes3ToolParser>(tokenizer);
    } else if (toolParserName == "phi4") {
        toolParser = std::make_unique<Phi4ToolParser>(tokenizer);
    } else if (toolParserName == "mistral") {
        toolParser = std::make_unique<MistralToolParser>(tokenizer);
    } else if (!toolParserName.empty()) {
        throw std::runtime_error("Unsupported tool parser: " + toolParserName);
    }

    if (reasoningParserName == "qwen3") {
        reasoningParser = std::make_unique<Qwen3ReasoningParser>(tokenizer);
    } else if (!reasoningParserName.empty()) {
        throw std::runtime_error("Unsupported reasoning parser: " + reasoningParserName);
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

static inline bool isParsingTagPartOfChunk(const std::string& chunk, const std::string& parsingTag) {
    return chunk.find(parsingTag) != std::string::npos;
}

static inline bool chunkContainsSpecialParsingTag(const std::string& chunk, const std::unordered_set<std::string>& specialParsingTags) {
    for (const auto& tag : specialParsingTags) {
        if (isParsingTagPartOfChunk(chunk, tag)) {
            return true;
        }
    }
    return false;
}

std::optional<rapidjson::Document> OutputParser::parseChunk(const std::string& chunkResponse, const bool toolsAvailable, ov::genai::GenerationFinishReason finishReason) {
    // Using appropriate parser based on the current processing phase
    // Call to this method should return either result from parser parseChunk implementation when we are in particular phase
    // or std::nullopt when we are waiting for more chunks to determine if we should switch phase or not.

    bool reasoningParserExistsAndSupportsStreaming = reasoningParser && !reasoningParser->getParsingStartTag().empty() && !reasoningParser->getParsingEndTag().empty();
    bool toolParserExistsAndSupportsStreaming = toolParser && !toolParser->getParsingStartTag().empty();
    bool applyToolParser = toolParserExistsAndSupportsStreaming && toolsAvailable;

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
                auto result = reasoningParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
                processingPhase = REASONING;
                streamOutputCache.clear();
                return result;
            }  // else startTagStatus is FOUND_INCOMPLETE or NOT_FOUND, we continue processing, so potential tool parser start tag is not missed
            anyStartTagStatus = reasoningStartTagStatus;
        }

        if (applyToolParser) {
            if (toolParser->isImmediateParsingEnabled()) {
                // If zero trigger parsing is enabled, we assume the start tag has been injected to the prompt, but for the unified parsing logic,
                // we still parse it to put parser in a proper state.
                processingPhase = TOOL_CALLS;
                toolParser->parseChunk(toolParser->getParsingStartTag(), finishReason);
                return toolParser->parseChunk(chunkResponse, finishReason);
            } else {
                // Check if tool call start tag has been received
                TagLookupStatus toolCallStartTagStatus = streamOutputCache.lookupTag(toolParser->getParsingStartTag());
                if (toolCallStartTagStatus == TagLookupStatus::NOT_FOUND) {
                    // If tool call start tag is not found, check if any of the special start tags are found
                    toolCallStartTagStatus = streamOutputCache.lookupTags(toolParser->getSpecialParsingStartTags());
                }
                if (toolCallStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                    auto result = toolParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
                    processingPhase = TOOL_CALLS;
                    streamOutputCache.clear();
                    return result;
                }  // else startTagStatus is FOUND_INCOMPLETE or NOT_FOUND, we continue processing
                if (toolCallStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE) {
                    anyStartTagStatus = toolCallStartTagStatus;  // We have at least one incomplete start tag
                }
            }
        }

        if ((!reasoningParserExistsAndSupportsStreaming && !applyToolParser) || finishReason != ov::genai::GenerationFinishReason::NONE || anyStartTagStatus == TagLookupStatus::NOT_FOUND) {
            // If no special parsers are available, generation has finished or we have no start tags we just return content chunks and switch to CONTENT phase.
            auto content = streamOutputCache.getBuffer();
            streamOutputCache.clear();
            processingPhase = CONTENT;
            return parseContentChunk(content);
        }
        // If we are here, it means we have incomplete start tag for either reasoning or tool parser, so we wait for more chunks
        return std::nullopt;
    } else if (processingPhase == REASONING) {
        // If we are in the REASONING phase, we check if parsing end tag is found and if so, switch to UNKNOWN phase.
        TagLookupStatus endTagStatus = streamOutputCache.lookupTag(reasoningParser->getParsingEndTag());
        if (endTagStatus == TagLookupStatus::FOUND_COMPLETE) {
            auto result = reasoningParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
            processingPhase = UNKNOWN;  // Switch back to UNKNOWN phase (we can have either CONTENT or TOOL_CALLS next)
            streamOutputCache.clear();
            return result;
        } else if (endTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;  // Wait for more chunks to determine if end tag is complete
        }
        auto result = reasoningParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
        streamOutputCache.clear();
        return result;
    } else if (processingPhase == CONTENT) {
        // If we are in the CONTENT phase, we check if tool parser start tag is found and if so, switch to TOOL_CALLS phase.
        // TOOL_CALLS is the only phase that can be processed after CONTENT.
        if (applyToolParser) {
            TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTag(toolParser->getParsingStartTag());
            if (toolStartTagStatus == TagLookupStatus::FOUND_COMPLETE) {
                auto result = toolParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
                processingPhase = TOOL_CALLS;
                streamOutputCache.clear();
                return result;
            } else if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
                return std::nullopt;  // Wait for more chunks to determine if end tag is complete
            }
            auto result = parseContentChunk(streamOutputCache.getBuffer());
            streamOutputCache.clear();
            return result;
        }
        return parseContentChunk(chunkResponse);
    } else if (processingPhase == TOOL_CALLS) {
        // Processing TOOL_CALLS is the last phase, so we always return the result of tool parser.
        TagLookupStatus toolStartTagStatus = streamOutputCache.lookupTag(toolParser->getParsingEndTag());
        if (toolStartTagStatus == TagLookupStatus::FOUND_INCOMPLETE && finishReason == ov::genai::GenerationFinishReason::NONE) {
            return std::nullopt;  // Wait for more chunks to determine if end tag is complete
        }
        auto result = toolParser->parseChunk(streamOutputCache.getBuffer(), finishReason);
        streamOutputCache.clear();
        return result;
    } else {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Unexpected processing phase: {}", static_cast<int>(processingPhase));
        throw std::runtime_error("Unexpected error during stream output parsing");
    }
}
}  // namespace ovms
