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
#include <unordered_set>

#include "../../logging.hpp"
#include "output_parser.hpp"
#include "llama3/tool_parser.hpp"
#include "hermes3/tool_parser.hpp"
#include "phi4/tool_parser.hpp"
#include "mistral/tool_parser.hpp"
#include "openai/tool_parser.hpp"
#include "qwen3/reasoning_parser.hpp"
#include "openai/reasoning_parser.hpp"

namespace ovms {

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
    } else if (toolParserName == "gpt") {
        toolParser = std::make_unique<GptToolParser>(tokenizer);
    } else if (!toolParserName.empty()) {
        throw std::runtime_error("Unsupported tool parser: " + toolParserName);
    }

    if (reasoningParserName == "qwen3") {
        reasoningParser = std::make_unique<Qwen3ReasoningParser>(tokenizer);
    } else if (reasoningParserName == "gpt") {
        reasoningParser = std::make_unique<GptReasoningParser>(tokenizer);
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
    // Call to this method should always return either result from parser parseChunk implementation or common parseContentChunk method.
    // If for any processing phase a nullopt should be returned, it should be done in the parser implementation.
    // Do not return nullopt directly from this method.

    bool reasoningParserExistsAndSupportsStreaming = reasoningParser && !reasoningParser->getParsingStartTag().empty() && !reasoningParser->getParsingEndTag().empty();
    bool toolParserExistsAndSupportsStreaming = toolParser && !toolParser->getParsingStartTag().empty();
    bool applyToolParser = toolParserExistsAndSupportsStreaming && toolsAvailable;

    if (processingPhase == UNKNOWN) {
        // If we are in the UNKNOWN phase, we need to determine if we should switch to CONTENT, REASONING, or TOOL_CALLS phase.
        if (reasoningParserExistsAndSupportsStreaming && isParsingTagPartOfChunk(chunkResponse, reasoningParser->getParsingStartTag())) {
            processingPhase = REASONING;
            return reasoningParser->parseChunk(chunkResponse, finishReason);
        } else if (applyToolParser) {
            if (isParsingTagPartOfChunk(chunkResponse, toolParser->getParsingStartTag()) || chunkContainsSpecialParsingTag(chunkResponse, toolParser->getSpecialParsingStartTags())) {
                processingPhase = TOOL_CALLS;
                return toolParser->parseChunk(chunkResponse, finishReason);
            } else if (toolParser->isImmediateParsingEnabled()) {
                // If zero trigger parsing is enabled, we assume the start tag has been injected to the prompt, but for the unified parsing logic,
                // we still parse it to put parser in a proper state.
                processingPhase = TOOL_CALLS;
                toolParser->parseChunk(toolParser->getParsingStartTag(), finishReason);
                return toolParser->parseChunk(chunkResponse, finishReason);
            } else {
                processingPhase = CONTENT;
                return parseContentChunk(chunkResponse);
            }
        } else {
            processingPhase = CONTENT;
            return parseContentChunk(chunkResponse);
        }
    } else if (processingPhase == REASONING) {
        // If we are in the REASONING phase, we check if parsing end tag is found and if so, switch to UNKNOWN phase.
        if (isParsingTagPartOfChunk(chunkResponse, reasoningParser->getParsingEndTag())) {
            processingPhase = UNKNOWN;  // Switch back to UNKNOWN phase (we can have either CONTENT or TOOL_CALLS next)
        }
        return reasoningParser->parseChunk(chunkResponse, finishReason);
    } else if (processingPhase == CONTENT) {
        // If we are in the CONTENT phase, we check if tool parser start tag is found and if so, switch to TOOL_CALLS phase.
        // TOOL_CALLS is the only phase that can be processed after CONTENT.
        if (applyToolParser && isParsingTagPartOfChunk(chunkResponse, toolParser->getParsingStartTag())) {
            processingPhase = TOOL_CALLS;
            return toolParser->parseChunk(chunkResponse, finishReason);
        } else {
            return parseContentChunk(chunkResponse);
        }
    } else if (processingPhase == TOOL_CALLS) {
        // Processing TOOL_CALLS is the last phase, so we always return the result of tool parser.
        return toolParser->parseChunk(chunkResponse, finishReason);
    } else {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Unexpected processing phase: {}", static_cast<int>(processingPhase));
        throw std::runtime_error("Unexpected error during stream output parsing");
    }
}
}  // namespace ovms
