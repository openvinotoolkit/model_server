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
#pragma once

#include <memory>
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "base_output_parser.hpp"

#include "src/llm/apis/tool_schema_wrapper.hpp"

namespace ovms {

// OutputParser orchestrates the streaming parsing pipeline.
//
// Responsibilities of OutputParser (the orchestrator):
//   - Phase lifecycle: detect phase transitions by looking for start/end tags declared
//     in each parser's OutputParsingConfig; switch the active phase accordingly.
//   - Buffer management: accumulate decoded text in StreamOutputCache, hold it while a
//     tag is only partially matched, flush it to the active parser when a boundary is
//     confirmed, and carry over any bytes that trail a phase-end tag so they seed the
//     next phase without loss.
//   - Routing: deliver each flush exclusively to the parser that owns the current phase.
//     A specific parser's parseChunk() is only ever called with text that belongs to its
//     active phase — the parser does not need to detect or handle phase transitions.
//   - Coordination: manage the interplay between tool, reasoning, and content parsers
//     across the full generation sequence.
//
// Responsibilities of BaseOutputParser subclasses (the specific parsers):
//   - Declare phase boundaries by returning a correctly populated OutputParsingConfig
//     (startTags, endTag, preambleStartTags, etc.).  This is the sole coupling point
//     with the orchestrator — no knowledge of OutputParser internals is required.
//   - Implement parseChunk() to process the text it receives during its active phase.
//     The parser may maintain arbitrary internal state and buffers to satisfy its own
//     format requirements; OutputParser does not inspect or constrain that state.
//   - Return a JSON delta (OpenAI streaming format) or nullopt to signal "nothing to
//     emit yet"; the orchestrator propagates that decision upstream unchanged.
//
// Design invariant: OutputParser must contain NO logic specific to any individual model
// format.  All format-specific behaviour must be encapsulated in the parser subclasses
// and expressed through their configuration and parseChunk() implementations.
class OutputParser {
    // Public types and enums
public:
    enum TagLookupStatus {
        NOT_FOUND,
        FOUND_COMPLETE,
        FOUND_INCOMPLETE
    };

    class StreamOutputCache {
        std::string buffer;

    public:
        TagLookupStatus lookupTag(const std::string& tag) const;
        TagLookupStatus lookupTags(const std::vector<std::string>& tags) const;
        void add(const std::string& chunk);
        void clear();
        const std::string& getBuffer() const;
    };

    enum ProcessingPhase {
        UNKNOWN,
        CONTENT,
        REASONING,
        TOOL_CALLS_PROCESSING_TOOL,
        TOOL_CALLS_WAITING_FOR_TOOL
    };

private:
    ov::genai::Tokenizer tokenizer;
    std::unique_ptr<BaseOutputParser> toolParser = nullptr;
    std::unique_ptr<BaseOutputParser> reasoningParser = nullptr;
    std::unique_ptr<BaseOutputParser> contentParser = nullptr;

    // Streaming related members
    ProcessingPhase processingPhase = UNKNOWN;
    StreamOutputCache streamOutputCache;
    bool implicitReasoningStart = false;

    // Baseline decode mode for content/unknown phases — true when the model/output format
    // needs special tokens visible before any parser-owned phase becomes active.
    // Set once in the constructor from parser names.
    bool defaultDecodingWithSpecialTokens = false;

    // Parsing methods below read chunks from streamOutputCache hence no string argument is needed

    // Regular content parsing method does not require finishReason as content is always parsed
    std::optional<rapidjson::Document> parseContentChunk(ProcessingPhase newPhase = CONTENT);

    std::optional<rapidjson::Document> parseToolCallChunk(const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase = TOOL_CALLS_PROCESSING_TOOL);
    std::optional<rapidjson::Document> parseReasoningChunk(const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason, ProcessingPhase newPhase = REASONING);

    // Configure parser to treat the output as already-in-reasoning from the first token.
    // Used when the chat template appends the reasoning start tag (e.g. "<think>\n") as
    // the prompt suffix - the model then emits only the reasoning body and the closing tag.
    // No-op when no reasoning parser is configured.
    void setImplicitReasoningStart(bool value);

public:
    OutputParser() = delete;
    explicit OutputParser(ov::genai::Tokenizer& tokenizer, const std::string toolParserName, const std::string reasoningParserName, const ToolsSchemas_t& toolNameSchemaMap);

    bool isToolParserAvailable() const;
    bool isReasoningParserAvailable() const;
    std::string getToolParserStartTag() const;

    // Reset streaming state and recreate parser instances to clear internal parser state.
    void resetStreamingState();

    // Auto-detect and apply implicit reasoning start based on the prompt produced by the chat template.
    void detectAndSetImplicitReasoningStart(const std::string& renderedPrompt);

    // Parse one decoded chunk in streaming mode.
    //
    // Contract:
    //   - Returns a JSON delta conforming to the OpenAI streaming API, or nullopt when no
    //     output can yet be produced (partial tag match, preamble stripping, etc.).
    //   - Processes AT MOST ONE phase per call.  If a chunk spans a phase boundary (e.g. a
    //     token whose text contains both an end tag and the start of the next phase), the bytes
    //     after the end tag are preserved in the internal buffer and processed on the next call.
    //     No content is ever discarded at a phase transition — "nothing properly parsed is lost".
    //   - Correctness requires at least one subsequent call after every phase transition.
    //     The caller must provide a final call with finishReason != NONE (typically an empty
    //     chunk) so the buffer is fully drained.  If that call is missing, any buffered
    //     remainder from the last transition will be silently dropped.
    //   - Known limitation: if finishReason != NONE arrives while a start tag is only
    //     partially matched (FOUND_INCOMPLETE), the partial text is flushed as content rather
    //     than held for completion — this is unavoidable without more tokens.
    //
    // Implementation must be fully generic: no parser-specific logic or special-casing of
    // individual model formats belongs here.  Behaviour must be driven entirely by the
    // configuration exposed through BaseOutputParser::getParsingConfig().
    //
    // tokens holds the token IDs that produced chunkResponse (informational; used for
    // token-ID-based phase-start detection in OVMSTextStreamer).
    std::optional<rapidjson::Document> parseChunk(const std::string& chunkResponse, const std::vector<int64_t>& tokens, const bool toolsAvailable, ov::genai::GenerationFinishReason finishReason);

    // Decide decode mode dynamically based on parser phase and user preference.
    // Content/unknown phases use defaultDecodingWithSpecialTokens OR user preference.
    // Reasoning/tool phases are driven solely by the active parser's needsSpecialTokens flag;
    // user preference does not override parser correctness requirements in those phases.
    bool needSpecialTokensForCurrentDecode(bool userWantsSpecialTokens = false) const;

    // If `tokenId` is a phase-start token, returns the corresponding tag string
    // (taken directly from resolvedStartTokenToTag — no tokenizer decode needed).
    // Tool phase-start tags are considered only when toolsAvailable is true.
    // Returns an empty string when the token is not a known phase-start token.
    // Used by OVMSTextStreamer to immediately flush the start-tag text without
    // going through the delay buffer.
    std::string getPhaseStartTagForToken(int64_t tokenId, bool toolsAvailable = true) const;
};
}  // namespace ovms
