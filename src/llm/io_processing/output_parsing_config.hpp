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
#pragma once

#include "model_generation_context.hpp"

#include <string>
#include <vector>

namespace ovms {

// Per-parser configuration for phase detection, generation startup, and tokenizer decode mode.
// Starts from predefined format settings or a config override. During construction, a parser
// may derive promptContinuationStartTags and startsInsidePhase from ModelGenerationContext.
// These request-specific settings describe how generated output continues the rendered prompt;
// they remain available across resets without retaining or replaying prompt text.
//
// Phase detection:
//   startTags          — text-based start-boundary strings (matched against decoded text).
//   tokenIdStartTags   — start-boundary strings that are single special tokens.
//                        On construction the base class encodes each via the tokenizer
//                        (add_special_tokens=false). If the encoding resolves to exactly
//                        one token the ID is stored in resolvedStartTokenToTag and checked
//                        against incoming token IDs BEFORE any string matching.
//                        A start-tag token detected this way is synthesised as text before
//                        being passed to parseChunk(), so the sub-parser state machine
//                        receives the expected boundary string.
//   unknownPhaseStartTags — additional starts checked while phase detection is UNKNOWN.
//                           UNKNOWN is re-entered after a phase ends, so these may identify
//                           later reasoning/tool segments, not only the first output tokens.
//   promptContinuationStartTags — start-tag continuations checked only at byte zero of the
//                                 first generated output when the prompt ends mid-header.
//                                 These identify the generated suffix; prompt text is not replayed.
//   endTags            — text-based end-boundary strings (checked in TOOL_CALLS_PROCESSING_TOOL
//                        and REASONING phases; the earliest matching boundary ends the phase).
//   stringsToErase     — strings stripped from this parser's output before emission
//                        (e.g. BOS/EOS tokens that leak due to special-token decode mode,
//                        or chat-template structural markers).
//
// Generation startup:
//   startsInsidePhase - the parser determines that the prompt already entered its phase.
//                       OutputParser can select that phase without waiting for a generated
//                       start tag (tool parsing still requires available tools). Reset restores
//                       this initial routing state; promptContinuationStartTags instead handle
//                       cases where the generated suffix must still identify the first phase.
//
// Tokenizer decode mode flag (evaluated by OutputParser::needSpecialTokensForCurrentDecode):
//   needsSpecialTokens — Decode with skip_special_tokens=false while this parser is in its
//                        active phase (REASONING for reasoning parsers; TOOL_CALLS_* for tool
//                        parsers). The parser's internal state machine relies on special-token
//                        strings being visible in the decoded text during that phase.
//
//   Parsers that detect phase boundaries via token IDs only (Llama3, Hermes3, Phi4, Mistral,
//   Qwen3, Qwen3Coder) leave this flag false — the proactive token-ID switch in OVMSTextStreamer
//   synthesises the start-tag text without requiring special-token decode in the active phase.
//
//   Whether the content/unknown phase also needs special tokens is determined at the
//   OutputParser level via defaultDecodingWithSpecialTokens, not in the per-parser config.
//
// Content/unknown phase decode mode:
//   defaultDecodingWithSpecialTokens — when true, decoding uses skip_special_tokens=false
//                                      even in the content/unknown phase. Set by parsers
//                                      whose model format emits structural special tokens
//                                      before their own active phase begins (e.g. GptOss,
//                                      devstral, minicpm5).
struct OutputParsingConfig {
    std::vector<std::string> startTags;
    std::vector<std::string> tokenIdStartTags;
    std::vector<std::string> unknownPhaseStartTags;
    std::vector<std::string> promptContinuationStartTags;
    std::vector<std::string> endTags;
    std::vector<std::string> stringsToErase;

    // The prompt already entered this parser's phase; reset preserves this initial state.
    bool startsInsidePhase = false;
    bool needsSpecialTokens = false;
    // See comment block above.
    bool defaultDecodingWithSpecialTokens = false;
};

/** Returns default reasoning boundaries without constructing a parser or resolving token IDs. */
OutputParsingConfig getDefaultReasoningParsingConfig(const std::string& parserName);

/** Determines the generation's reasoning state from parser metadata and the rendered prompt. */
ModelGenerationContext determineModelGenerationContext(const OutputParsingConfig& config, const std::string& renderedPrompt);

}  // namespace ovms
