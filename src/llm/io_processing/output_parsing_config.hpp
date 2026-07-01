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

#include <string>
#include <vector>

namespace ovms {

// Configuration for a parser's phase-boundary detection and tokenizer decode mode.
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
//   preambleStartTags  — text-based tags checked only in the UNKNOWN (preamble) phase.
//                        These are alternative entry points that cannot appear mid-stream.
//   endTag             — text-based end-boundary string (checked in TOOL_CALLS_PROCESSING_TOOL
//                        and REASONING phases).
//   contentTagsToErase — control tags removed from plain-content deltas produced by
//                        OutputParser::parseContentChunk(). This keeps parser-specific
//                        control tokens out of content without hardcoding parser names.
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
//   OutputParser level (defaultDecodingWithSpecialTokens), not in the per-parser config.
struct OutputParsingConfig {
    std::vector<std::string> startTags;
    std::vector<std::string> tokenIdStartTags;
    std::vector<std::string> preambleStartTags;
    std::string endTag;
    std::vector<std::string> contentTagsToErase;

    bool needsSpecialTokens = false;
};

}  // namespace ovms
