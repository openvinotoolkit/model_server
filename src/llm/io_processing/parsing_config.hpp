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
//   startTags              — text-based start-boundary strings (matched against decoded text).
//   specialTokenStartTags  — start-boundary strings that are single special tokens.
//                            On construction the base class encodes each via the tokenizer
//                            (add_special_tokens=false). If the encoding resolves to exactly
//                            one token the ID is stored in resolvedStartTokenToTag and checked
//                            against incoming token IDs BEFORE any string matching.
//                            A start-tag token detected this way is synthesised as text before
//                            being passed to parseChunk(), so the sub-parser state machine
//                            receives the expected boundary string.
//   specialStartTags       — text-based tags checked only in the UNKNOWN phase (first output).
//   endTag                 — text-based end-boundary string (checked in TOOL_CALLS_PROCESSING_TOOL
//                            and REASONING phases).
//   contentTagsToErase     — control tags removed from plain-content deltas produced by
//                            OutputParser::parseContentChunk(). This keeps parser-specific
//                            control tokens out of content without hardcoding parser names.
//
// Tokenizer decode mode flags (evaluated by OutputParser::needSpecialTokensForCurrentDecode):
//   alwaysNeedsSpecialTokens          — Decode with skip_special_tokens=false at all times.
//                                       For parsers where special tokens appear throughout the
//                                       output (reasoning channels, GptOss, Gemma4 reasoning).
//   toolCallPhaseNeedsSpecialTokens   — Decode with skip_special_tokens=false only while in
//                                       TOOL_CALLS_PROCESSING_TOOL phase. For parsers whose
//                                       internal parseChunk() searches for special-token strings
//                                       mid-call (Gemma4 tool, LFM2, devstral).
//
// Parsers that need special tokens nowhere (Llama3, Hermes3, Phi4, Mistral, Qwen3, Qwen3Coder)
// leave both flags false. Special-token start boundaries are detected via token IDs and
// synthesised into text, so no special-token decode is required for phase detection.
struct ParsingConfig {
    std::vector<std::string> startTags;
    std::vector<std::string> specialTokenStartTags;
    std::vector<std::string> specialStartTags;
    std::string              endTag;
    std::vector<std::string> contentTagsToErase;

    bool alwaysNeedsSpecialTokens        = false;
    bool toolCallPhaseNeedsSpecialTokens = false;
};

}  // namespace ovms
