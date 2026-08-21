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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/text_streamer.hpp>

#include "io_processing/output_parser.hpp"

namespace ovms {

// OVMSTextStreamer is the bridge between the OpenVINO GenAI token generator and OutputParser.
// It is responsible for decoding raw token IDs into text and delivering correctly framed
// chunks to OutputParser::parseChunk().  Its behaviour is a precondition for OutputParser's
// correctness guarantees (see output_parser.hpp).
//
// Guarantees provided by OVMSTextStreamer:
//   - Ordered delivery: tokens are passed to OutputParser in the exact generation order,
//     one logical chunk at a time.
//   - Final flush: end() ALWAYS calls parseChunk("", [], finishReason=STOP) after all tokens
//     have been processed.  This is the "at least one subsequent call after every phase
//     transition" guarantee that OutputParser depends on to drain buffered remainders.
//   - Phase-aware decode mode: after every write(), the streamer queries
//     OutputParser::needSpecialTokensForCurrentDecode() and adjusts skip_special_tokens for
//     the next decode pass.  This ensures structural special tokens (e.g. <|im_end|>) are
//     visible as text during the phases whose parsers require them, and are suppressed
//     (noise-free) in the content/unknown phase.
//   - Token-ID phase detection: before decoding, write() checks
//     OutputParser::getPhaseStartTagForToken() for the incoming token ID.  If a match is
//     found, the delay buffer is flushed immediately, the start-tag text is injected directly
//     (without BPE decoding), and the decode mode is updated for the new phase — providing
//     zero-latency phase entry without waiting for BPE to confirm the tag string.
//   - Delay buffer: the last DELAY_N_TOKENS tokens are held back to prevent emitting partial
//     BPE-fused text mid-word.  Boundaries (phase starts, end()) force an immediate flush.
//
// OVMSTextStreamer does NOT perform any parsing, phase detection, or content routing.
// All of that is delegated to OutputParser.  The streamer's sole responsibility is to
// ensure OutputParser receives correctly decoded, correctly ordered, and correctly framed
// input — including the mandatory final call on end().
//
// Inherits ov::genai::TextStreamer to reuse its protected decode-loop state
// (m_tokenizer, m_tokens_cache, m_decoded_lengths, m_printed_len,
// m_additional_detokenization_params). write(int64_t) and end() are fully overridden;
// the no-op callback passed at construction is never invoked.
class OVMSTextStreamer : public ov::genai::TextStreamer {
public:
    // Callback receives a Delta and the isLast flag, and returns the streaming status.
    // For the finish-only case (nullopt from parseChunk + STOP finishReason),
    // a FinishDelta{} is passed so the caller can emit the finish_reason chunk.
    // isLast is true when finish_reason != NONE — callers that push into a DeltaChannel
    // should forward this flag to DeltaChannel::push() so the final delta and the
    // completion signal are observed atomically (no separate signalComplete() needed).
    using Callback = std::function<ov::genai::StreamingStatus(Delta, bool /*isLast*/)>;

    // outputParser may be nullptr (e.g. for the unary VLM path).
    // toolsAvailable must be evaluated after parseRequest() has processed the body.
    // decodeParams controls skip_special_tokens etc. — the value is used as the baseline
    // user preference; the parser's per-phase requirements are layered on top dynamically.
    OVMSTextStreamer(
        const ov::genai::Tokenizer& tokenizer,
        std::shared_ptr<OutputParser> output_parser,
        bool tools_available,
        Callback callback,
        const ov::AnyMap& decode_params);

    ov::genai::StreamingStatus write(int64_t token) override;
    ov::genai::StreamingStatus write(const std::vector<int64_t>& tokens) override;
    void end() override;

private:
    std::shared_ptr<OutputParser> m_output_parser;
    bool m_tools_available;
    Callback m_callback;
    // Whether the user's request specified skip_special_tokens=false.
    bool m_user_wants_special = false;
    // Whether the current decode pass should include special tokens (skip_special_tokens=false).
    // Kept in sync with m_additional_detokenization_params via apply_decode_params.
    bool m_decode_special_tokens = false;

    static constexpr size_t DELAY_N_TOKENS = 3;

    // Writes skip_special_tokens into m_additional_detokenization_params and updates m_decode_special_tokens.
    void apply_decode_params(bool decode_special_tokens);

    // Flushes pending cache and switches decode mode; returns a status if the token was consumed as a phase-start token.
    std::optional<ov::genai::StreamingStatus> handle_decoding_params_change(int64_t token);

    // Like write() but with immediate_flush=true skips the delay buffer and flushes the token's contribution right away.
    ov::genai::StreamingStatus write(int64_t token, bool immediate_flush);

    ov::genai::StreamingStatus flush_chunk(
        const std::string& text,
        size_t print_until,
        ov::genai::GenerationFinishReason finish_reason);

    // All token IDs received by write() in order, used for end() trace logging.
    std::vector<int64_t> m_all_tokens;
};

}  // namespace ovms
