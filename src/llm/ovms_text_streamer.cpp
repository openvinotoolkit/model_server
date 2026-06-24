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
#include "ovms_text_streamer.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include <rapidjson/document.h>

namespace {
// Matches GenAI's is_incomplete() in text_streamer.cpp.
// The tokenizer outputs U+FFFD (\xef\xbf\xbd) as a 3-byte replacement
// character when the token cache ends with an incomplete multibyte sequence.
bool is_incomplete(const std::string& text) {
    constexpr char replacement[] = "\xef\xbf\xbd";
    return text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0;
}
}  // namespace

namespace ovms {

// No-op callback passed to the base TextStreamer constructor.
// OVMSTextStreamer overrides write(int64_t) and end() completely, so the base
// callback is never invoked through the normal TextStreamer code path.
static ov::genai::StreamingStatus noop_string_callback(std::string) {
    return ov::genai::StreamingStatus::RUNNING;
}

OVMSTextStreamer::OVMSTextStreamer(
    const ov::genai::Tokenizer& tokenizer,
    std::shared_ptr<OutputParser> output_parser,
    bool tools_available,
    Callback callback,
    const ov::AnyMap& decode_params) :
    ov::genai::TextStreamer(tokenizer, noop_string_callback, decode_params),
    m_output_parser(output_parser),
    m_tools_available(tools_available),
    m_callback(std::move(callback)) {}

// -----------------------------------------------------------------------------
// write(int64_t) — owned decode loop (does NOT delegate to TextStreamer::write)
//
// Replicates TextStreamer's flush heuristics:
//   1. Newline flush: emit immediately when text ends with '\n'.
//   2. Incomplete UTF-8 guard: if text ends with U+FFFD replacement char, mark as -1.
//   3. Delay buffer: hold back the last DELAY_N_TOKENS positions before flushing.
//
// Operates directly on the protected members inherited from TextStreamer:
//   m_tokens_cache, m_decoded_lengths, m_printed_len,
//   m_tokenizer, m_additional_detokenization_params.
// -----------------------------------------------------------------------------
ov::genai::StreamingStatus OVMSTextStreamer::write(int64_t token) {
    m_tokens_cache.push_back(token);
    const std::string text = m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
    m_decoded_lengths.push_back(static_cast<int64_t>(text.size()));

    // 1. Newline flush: emit everything and reset.
    if (!text.empty() && text.back() == '\n' && text.size() > m_printed_len) {
        const auto status = flush_chunk(text, text.size(), ov::genai::GenerationFinishReason::NONE);
        m_tokens_cache.clear();
        m_decoded_lengths.clear();
        m_printed_len = 0;
        return status;
    }

    // 2. Incomplete UTF-8: decoded text ends with the Unicode replacement character
    //    (U+FFFD, \xef\xbf\xbd) — the tokenizer's signal for a partial multibyte
    //    sequence. Mark this slot as -1 so the delay check skips it.
    if (is_incomplete(text)) {
        m_decoded_lengths.back() = -1;
        return ov::genai::StreamingStatus::RUNNING;
    }

    // 3. Delay buffer: need at least DELAY_N_TOKENS entries before flushing.
    const size_t n = m_decoded_lengths.size();
    if (n < DELAY_N_TOKENS) {
        return ov::genai::StreamingStatus::RUNNING;
    }

    // Flush up to the decoded length DELAY_N_TOKENS positions from the end.
    const int64_t print_until_len = m_decoded_lengths[n - DELAY_N_TOKENS];
    if (print_until_len <= 0 || static_cast<size_t>(print_until_len) <= m_printed_len) {
        return ov::genai::StreamingStatus::RUNNING;
    }

    return flush_chunk(text, static_cast<size_t>(print_until_len),
        ov::genai::GenerationFinishReason::NONE);
}

ov::genai::StreamingStatus OVMSTextStreamer::write(const std::vector<int64_t>& tokens) {
    ov::genai::StreamingStatus status = ov::genai::StreamingStatus::RUNNING;
    for (const int64_t token : tokens) {
        status = write(token);
        if (status != ov::genai::StreamingStatus::RUNNING) {
            return status;
        }
    }
    return status;
}

// -----------------------------------------------------------------------------
//
// Decodes the remaining token cache (up to DELAY_N_TOKENS - 1 tokens that
// write() deliberately held back) and flushes with GenerationFinishReason::STOP.
//
// Does NOT call TextStreamer::end() — the base would fire its no-op callback
// and attempt to clear the protected state that we have already managed.
// -----------------------------------------------------------------------------
void OVMSTextStreamer::end() {
    // Always send a STOP flush so parsers that rely on finish_reason == STOP for
    // cleanup (e.g. Hermes3 closing the argument string) receive the signal even
    // when m_tokens_cache was cleared by a prior newline flush in write().
    if (!m_tokens_cache.empty()) {
        const std::string text = m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
        if (text.size() > m_printed_len) {
            flush_chunk(text, text.size(), ov::genai::GenerationFinishReason::STOP);
        } else {
            flush_chunk(text, m_printed_len, ov::genai::GenerationFinishReason::STOP);
        }
    } else {
        // Cache already cleared (e.g. by a newline flush). No new text, but the
        // STOP signal must still reach the parser.
        flush_chunk("", 0, ov::genai::GenerationFinishReason::STOP);
    }
    m_tokens_cache.clear();
    m_decoded_lengths.clear();
    m_printed_len = 0;
}

// -----------------------------------------------------------------------------
// flush_chunk — compute token slice, call OutputParser::parseChunk, fire callback
//
// Token slice computation mirrors ov::genai::TextParserStreamer::write(string):
//   first_idx = upper_bound(m_decoded_lengths, m_printed_len)
//   last_idx  = upper_bound(m_decoded_lengths, print_until)
//
// The resulting tokens sub-vector is passed to OutputParser::parseChunk alongside
// the decoded text chunk. All existing parsers ignore tokens in Phase 1; the
// parameter is available for future phase-aware parsers.
//
// Callback is always fired when:
//   - parseChunk returns a non-nullopt Document, OR
//   - finish_reason != NONE (ensures finish_reason chunk is always emitted even
//     when the parser produces no final delta, e.g. after a completed tool call).
// When no Document is available for a STOP flush, an empty Document{} is passed.
// -----------------------------------------------------------------------------
ov::genai::StreamingStatus OVMSTextStreamer::flush_chunk(
    const std::string& text,
    size_t print_until,
    ov::genai::GenerationFinishReason finish_reason) {
    const std::string chunk{text.data() + m_printed_len, print_until - m_printed_len};

    // Token slice: tokens whose decoded positions fall in (m_printed_len, print_until].
    const auto first_it = std::upper_bound(
        m_decoded_lengths.begin(), m_decoded_lengths.end(),
        static_cast<int64_t>(m_printed_len));
    const auto last_it = std::upper_bound(
        m_decoded_lengths.begin(), m_decoded_lengths.end(),
        static_cast<int64_t>(print_until));
    const auto first_idx = static_cast<size_t>(first_it - m_decoded_lengths.begin());
    const auto last_idx = static_cast<size_t>(last_it - m_decoded_lengths.begin());
    const std::vector<int64_t> tokens(
        m_tokens_cache.begin() + static_cast<std::ptrdiff_t>(first_idx),
        m_tokens_cache.begin() + static_cast<std::ptrdiff_t>(last_idx));

    m_printed_len = print_until;

    std::optional<rapidjson::Document> delta;
    if (m_output_parser != nullptr) {
        delta = m_output_parser->parseChunk(chunk, tokens, m_tools_available, finish_reason);
    } else if (!chunk.empty()) {
        // No parser: wrap raw text in a trivial {"delta":{"content":"..."}} document.
        // Skip when chunk is empty (e.g. STOP flush after a newline-clearing write).
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
        rapidjson::Value delta_obj(rapidjson::kObjectType);
        delta_obj.AddMember("content",
            rapidjson::Value(chunk.c_str(), alloc),
            alloc);
        doc.AddMember("delta", delta_obj, alloc);
        delta = std::move(doc);
    }

    const bool isLast = (finish_reason != ov::genai::GenerationFinishReason::NONE);
    if (delta.has_value()) {
        return m_callback(std::move(*delta), isLast);
    }
    if (isLast) {
        // Parser produced no delta for the final flush (e.g. generation ended on a
        // special token the parser absorbed). Still fire the callback with an empty
        // Document so the caller can emit the finish_reason chunk.
        return m_callback(rapidjson::Document{}, true);
    }
    return ov::genai::StreamingStatus::RUNNING;
}

}  // namespace ovms
