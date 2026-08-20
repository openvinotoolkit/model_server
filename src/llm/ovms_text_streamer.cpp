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

#include "../logging.hpp"

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
    m_callback(std::move(callback)) {
    // Extract user's skip_special_tokens preference from decode_params.
    // The OV any-map stores it as a bool under the canonical key name.
    auto it = decode_params.find(ov::genai::skip_special_tokens.name());
    if (it != decode_params.end()) {
        try {
            // skip_special_tokens=true means we DON'T want special tokens.
            const bool skip_special = it->second.as<bool>();
            m_user_wants_special = !skip_special;
        } catch (...) {
        }
    }
    // Sync m_additional_detokenization_params with the parser's initial phase requirements.
    apply_decode_params(m_output_parser
                            ? m_output_parser->needSpecialTokensForCurrentDecode(m_user_wants_special)
                            : m_user_wants_special);
}

void OVMSTextStreamer::apply_decode_params(bool decode_special_tokens) {
    m_additional_detokenization_params[ov::genai::skip_special_tokens.name()] = !decode_special_tokens;
    m_decode_special_tokens = decode_special_tokens;
}

std::optional<ov::genai::StreamingStatus> OVMSTextStreamer::handle_decoding_params_change(int64_t token) {
    if (m_output_parser && !m_decode_special_tokens) {
        const std::string startTag = m_output_parser->getPhaseStartTagForToken(token, m_tools_available);
        if (!startTag.empty()) {
            // Flush pending text with the current mode, then immediately flush the start
            // tag so the phase switches before the next token's mode check fires.
            if (!m_tokens_cache.empty()) {
                const std::string pending = m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
                if (pending.size() > m_printed_len) {
                    const auto s = flush_chunk(pending, pending.size(), ov::genai::GenerationFinishReason::NONE);
                    if (s != ov::genai::StreamingStatus::RUNNING)
                        return s;
                }
            }
            m_tokens_cache.clear();
            m_decoded_lengths.clear();
            m_printed_len = 0;
            m_tokens_cache.push_back(token);
            m_decoded_lengths.push_back(static_cast<int64_t>(startTag.size()));
            const auto s = flush_chunk(startTag, startTag.size(), ov::genai::GenerationFinishReason::NONE);
            m_tokens_cache.clear();
            m_decoded_lengths.clear();
            m_printed_len = 0;
            apply_decode_params(m_output_parser->needSpecialTokensForCurrentDecode(m_user_wants_special));
            if (s != ov::genai::StreamingStatus::RUNNING)
                return s;
            return ov::genai::StreamingStatus::RUNNING;
        }
    }

    if (m_output_parser) {
        const bool decode_with_special_tokens = m_output_parser->needSpecialTokensForCurrentDecode(m_user_wants_special);
        if (decode_with_special_tokens != m_decode_special_tokens) {
            if (!m_tokens_cache.empty()) {
                const std::string text = m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
                if (text.size() > m_printed_len) {
                    const auto s = flush_chunk(text, text.size(), ov::genai::GenerationFinishReason::NONE);
                    if (s != ov::genai::StreamingStatus::RUNNING)
                        return s;
                }
            }
            m_tokens_cache.clear();
            m_decoded_lengths.clear();
            m_printed_len = 0;
            apply_decode_params(decode_with_special_tokens);
        }
    }

    return std::nullopt;
}

ov::genai::StreamingStatus OVMSTextStreamer::write(int64_t token) {
    if (llm_calculator_logger->should_log(spdlog::level::trace))
        m_all_tokens.push_back(token);

    return write(token, /*immediate_flush=*/false);
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

ov::genai::StreamingStatus OVMSTextStreamer::write(int64_t token, bool immediate_flush) {
    if (const auto status = handle_decoding_params_change(token))
        return *status;

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

    if (immediate_flush) {
        // Flush this token's contribution immediately (no delay).
        if (text.size() > m_printed_len) {
            return flush_chunk(text, text.size(), ov::genai::GenerationFinishReason::NONE);
        }
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

void OVMSTextStreamer::end() {
    if (llm_calculator_logger->should_log(spdlog::level::trace) && !m_all_tokens.empty()) {
        const ov::AnyMap no_skip_params{{ov::genai::skip_special_tokens.name(), false}};
        const std::string full_decode = m_tokenizer.decode(m_all_tokens, no_skip_params);
        std::string token_ids;
        token_ids.reserve(m_all_tokens.size() * 7);
        for (size_t i = 0; i < m_all_tokens.size(); ++i) {
            if (i > 0)
                token_ids += ", ";
            token_ids += std::to_string(m_all_tokens[i]);
        }
        SPDLOG_LOGGER_TRACE(llm_calculator_logger,
            "OVMSTextStreamer: {} tokens generated; full decode (skip_special=false): \"{}\"; ids: [{}]",
            m_all_tokens.size(), full_decode, token_ids);
    }

    // Find the first token in m_tokens_cache that has not yet been printed.
    // Tokens with decoded position <= m_printed_len were already flushed by write();
    // they must stay in the cache as BPE decode context for the tokens we drain below.
    // A decoded length of -1 marks an incomplete UTF-8 sequence (also unprinted).
    size_t first_unprinted_idx = 0;
    while (first_unprinted_idx < m_decoded_lengths.size()) {
        const int64_t dlen = m_decoded_lengths[first_unprinted_idx];
        if (dlen > 0 && static_cast<size_t>(dlen) <= m_printed_len) {
            first_unprinted_idx++;
        } else {
            break;
        }
    }

    // Extract the unprinted tokens; leave the printed ones in place as decode context.
    const std::vector<int64_t> unprinted(
        m_tokens_cache.begin() + static_cast<std::ptrdiff_t>(first_unprinted_idx),
        m_tokens_cache.end());
    m_tokens_cache.resize(first_unprinted_idx);
    m_decoded_lengths.resize(first_unprinted_idx);
    // m_printed_len is intentionally kept as-is: it points to the end of the
    // already-printed portion of the current cache so drainToken's flush starts
    // at the right offset.

    for (const int64_t token : unprinted) {
        const auto status = write(token, /*immediate_flush=*/true);
        if (status != ov::genai::StreamingStatus::RUNNING) {
            break;  // cancelled mid-drain; still deliver the STOP signal below
        }
    }

    // Always deliver the STOP signal so parsers that rely on finishReason==STOP
    // for cleanup receive it (e.g. hasPendingState flush in Lfm2ToolParser,
    // argument string finalisation in Hermes3ToolParser).
    const std::string final_text = m_tokens_cache.empty()
                                       ? std::string{}
                                       : m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
    flush_chunk(final_text, m_printed_len, ov::genai::GenerationFinishReason::STOP);

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
// the decoded text chunk.
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

    std::optional<Delta> delta;
    if (m_output_parser != nullptr) {
        delta = m_output_parser->parseChunk(chunk, tokens, m_tools_available, finish_reason);
    } else if (!chunk.empty()) {
        // No parser: pass raw text as a ContentDelta.
        delta = ContentDelta{chunk};
    }

    const bool isLast = (finish_reason != ov::genai::GenerationFinishReason::NONE);
    if (delta.has_value()) {
        return m_callback(std::move(*delta), isLast);
    }
    if (isLast) {
        // Parser produced no delta for the final flush (e.g. generation ended on a
        // special token the parser absorbed). Fire the callback with FinishDelta so
        // the caller can emit the finish_reason chunk.
        return m_callback(FinishDelta{}, true);
    }
    return ov::genai::StreamingStatus::RUNNING;
}

}  // namespace ovms
