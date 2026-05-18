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
#include <rapidjson/document.h>

#include "io_processing/output_parser.hpp"

namespace ovms {

// OVMSTextStreamer inherits ov::genai::TextStreamer to reuse its protected
// decode-loop state (m_tokenizer, m_tokens_cache, m_decoded_lengths,
// m_printed_len, m_additional_detokenization_params). It overrides
// write(int64_t) and end() completely — the no-op callback passed at
// construction is never invoked.
//
// On every flush event the streamer:
//   1. Computes the token slice that produced the current text chunk via the
//      same upper_bound logic used by ov::genai::TextParserStreamer.
//   2. Calls OutputParser::parseChunk(chunk, tokens, tools_available, finish_reason).
//   3. If the result is non-nullopt (or this is the final flush), fires the
//      registered Callback with the Document.
//
// The Callback accumulates Documents in pendingDeltas on the execution context.
// preparePartialResponse drains pendingDeltas after each write()/end() cycle.
//
// When output_parser is nullptr (e.g. /v1/completions endpoint), the streamer
// wraps the raw text in a trivial {"delta":{"content":"..."}} Document and
// fires the callback unconditionally, preserving existing behavior.
class OVMSTextStreamer : public ov::genai::TextStreamer {
public:
    // Callback receives a Document and returns the streaming status.
    // Document shape is always {"delta":{...}} matching the OpenAI delta format.
    // For the finish-only case (nullopt from parseChunk + STOP finishReason),
    // an empty Document{} is passed so the caller can emit the finish_reason chunk.
    using Callback = std::function<ov::genai::StreamingStatus(rapidjson::Document)>;

    // outputParser may be nullptr.
    // toolsAvailable must be evaluated after parseRequest() has processed the body.
    // decodeParams controls skip_special_tokens etc. — static for Phase 1.
    OVMSTextStreamer(
        const ov::genai::Tokenizer& tokenizer,
        const std::unique_ptr<OutputParser>& output_parser,
        bool tools_available,
        Callback callback,
        const ov::AnyMap& decode_params);

    ov::genai::StreamingStatus write(int64_t token) override;
    void end() override;

private:
    const std::unique_ptr<OutputParser>& m_output_parser;  // non-owning ref; lifetime is the HTTP request
    bool m_tools_available;
    Callback m_callback;

    // Must match the file-scope constexpr in openvino/genai text_streamer.cpp.
    // Named here so a future GenAI change is a single update point.
    static constexpr size_t DELAY_N_TOKENS = 3;

    // Flush text[m_printed_len : print_until] with the corresponding token slice.
    ov::genai::StreamingStatus flush_chunk(
        const std::string& text,
        size_t print_until,
        ov::genai::GenerationFinishReason finish_reason);
};

}  // namespace ovms
