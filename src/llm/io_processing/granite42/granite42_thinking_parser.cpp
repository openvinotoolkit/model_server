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

#include <algorithm>
#include <string>
#include <utility>
#include <variant>

#include "granite42_thinking_parser.hpp"

namespace ovms {

std::optional<Delta> Granite42ThinkingParser::parseChunk(
    const std::string& chunk,
    const std::vector<int64_t>& tokens,
    ov::genai::GenerationFinishReason /*finishReason*/) {
    if (chunk.empty()) {
        return std::nullopt;
    }

    // The Granite template can place <think> in the prompt. Its generated
    // continuation then starts inside reasoning; a fused marker without a
    // closing tag remains literal reasoning text, while a same-delta pair is
    // handled by the token-aware boundary logic below.
    if (implicitStart) {
        phaseEntryTagConsumed_ = true;
    }

    const std::string& startTag = parsingConfig.startTags.front();
    const std::string& endTag = parsingConfig.endTag;
    const size_t startTagPos = chunk.find(startTag);
    const size_t endTagPos = chunk.find(endTag);
    const bool startTokenInCurrentDelta = std::any_of(tokens.begin(), tokens.end(),
        [&](int64_t token) { return getResolvedStartTokenToTag().find(token) != getResolvedStartTokenToTag().end(); });
    // Parser unit tests may call parseChunk without token IDs. The textual
    // marker remains a useful fallback for that mode; production
    // OVMSTextStreamer supplies IDs for Granite's single-token marker.
    const bool startInCurrentDelta = startTokenInCurrentDelta || startTagPos != std::string::npos;
    const bool firstGeneratedStartTag = startInCurrentDelta && !generatedStartTagSeen_;
    if (firstGeneratedStartTag) {
        generatedStartTagSeen_ = true;
        generatedStartTagEventReasoningOrdinal_ = emittedReasoningDeltaCount_;
        generatedStartTagConsumed_ = false;
    }

    // The upstream streaming parser consumes a standalone start-token delta as
    // structural control text even when it appears after markerless reasoning.
    // Require the resolved single-token ID when the streamer supplies IDs;
    // empty IDs are accepted for direct parser fixtures. A marker fused with
    // surrounding text or represented by multiple tokens remains literal below.
    const bool standaloneStartToken =
        (tokens.size() == 1 &&
            getResolvedStartTokenToTag().find(tokens.front()) != getResolvedStartTokenToTag().end()) ||
        (tokens.empty() && chunk == startTag);
    if (standaloneStartToken) {
        phaseEntryTagConsumed_ = true;
        generatedStartTagSeen_ = true;
        standaloneStartTagEventReasoningOrdinals_.push_back(emittedReasoningDeltaCount_);
        if (!generatedStartTagEventReasoningOrdinal_.has_value()) {
            generatedStartTagEventReasoningOrdinal_ = emittedReasoningDeltaCount_;
        }
        if (firstGeneratedStartTag) {
            generatedStartTagConsumed_ = true;
            firstGeneratedStartTagWasStandalone_ = true;
        }
        return std::nullopt;
    }

    // Without an implicit prompt marker, vLLM's start-in-current streaming
    // branch strips the start tag only when the same delta also contains the
    // closing tag. A fused marker-plus-body delta remains literal reasoning.
    std::string reasoning = chunk;
    if (!phaseEntryTagConsumed_) {
        phaseEntryTagConsumed_ = true;
    }
    // BaseThinkingReasoningParser uses the start/end IDs from the current
    // delta when no earlier generated start marker is present, including after
    // an implicit prompt start. Once a generated marker has been observed,
    // later markers remain literal reasoning text. A fused first marker without
    // an end remains literal until unary finalization normalizes it.
    if (firstGeneratedStartTag && startInCurrentDelta && startTagPos != std::string::npos &&
        endTagPos != std::string::npos && endTagPos >= startTagPos + startTag.size()) {
        reasoning = chunk.substr(startTagPos + startTag.size(),
            endTagPos - (startTagPos + startTag.size()));
        if (firstGeneratedStartTag)
            generatedStartTagConsumed_ = true;
    }

    // OutputParser preserves bytes following the first end tag for the next
    // phase. Use find rather than rfind to match Python str.partition().
    const size_t reasoningEndTagPos = reasoning.find(endTag);
    if (reasoningEndTagPos != std::string::npos) {
        reasoning.erase(reasoningEndTagPos);
    }

    if (reasoning.empty()) {
        return std::nullopt;
    }
    ++emittedReasoningDeltaCount_;
    return ReasoningDelta{std::move(reasoning)};
}

void Granite42ThinkingParser::finalizeUnaryDeltas(
    std::vector<Delta>& deltas,
    bool finalContentWasPresent) const {
    // IBM's unary parent partitions on the first generated <think> marker.
    // Streaming intentionally keeps a fused marker, so normalize the collected
    // reasoning deltas here, after generation has ended. A standalone marker or
    // a same-delta start/end pair leaves no marker text; its ordinal identifies
    // which earlier reasoning deltas belong to the discarded prefix.
    if (generatedStartTagSeen_ || generatedStartTagEventReasoningOrdinal_.has_value()) {
        const std::string& startTag = parsingConfig.startTags.front();
        const size_t markerEventOrdinal = generatedStartTagEventReasoningOrdinal_.value_or(0);
        if (generatedStartTagConsumed_) {
            // The first marker was consumed structurally, so its text cannot
            // be found in the deltas. Clear only reasoning emitted before its
            // event; later literal markers must remain untouched.
            size_t ordinal = 0;
            for (Delta& delta : deltas) {
                if (auto* reasoning = std::get_if<ReasoningDelta>(&delta)) {
                    if (ordinal < markerEventOrdinal)
                        reasoning->text.clear();
                    ++ordinal;
                }
            }
        } else {
            size_t reasoningOrdinal = 0;
            bool normalized = false;
            for (size_t i = 0; i < deltas.size(); ++i) {
                auto* reasoning = std::get_if<ReasoningDelta>(&deltas[i]);
                if (reasoning == nullptr)
                    continue;

                if (!normalized) {
                    const size_t markerPos = reasoning->text.find(startTag);
                    if (markerPos != std::string::npos) {
                        // Python's str.partition() drops every generated byte
                        // before the first <think>, including the prefix in the
                        // same delta, then removes that first marker. Later
                        // literal markers remain part of the reasoning text.
                        for (size_t previous = 0; previous < i; ++previous) {
                            if (auto* priorReasoning = std::get_if<ReasoningDelta>(&deltas[previous]))
                                priorReasoning->text.clear();
                        }
                        reasoning->text.erase(0, markerPos + startTag.size());
                        normalized = true;
                    } else if (reasoningOrdinal >= markerEventOrdinal) {
                        // A marker may have been swallowed separately from all
                        // reasoning text. Preserve this later delta while
                        // clearing the pre-marker prefix.
                        for (size_t previous = 0; previous < i; ++previous) {
                            if (auto* priorReasoning = std::get_if<ReasoningDelta>(&deltas[previous]))
                                priorReasoning->text.clear();
                        }
                        normalized = true;
                    }
                }
                ++reasoningOrdinal;
            }
            if (!normalized && generatedStartTagEventReasoningOrdinal_.has_value()) {
                // The marker may have been the final generated token, so no
                // post-marker reasoning delta exists to trigger the ordinal
                // branch above. Clear the prefix eagerly in that case.
                size_t ordinal = 0;
                for (Delta& delta : deltas) {
                    if (auto* reasoning = std::get_if<ReasoningDelta>(&delta)) {
                        if (ordinal < markerEventOrdinal)
                            reasoning->text.clear();
                        ++ordinal;
                    }
                }
            }
        }

        // vLLM's streaming parser suppresses every standalone start token,
        // but Granite's unary parent removes only the first generated marker
        // with str.partition(). Restore later standalone markers at the
        // reasoning-delta boundary where each token was observed.
        size_t firstStandaloneToRestore = firstGeneratedStartTagWasStandalone_ ? 1 : 0;
        if (firstStandaloneToRestore < standaloneStartTagEventReasoningOrdinals_.size()) {
            size_t event = firstStandaloneToRestore;
            while (event < standaloneStartTagEventReasoningOrdinals_.size()) {
                const size_t ordinal = standaloneStartTagEventReasoningOrdinals_[event];
                size_t count = 0;
                while (event + count < standaloneStartTagEventReasoningOrdinals_.size() &&
                       standaloneStartTagEventReasoningOrdinals_[event + count] == ordinal) {
                    ++count;
                }
                std::string markers;
                markers.reserve(count * startTag.size());
                for (size_t i = 0; i < count; ++i)
                    markers += startTag;

                size_t reasoningOrdinal = 0;
                bool inserted = false;
                for (Delta& delta : deltas) {
                    if (auto* reasoning = std::get_if<ReasoningDelta>(&delta)) {
                        if (reasoningOrdinal == ordinal) {
                            reasoning->text.insert(0, markers);
                            inserted = true;
                            break;
                        }
                        ++reasoningOrdinal;
                    }
                }
                if (!inserted) {
                    for (auto it = deltas.rbegin(); it != deltas.rend(); ++it) {
                        if (auto* reasoning = std::get_if<ReasoningDelta>(&*it)) {
                            reasoning->text += markers;
                            inserted = true;
                            break;
                        }
                    }
                }
                if (!inserted) {
                    const auto finishIt = std::find_if(deltas.begin(), deltas.end(),
                        [](const Delta& delta) { return std::holds_alternative<FinishDelta>(delta); });
                    deltas.insert(finishIt, ReasoningDelta{std::move(markers)});
                }
                event += count;
            }
        }
    }

    if (!promoteReasoningToContent_ || finalContentWasPresent) {
        return;
    }

    // IBM promotes only when Python would have returned final_content=None.
    // The caller tracked raw post-reasoning bytes before a content or tool parser
    // could consume them, preserving the Python None-versus-empty distinction.
    for (Delta& delta : deltas) {
        if (auto* reasoning = std::get_if<ReasoningDelta>(&delta)) {
            std::string text = std::move(reasoning->text);
            delta = ContentDelta{std::move(text)};
        }
    }
}

}  // namespace ovms
