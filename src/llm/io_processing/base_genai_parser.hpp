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

#include <openvino/genai/parsers.hpp>

namespace ovms {

/**
 * @brief Extension of ov::genai::Parser that exposes OVMS OutputParser metadata.
 *
 * Concrete implementations should override the tag getters to tell
 * GenAIParserAdapter (and therefore the surrounding OutputParser machinery)
 * where parsing segments begin and end in the generated text, and whether
 * the text streamer must preserve special tokens.
 *
 * Defaults:
 *   - getParsingStartTags()        pure virtual — must be overridden
 *   - getSpecialParsingStartTags() returns empty vector
 *   - getParsingEndTag()           pure virtual — must be overridden
 *   - requiresStreamingWithSpecialTokens() returns false
 */
class BaseGenAIParser : public ov::genai::Parser {
public:
    BaseGenAIParser() = default;
    ~BaseGenAIParser() override = default;

    /**
     * @brief Tags that trigger the start of a parsed segment.
     *
     * Used by OutputParser to transition into the phase handled by this parser.
     * Return an empty vector if the parser handles all output unconditionally.
     */
    virtual const std::vector<std::string>& getParsingStartTags() const = 0;

    /**
     * @brief Tags that trigger parsing only when they appear at the very start of output.
     *
     * These are considered exclusively during the UNKNOWN phase of OutputParser.
     */
    virtual const std::vector<std::string>& getSpecialParsingStartTags() const {
        static const std::vector<std::string> empty{};
        return empty;
    }

    /**
     * @brief Tag that marks the end of the segment processed by this parser.
     *
     * Return an empty string if the parser consumes until end-of-generation.
     */
    virtual const std::string& getParsingEndTag() const = 0;

    /**
     * @brief Whether the text streamer must include special tokens.
     *
     * If true, the tokenizer used in the TextStreamer should be configured to
     * not skip special tokens so that this parser can use them as boundaries.
     */
    virtual bool requiresStreamingWithSpecialTokens() const {
        return false;
    }
};

/**
 * @brief Extension of ov::genai::IncrementalParser that exposes OVMS OutputParser metadata.
 *
 * Mirrors BaseGenAIParser but for the streaming (incremental) variant.
 * The same defaults apply.
 */
class BaseGenAIIncrementalParser : public ov::genai::IncrementalParser {
public:
    BaseGenAIIncrementalParser() = default;
    ~BaseGenAIIncrementalParser() override = default;

    /**
     * @brief Tags that trigger the start of a parsed segment.
     */
    virtual const std::vector<std::string>& getParsingStartTags() const = 0;

    /**
     * @brief Tags that trigger parsing only when they appear at the very start of output.
     */
    virtual const std::vector<std::string>& getSpecialParsingStartTags() const {
        static const std::vector<std::string> empty{};
        return empty;
    }

    /**
     * @brief Tag that marks the end of the segment processed by this parser.
     */
    virtual const std::string& getParsingEndTag() const = 0;

    /**
     * @brief Whether the text streamer must include special tokens.
     */
    virtual bool requiresStreamingWithSpecialTokens() const {
        return false;
    }

    /**
     * @brief Flush any internally buffered state after the final chunk.
     *
     * Called by GenAIParserAdapter::parseChunk when finishReason != NONE,
     * immediately after the last parse() call. Implementations that use a
     * delay window (or any other look-ahead buffering) must drain the pending
     * content here and populate @p delta_message with a final delta if one is
     * available.
     *
     * The default implementation is a no-op.
     *
     * @param delta_message  Populated with a structured delta when pending
     *                       content exists; left untouched otherwise.
     */
    virtual void flush(ov::genai::JsonContainer& /*delta_message*/) {}
};

}  // namespace ovms
