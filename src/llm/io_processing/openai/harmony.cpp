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
#include "harmony.hpp"

#include "../../../logging.hpp"

// Based on https://cookbook.openai.com/articles/openai-harmony

namespace ovms {
namespace openai {

Harmony::Harmony(ov::genai::Tokenizer& tokenizer, const std::vector<int64_t>& tokens) : tokenizer(tokenizer), tokens(tokens) {}

enum class HarmonyState : int {
    UNKNOWN,
    READING_CHANNEL,
    READING_MESSAGE,
    READING_CONSTRAIN,
};

bool Harmony::parse() {
    if (tokens.empty())
        return true;

    // Parse the token stream
    size_t pos = 0;

    HarmonyState cur_state = HarmonyState::UNKNOWN;
    std::string cur_channel;
    std::string cur_constrain;

    std::vector<int64_t> token_cache;

    while (pos < tokens.size()) {
        //SPDLOG_INFO("POS: {}", pos);
        int64_t token = tokens[pos];

        if (token == TokenID::CHANNEL) {
            cur_state = HarmonyState::READING_CHANNEL;
            cur_channel.clear();
            cur_constrain.clear();
            token_cache.clear();
            pos++;
            continue;
        }

        else if (token == TokenID::CONSTRAIN) {
            if (cur_state != HarmonyState::READING_CHANNEL) {
                SPDLOG_INFO("Error - found <|constain|> outside of channel reading state");
                return false;
            }

            // Dump Channel
            if (!token_cache.empty()) {
                std::string decoded = tokenizer.decode(token_cache, ov::AnyMap{ov::genai::skip_special_tokens(false)});
                if (cur_state == HarmonyState::READING_CHANNEL) {
                    cur_channel = decoded;
                    cur_state = HarmonyState::READING_CONSTRAIN;
                }
                token_cache.clear();
            }
            pos++;
            continue;
        }

        else if (token == TokenID::MESSAGE) {
            // Dump Channel
            if (!token_cache.empty()) {
                std::string decoded = tokenizer.decode(token_cache, ov::AnyMap{ov::genai::skip_special_tokens(false)});
                if (cur_state == HarmonyState::READING_CHANNEL) {
                    cur_channel = decoded;
                    cur_state = HarmonyState::READING_MESSAGE;
                }
                if (cur_state == HarmonyState::READING_CONSTRAIN) {
                    cur_constrain = decoded;
                    cur_state = HarmonyState::READING_MESSAGE;
                }
                token_cache.clear();
            }
            pos++;
            continue;
        }
        else if (token == TokenID::END) {
            if (!token_cache.empty()) {
                std::string decoded = tokenizer.decode(token_cache, ov::AnyMap{ov::genai::skip_special_tokens(false)});
                if (cur_state == HarmonyState::READING_MESSAGE) {
                    SPDLOG_INFO("Message in channel [{}] constrain[{}]: [{}]", cur_channel, cur_constrain, decoded);
                }
                cur_state = HarmonyState::UNKNOWN;
                token_cache.clear();
            }
            pos++;
            continue;
        }
        else if (token == TokenID::RETURN) {
            if (!token_cache.empty()) {
                std::string decoded = tokenizer.decode(token_cache, ov::AnyMap{ov::genai::skip_special_tokens(false)});
                if (cur_state == HarmonyState::READING_MESSAGE) {
                    SPDLOG_INFO("Message in channel [{}] constrain[{}]: [{}]", cur_channel, cur_constrain, decoded);
                }
                cur_state = HarmonyState::UNKNOWN;
                token_cache.clear();
            }
            pos++;
            continue;
        }
        else if (token == TokenID::CALL) {
            if (!token_cache.empty()) {
                std::string decoded = tokenizer.decode(token_cache, ov::AnyMap{ov::genai::skip_special_tokens(false)});
                if (cur_state == HarmonyState::READING_MESSAGE) {
                    SPDLOG_INFO("Message in channel [{}] constrain[{}]: [{}]", cur_channel, cur_constrain, decoded);
                }
                cur_state = HarmonyState::UNKNOWN;
                token_cache.clear();
            }
            pos++;
            continue;
        }
        token_cache.push_back(token);
        pos++;
    }

    return true;
}

bool Harmony::parse2() {
    if (tokens.empty()) {
        return false;
    }

    // Decode whole token stream including special tokens
    std::string decoded = tokenizer.decode(tokens, ov::AnyMap{ov::genai::skip_special_tokens(false)});

    // Find start token
    size_t pos = decoded.find(TOKEN_START);
    if (pos == std::string::npos) {
        // no structured harmony content found - treat whole decoded as content
        content = decoded;
        return true;
    }

    // Move after start
    pos += TOKEN_START.size();

    // Expect sequences of markers: <|message|>, <|channel|> etc. We'll scan and extract segments
    while (pos < decoded.size()) {
        size_t nextMsg = decoded.find(TOKEN_MESSAGE, pos);
        if (nextMsg == std::string::npos) break;
        nextMsg += TOKEN_MESSAGE.size();

        // optional channel
        size_t chPos = decoded.find(TOKEN_CHANNEL, nextMsg);
        std::string channelVal;
        if (chPos == nextMsg) {
            // channel immediately after message marker
            size_t chStart = chPos + TOKEN_CHANNEL.size();
            // channel value is until next marker (<|message|> or <|call|> or <|return|> or <|end|>)
            size_t chEnd = decoded.find(TOKEN_MESSAGE, chStart);
            size_t callPos = decoded.find(TOKEN_CALL, chStart);
            size_t returnPos = decoded.find(TOKEN_RETURN, chStart);
            size_t endPos = decoded.find(TOKEN_END, chStart);
            size_t candidate = std::string::npos;
            for (size_t p : {chEnd, callPos, returnPos, endPos}) {
                if (p != std::string::npos) {
                    if (candidate == std::string::npos || p < candidate) candidate = p;
                }
            }
            if (candidate == std::string::npos) candidate = decoded.size();
            channelVal = decoded.substr(chStart, candidate - chStart);
            nextMsg = candidate;
        }

        // Look for call or return or end
        size_t callPos = decoded.find(TOKEN_CALL, nextMsg);
        size_t returnPos = decoded.find(TOKEN_RETURN, nextMsg);
        size_t endPos = decoded.find(TOKEN_END, nextMsg);

        size_t segmentEnd = std::string::npos;
        std::string segment;
        if (callPos != std::string::npos && (callPos < returnPos || returnPos == std::string::npos) && (callPos < endPos || endPos == std::string::npos)) {
            segment = decoded.substr(nextMsg, callPos - nextMsg);
            segmentEnd = callPos + TOKEN_CALL.size();
            // treat as tool call content
            ToolCall tc;
            // Try to parse tool name from segment like 'to=functions.NAME ...'
            size_t marker = segment.find("to=functions.");
            if (marker != std::string::npos) {
                marker += strlen("to=functions.");
                size_t sp = segment.find_first_of(" \t\n\r<", marker);
                tc.name = segment.substr(marker, sp - marker);
            }
            tc.arguments = segment;
            toolCalls.push_back(tc);
        } else if (returnPos != std::string::npos && (returnPos < callPos || callPos == std::string::npos) && (returnPos < endPos || endPos == std::string::npos)) {
            segment = decoded.substr(nextMsg, returnPos - nextMsg);
            segmentEnd = returnPos + TOKEN_RETURN.size();
            // treat as normal content / assistant reply
            content += segment;
        } else if (endPos != std::string::npos) {
            segment = decoded.substr(nextMsg, endPos - nextMsg);
            content += segment;
            break;
        } else {
            // no markers left
            segment = decoded.substr(nextMsg);
            content += segment;
            break;
        }

        pos = segmentEnd;
    }

    // Basic heuristic: if content contains 'analysis' channel, set reasoning
    if (!content.empty()) {
        // If channel indicated analysis earlier, assume content is reasoning
        if (content.find("analysis") != std::string::npos) {
            reasoning = content;
            content.clear();
        }
    }

    return true;
}

const std::string Harmony::TOKEN_START = "<|start|>";
const std::string Harmony::TOKEN_END = "<|end|>";  // valid stop token (sometimes)
const std::string Harmony::TOKEN_MESSAGE = "<|message|>";
const std::string Harmony::TOKEN_CHANNEL = "<|channel|>";
const std::string Harmony::TOKEN_CONSTAIN = "<|constain|>";
const std::string Harmony::TOKEN_RETURN = "<|return|>";  // valid stop token
const std::string Harmony::TOKEN_CALL = "<|call|>";  // valid stop token

}  // namespace openai
}  // namespace ovms
