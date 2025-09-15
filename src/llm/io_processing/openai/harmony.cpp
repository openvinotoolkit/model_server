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
#include "../../../stringutils.hpp"
#include "../utils.hpp"

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

std::string Harmony::getContent() {
    std::string content;
    int i = 0;
    for (const auto& msg : messages) {
        if (msg.getChannel() == "final") {
            if (i++ > 0) {
                content += " ";
            }
            content += msg.getContent();
        }

        //  Preambles
        /*
            Preambles
            At times the model might choose to generate a “preamble” to inform the user about the tools it is about to call.
            For example, when it plans to call multiple tools.
            If this is the case it will generate an assistant message on the commentary channel that, unlike the chain-of-thought, is intended TO BE SHOWN to the end-user.
        */
        if (msg.getChannel() == "commentary") {
            if (i++ > 0) {
                content += " ";
            }
            content += msg.getContent();
        }
    }
    return content;
}

std::string Harmony::getReasoning() {
    std::string reasoning;
    int i = 0;
    for (const auto& msg : messages) {
        if (msg.getChannel() == "analysis") {  // TODO: Reasoning might be also in commentary part // or is it content? https://cookbook.openai.com/articles/openai-harmony#preambles
            if (i++ > 0) {
                content += " ";
            }
            reasoning += msg.getContent();
        }
    }
    return reasoning;
}

ToolCalls Harmony::getToolCalls() {
    ToolCalls tool_calls;
    // Message in channel [commentary to=functions.get_humidity ] constrain[json]: [{"location":"Paris"}]
    for (const auto& msg : messages) {
        //SPDLOG_INFO("Channel: [{}] Constrain:[{}] Content:[{}]", msg.getChannel(), msg.getConstrain(), msg.getContent());
        if (startsWith(msg.getChannel(), "commentary")) {
            //SPDLOG_INFO("Found commentary");
            // Try to parse tool name from segment like 'to=functions.NAME ...'
            const static std::string tool_prefix = "to=functions.";
            size_t marker = msg.getChannel().find(tool_prefix);
            if (marker != std::string::npos) {
                marker += tool_prefix.length();
                size_t sp = msg.getChannel().find_first_of(" \t\n\r<", marker);
                ToolCall tc;
                if (sp == std::string::npos) {
                    tc.name = msg.getChannel().substr(marker); // take the rest of the string
                } else {
                    tc.name = msg.getChannel().substr(marker, sp - marker);
                }
                tc.arguments = msg.getContent();
                tc.id = generateRandomId();
                tool_calls.push_back(tc);
            } else {
                // SPDLOG_INFO("Could not find tool name in channel [{}]", msg.getChannel());
            }
        }
    }
    return tool_calls;
}

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
                SPDLOG_DEBUG("Error - found <|constain|> outside of channel reading state");
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
                    //SPDLOG_INFO("Message in channel [{}] constrain[{}]: [{}]", cur_channel, cur_constrain, decoded);
                    messages.emplace_back(cur_channel, cur_constrain, decoded);
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
                    //SPDLOG_INFO("Message in channel [{}] constrain[{}]: [{}]", cur_channel, cur_constrain, decoded);
                    messages.emplace_back(cur_channel, cur_constrain, decoded);
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
                    //SPDLOG_INFO("Message in channel [{}] constrain[{}]: [{}]", cur_channel, cur_constrain, decoded);
                    messages.emplace_back(cur_channel, cur_constrain, decoded);
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
const std::string Harmony::TOKEN_START = "<|start|>";
const std::string Harmony::TOKEN_END = "<|end|>";  // valid stop token (sometimes)
const std::string Harmony::TOKEN_MESSAGE = "<|message|>";
const std::string Harmony::TOKEN_CHANNEL = "<|channel|>";
const std::string Harmony::TOKEN_CONSTAIN = "<|constain|>";
const std::string Harmony::TOKEN_RETURN = "<|return|>";  // valid stop token
const std::string Harmony::TOKEN_CALL = "<|call|>";  // valid stop token

}  // namespace openai
}  // namespace ovms
