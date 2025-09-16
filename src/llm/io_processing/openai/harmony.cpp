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

/*
    There are 2 ways model can put content:
    - Messages in channel "final" - these are final content messages; example: <|channel|>final<|message|>The weather is sunny.<|end|>
    - Messages in channel "commentary" - these are preambles that model can use to inform user about tools it is going to call; example: <|channel|>commentary<|message|>I will call the get_weather function to get the current weather.<|end|>

    Both types of messages are concatenated to form the final content.
*/
std::string Harmony::getContent() {
    std::string content;
    int i = 0;
    for (const auto& msg : messages) {
        // Regular content. Usually appears at the end of generation.
        if (msg.getChannel() == "final") {
            if (i++ > 0) {
                content += " ";  // TODO: how to separate multiple preambles/content? For now this is space
            }
            content += msg.getContent();
        }

        /*
            Preambles
            At times the model might choose to generate a “preamble” to inform the user about the tools it is about to call.
            For example, when it plans to call multiple tools.
            If this is the case it will generate an assistant message on the commentary channel that, unlike the chain-of-thought, is intended TO BE SHOWN to the end-user.
        */
        if (msg.getChannel() == "commentary") {
            if (i++ > 0) {
                content += " ";  // TODO: how to separate multiple preambles/content? For now this is space
            }
            content += msg.getContent();
        }
    }
    return content;
}

/*
    Reasoning is extracted from messages in channel "analysis"; example: <|channel|>analysis<|message|>I need to find out the weather in Paris.<|end|>
*/
std::string Harmony::getReasoning() {
    std::string reasoning;
    int i = 0;
    for (const auto& msg : messages) {
        if (msg.getChannel() == "analysis") {
            if (i++ > 0) {
                reasoning += " ";
            }
            reasoning += msg.getContent();
        }
    }
    return reasoning;
}

/*
    Tool calls are extracted from messages in channel "commentary" that contain "to=functions.NAME" in the channel content; example:
    <|channel|>commentary to=functions.get_humidity <|message|>{"location":"Paris"}<|end|>
*/
ToolCalls Harmony::getToolCalls() {
    ToolCalls toolCalls;
    for (const auto& msg : messages) {
        if (startsWith(msg.getChannel(), "commentary")) {
            const static std::string tool_prefix = "to=functions.";
            size_t marker = msg.getChannel().find(tool_prefix);
            if (marker != std::string::npos) {
                marker += tool_prefix.length();
                size_t firstWhiteSpaceOrSpecialBegin = msg.getChannel().find_first_of(" \t\n\r<", marker);
                ToolCall toolCall;
                if (firstWhiteSpaceOrSpecialBegin == std::string::npos) {
                    // Take the remaining part of the string
                    toolCall.name = msg.getChannel().substr(marker);
                } else {
                    // Take up to the first whitespace or special token begin
                    toolCall.name = msg.getChannel().substr(marker, firstWhiteSpaceOrSpecialBegin - marker);
                }
                toolCall.arguments = msg.getContent();
                toolCall.id = generateRandomId();
                toolCalls.push_back(toolCall);
            } else {
                SPDLOG_DEBUG("Skipping tool call. Could not find tool name in channel [{}]", msg.getChannel());
            }
        }
    }
    return toolCalls;
}

bool Harmony::parse() {

    // Simply display all text with special tokens
    SPDLOG_INFO("AAAAA: [{}]", tokenizer.decode(tokens, ov::AnyMap{ov::genai::skip_special_tokens(false)}));

    if (tokens.empty())
        return true;

    size_t pos = 0;

    HarmonyState cur_state = HarmonyState::UNKNOWN;
    std::string cur_channel;
    std::string cur_constrain;

    std::vector<int64_t> token_cache;

    while (pos < tokens.size()) {
        int64_t token = tokens[pos];
 
        // New channel starts
        if (token == TokenID::CHANNEL) {
            cur_state = HarmonyState::READING_CHANNEL;
            cur_channel.clear();
            cur_constrain.clear();
            token_cache.clear();
            pos++;
            continue;
        }

        // Constrain starts. Can appear only inside channel definition
        else if (token == TokenID::CONSTRAIN) {
            if (cur_state != HarmonyState::READING_CHANNEL) {
                SPDLOG_DEBUG("Error parsing harmony format - found <|constrain|> outside of channel reading state");
                return false;
            }

            // Reading channel finished, save channel title
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

        // Message starts. Can appear only after channel definition and optionally after constrain
        else if (token == TokenID::MESSAGE) {
            if (!token_cache.empty()) {
                std::string decoded = tokenizer.decode(token_cache, ov::AnyMap{ov::genai::skip_special_tokens(false)});
                
                // Depending on current state, save channel or constrain and clear the cache - prepare clean state for reading message
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

        // Finished reading entire message. Message can be regular content, reasoning or tool call
        // Channel name, metadata, constrains should be already read at this point
        else if (token == TokenID::END || token == TokenID::RETURN || token == TokenID::CALL) {
            if (!token_cache.empty()) {
                std::string decoded = tokenizer.decode(token_cache, ov::AnyMap{ov::genai::skip_special_tokens(false)});
                
                // Message reading is complete. Pushing its content
                if (cur_state == HarmonyState::READING_MESSAGE) {
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
