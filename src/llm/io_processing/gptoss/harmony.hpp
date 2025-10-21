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
#pragma once

#include <string>
#include <vector>
#include <openvino/genai/tokenizer.hpp>

#include "../base_output_parser.hpp"

namespace ovms {
namespace openai {

class Message {
    std::string channel;
    std::string constrain;
    std::string content;

public:
    Message(const std::string& channel, const std::string& constrain, const std::string& content) :
        channel(channel),
        constrain(constrain),
        content(content) {}
    std::string getChannel() const { return channel; }
    std::string getConstrain() const { return constrain; }
    std::string getContent() const { return content; }
};

class Harmony {
    ov::genai::Tokenizer& tokenizer;
    std::vector<int64_t> tokens;

    // Valid messages parsed from unary output, used in final accessors
    std::vector<Message> messages;

    // Intermediate state during parsing of each message, not used in final accessors
    std::string content;
    std::string reasoning;
    ToolCalls_t toolCalls;

public:
    Harmony(ov::genai::Tokenizer& tokenizer, const std::vector<int64_t>& tokens);

    bool parse();

    std::string getContent();
    std::string getReasoning();
    ToolCalls_t getToolCalls();

    static const std::string TOKEN_START;
    static const std::string TOKEN_END;
    static const std::string TOKEN_MESSAGE;
    static const std::string TOKEN_CHANNEL;
    static const std::string TOKEN_CONSTRAIN;
    static const std::string TOKEN_RETURN;
    static const std::string TOKEN_CALL;

    enum class TokenID : int64_t {
        START = 200006,
        END = 200007,
        MESSAGE = 200008,
        CHANNEL = 200005,
        CONSTRAIN = 200003,
        RETURN = 200002,
        CALL = 200012
    };
};

inline bool operator==(Harmony::TokenID token, int64_t value) {
    return static_cast<int64_t>(token) == value;
}

inline bool operator==(int64_t value, Harmony::TokenID token) {
    return token == value;  // reuse above
}

}  // namespace openai
}  // namespace ovms
