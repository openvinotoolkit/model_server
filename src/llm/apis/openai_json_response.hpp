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
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include <string>
#include "openai_completions.hpp"

using namespace rapidjson;

namespace ovms {
class OpenAiJsonResponse : public Writer<StringBuffer> {
    StringBuffer buffer;

public:
    OpenAiJsonResponse() :
        Writer<StringBuffer>(buffer) {}

    // Returns the serialized JSON string
    std::string ToString() {
        Writer<StringBuffer>::Flush();
        return buffer.GetString();
    }

    // Generic wrappers for key-value pairs
    // All return dummy boolean true to maintain consistency with the original API
    bool StartObject(const std::string& name = "") {
        if (!name.empty()) {
            Writer<StringBuffer>::String(name.c_str());
        }
        Writer<StringBuffer>::StartObject();
        return true;
    }

    bool StartArray(const std::string& name = "") {
        if (!name.empty()) {
            Writer<StringBuffer>::String(name.c_str());
        }
        Writer<StringBuffer>::StartArray();
        return true;
    }

    bool Null(const std::string& name = "") {
        if (!name.empty()) {
            Writer<StringBuffer>::String(name.c_str());
        }
        Writer<StringBuffer>::Null();
        return true;
    }

    bool String(const std::string& value) {
        Writer<StringBuffer>::String(value.c_str());
        return true;
    }

    bool String(const std::string& name, const std::string& value) {
        String(name);
        String(value);
        return true;
    }

    bool Int(const std::string& name, int value) {
        String(name);
        Writer<StringBuffer>::Int(value);
        return true;
    }

    bool Uint(const std::string& name, size_t value) {
        String(name);
        Writer<StringBuffer>::Uint(value);
        return true;
    }

    /* 
    API specific methods. Following convention:
    - regular name methods (e.g., FinishReason) write a key-value pair
    - methods with Object suffix (e.g., LogprobObject) write a JSON object
    - methods with Value suffix (e.g., LogprobValue) write a single value directly
    */

    bool FinishReason(const std::string& reason = "") {
        String("finish_reason");
        if (!reason.empty()) {
            String(reason);
        } else {
            Null();
        }
        return true;
    }

    bool Index(int index) {
        String("index");
        Writer<StringBuffer>::Int(index);
        return true;
    }

    bool TextOffsetValue(size_t offset) {
        Writer<StringBuffer>::Uint(offset);
        return true;
    }

    bool LogprobValue(const float logprob) {
        if (logprob <= 0.0)
            Writer<StringBuffer>::Double(logprob);
        else
            Null();
        return true;
    }

    bool Logprob(const std::string& name, const float logprob) {
        String(name);
        LogprobValue(logprob);
        return true;
    }

    bool LogprobObject(const std::string& token, const float logprob) {
        StartObject();
        String("token", token);

        Logprob("logprob", logprob);
        const unsigned char* tokenBytes = reinterpret_cast<const unsigned char*>(token.c_str());
        StartArray("bytes");
        for (int j = 0; tokenBytes[j] != 0; j++)
            Writer<StringBuffer>::Int(tokenBytes[j]);
        Writer<StringBuffer>::EndArray();

        StartArray("top_logprobs");
        // See comment in original code
        Writer<StringBuffer>::EndArray();

        Writer<StringBuffer>::EndObject();
        return true;
    }

    bool MessageObject(const ParsedOutput& parsedOutput) {
        StartObject("message");
        String("content", parsedOutput.content);

        if (!parsedOutput.reasoning.empty()) {
            String("reasoning_content", parsedOutput.reasoning);
        }
        String("role", "assistant");

        StartArray("tool_calls");
        for (const ToolCall& toolCall : parsedOutput.toolCalls) {
            StartObject();
            String("id", toolCall.id);
            String("type", "function");

            StartObject("function");
            String("name", toolCall.name);
            String("arguments", toolCall.arguments);
            Writer<StringBuffer>::EndObject();

            Writer<StringBuffer>::EndObject();
        }
        Writer<StringBuffer>::EndArray();
        Writer<StringBuffer>::EndObject();
        return true;
    }

    bool Text(const ParsedOutput& parsedOutput) {
        String("text", parsedOutput.content);
        return true;
    }

    void UsageObject(const CompletionUsageStatistics& usage) {
        StartObject("usage");
        Int("prompt_tokens", usage.promptTokens);
        Int("completion_tokens", usage.completionTokens);
        Int("total_tokens", usage.calculateTotalTokens());
        Writer<StringBuffer>::EndObject();
    }
};
}  // namespace ovms
