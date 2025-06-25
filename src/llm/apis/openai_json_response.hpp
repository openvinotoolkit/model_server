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

namespace ovms {
class OpenAiJsonResponse {
    StringBuffer buffer;
    Writer<StringBuffer> writer;

public:
    OpenAiJsonResponse() :
        writer(buffer) {}

    // Returns the serialized JSON string
    std::string toString() {
        writer.Flush();
        return buffer.GetString();
    }

    // Generic methods to start and end JSON objects and arrays
    void initialize() {
        writer.StartObject();  // {
    }

    void finalize() {
        // TODO: we can potentially validate the JSON structure here if needed
        writer.EndObject();  // }
    }

    void startObject(const std::string& name = "") {
        if (!name.empty()) {
            writer.String(name.c_str());
        }
        writer.StartObject();  // [
    }

    void endObject() {
        writer.EndObject();  // ]
    }

    void startArray(const std::string& name = "") {
        if (!name.empty()) {
            writer.String(name.c_str());
        }
        writer.StartArray();  // [
    }

    void endArray() {
        writer.EndArray();  // ]
    }

    void writeNull(const std::string& name = "") {
        if (!name.empty()) {
            writer.String(name.c_str());
        }
        writer.Null();
    }

    void writeString(const std::string& value) {
        writer.String(value.c_str());
    }

    void writeString(const std::string& name, const std::string& value) {
        writer.String(name.c_str());
        writeString(value);
    }

    void writeInt(int value) {
        writer.Int(value);
    }

    void writeInt(const std::string& name, int value) {
        writer.String(name.c_str());
        writeInt(value);
    }

    void writeUint(size_t value) {
        writer.Uint(value);
    }

    void writeUint(const std::string& name, size_t value) {
        writer.String(name.c_str());
        writeUint(value);
    }

    // API specific methods

    void writeFinishReason(const std::string& reason = "") {
        writer.String("finish_reason");
        if (!reason.empty()) {
            writer.String(reason.c_str());
        } else {
            writer.Null();
        }
    }

    void writeIndex(int index) {
        writer.String("index");
        writer.Int(index);
    }

    void writeLogprobValue(const float logprob) {
        // genai returns logaritm of probability per token which should be in the range of -inf-0
        // other values could be potentially invalid and should be treated as such
        if (logprob <= 0.0)
            writer.Double(logprob);
        else
            writer.Null();
    }

    void writeLogprobValue(const std::string& token, const float logprob) {
        writeString(token);
        writeLogprobValue(logprob);
    }

    void writeLogprob(const std::string& token, const float logprob) {
        writer.StartObject();  // {
        writer.String("token");
        writer.String(token.c_str());

        writer.String("logprob");
        writeLogprobValue(logprob);

        // Assuming tokenizer returned UTF-8 encoded string
        const unsigned char* tokenBytes = reinterpret_cast<const unsigned char*>(token.c_str());
        writer.String("bytes");
        writer.StartArray();  // [
        for (int j = 0; tokenBytes[j] != 0; j++)
            writer.Int(tokenBytes[j]);
        writer.EndArray();  // ]

        // top_logprobs are currently hardcoded to return empty array to comply with the API
        // for full support significant changes on GenAI side are required
        writer.String("top_logprobs");
        writer.StartArray();  // [
                              /*                  
        Commented out due to supported only top_logprobs 1
        writer.StartObject();  // {

        writer.String("token");
        writer.String(token.c_str());

        writer.String("logprob");
        writeLogprob(writer, logprob);
        writer.String("bytes");
        writer.StartArray();  // [
        for (int j = 0; tokenBytes[j] != 0; j++)
            writer.Int(tokenBytes[j]);
        writer.EndArray();  // ]

        writer.EndObject();  // } */
        writer.EndArray();    // ]

        writer.EndObject();  // }
    }

    void writeParsedResponse(const ParsedResponse& parsedResponse, Endpoint endpoint) {
        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            startObject("message");  // "message": {
            // content: string; Actual content of the text produced
            writeString("content", parsedResponse.content);

            if (!parsedResponse.reasoning.empty()) {
                writeString("reasoning_content", parsedResponse.reasoning);
            }
            // role: string; Role of the text producer
            // Will make sense once we have chat templates? TODO(atobisze)
            writeString("role", "assistant");  // TODO - hardcoded

            startArray("tool_calls");  // "tool_calls": [
            for (const ToolCall& toolCall : parsedResponse.toolCalls) {
                startObject();                   // {
                writeString("id", toolCall.id);  // Generate a random ID for the tool call
                writeString("type", "function");

                startObject("function");  // "function": {
                writeString("name", toolCall.name);
                writeString("arguments", toolCall.arguments);  // Assuming toolsResponse is a valid JSON
                endObject();                                   // }

                endObject();  // }
            }
            endArray();   // ]
            endObject();  // }
        } else if (endpoint == Endpoint::COMPLETIONS) {
            writeString("text", parsedResponse.content);
        }
    }

    void writeUsage(const CompletionUsageStatistics& usage) {
        startObject("usage");  // "usage": {
        writeInt("prompt_tokens", usage.promptTokens);
        writeInt("completion_tokens", usage.completionTokens);
        writeInt("total_tokens", usage.calculateTotalTokens());
        endObject();  // }
    }
};
}  // namespace ovms
