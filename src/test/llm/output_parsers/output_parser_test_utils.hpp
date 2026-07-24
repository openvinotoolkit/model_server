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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <gtest/gtest.h>

#include "../../../llm/io_processing/base_output_parser.hpp"
#include "../../../llm/io_processing/output_parser.hpp"
#include "../../../llm/ovms_text_streamer.hpp"

namespace ovms {
namespace test {

// Drives a complete token sequence through OVMSTextStreamer and accumulates all
// emitted deltas into a ParsedOutput.  This mirrors exactly what the production
// servable does in unary (non-streaming) mode: push all tokens to the streamer,
// then read the accumulated deltas.
//
// The streamer handles BPE-correct decoding, dynamic skip_special_tokens
// switching, and token-ID-based phase detection — no special "unary mode" logic
// is needed; the caller simply collects everything the callback produces.
inline ParsedOutput parseWithStreamer(
    const ov::genai::Tokenizer& tokenizer,
    OutputParser& outputParser,
    const std::vector<int64_t>& generatedTokens,
    bool toolsAvailable,
    bool userWantsSpecialTokens = false) {

    outputParser.resetStreamingState();

    ParsedOutput result;
    std::vector<ToolCall> toolCalls;

    auto callback = [&](rapidjson::Document doc, bool isLast) {
        if (!doc.IsObject()) {
            ADD_FAILURE() << "parseWithStreamer callback received non-object Document (isLast=" << isLast << ")";
            return ov::genai::StreamingStatus::RUNNING;
        }
        if (!doc.HasMember("delta")) {
            // Empty object fired at STOP when parser emitted no final delta — expected, skip silently.
            return ov::genai::StreamingStatus::RUNNING;
        }
        const auto& d = doc["delta"];
        if (!d.IsObject())
            return ov::genai::StreamingStatus::RUNNING;
        if (d.HasMember("content") && d["content"].IsString())
            result.content.append(d["content"].GetString());
        if (d.HasMember("reasoning_content") && d["reasoning_content"].IsString())
            result.reasoning.append(d["reasoning_content"].GetString());
        if (d.HasMember("tool_calls") && d["tool_calls"].IsArray()) {
            for (const auto& entry : d["tool_calls"].GetArray()) {
                if (!entry.IsObject() || !entry.HasMember("index")) continue;
                const int idx = entry["index"].GetInt();
                if (idx < 0) continue;
                const auto uidx = static_cast<size_t>(idx);
                if (uidx >= toolCalls.size()) toolCalls.resize(uidx + 1);
                auto& tc = toolCalls[uidx];
                if (entry.HasMember("id") && entry["id"].IsString())
                    tc.id = entry["id"].GetString();
                if (entry.HasMember("function") && entry["function"].IsObject()) {
                    const auto& fn = entry["function"];
                    if (fn.HasMember("name") && fn["name"].IsString())
                        tc.name = fn["name"].GetString();
                    if (fn.HasMember("arguments") && fn["arguments"].IsString())
                        tc.arguments.append(fn["arguments"].GetString());
                }
            }
        }
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Non-owning shared_ptr: outputParser is owned by the test fixture and
    // outlives the streamer which is a local variable.
    auto parserPtr = std::shared_ptr<OutputParser>(&outputParser, [](OutputParser*) {});

    const ov::AnyMap decodeParams{{ov::genai::skip_special_tokens.name(), !userWantsSpecialTokens}};
    OVMSTextStreamer streamer(tokenizer, parserPtr, toolsAvailable,
                              std::move(callback), decodeParams);

    for (int64_t token : generatedTokens)
        streamer.write(token);
    streamer.end();

    // Compact arguments JSON and drop incomplete calls that never emitted args.
    // Streaming may emit an initial name delta before malformed calls terminate;
    // unary aggregation should keep only fully materialized calls.
    ToolCalls_t completedToolCalls;
    completedToolCalls.reserve(toolCalls.size());
    for (auto& tc : toolCalls) {
        if (tc.arguments.empty()) {
            continue;
        }
        rapidjson::Document argsDoc;
        if (!argsDoc.Parse(tc.arguments.c_str()).HasParseError()) {
            rapidjson::StringBuffer sb;
            rapidjson::Writer<rapidjson::StringBuffer> w(sb);
            argsDoc.Accept(w);
            tc.arguments = sb.GetString();
        }
        completedToolCalls.push_back(std::move(tc));
    }
    result.toolCalls = std::move(completedToolCalls);
    return result;
}

}  // namespace test
}  // namespace ovms
