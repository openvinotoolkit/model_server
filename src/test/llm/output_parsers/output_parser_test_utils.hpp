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
#include <utility>
#include <vector>

#include <openvino/genai/tokenizer.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <gtest/gtest.h>

#include "../../../llm/io_processing/base_output_parser.hpp"
#include "../../../llm/io_processing/output_parser.hpp"
#include "../../../llm/apis/openai_rapidjson_delta_serializer.hpp"
#include "../../../llm/ovms_text_streamer.hpp"

namespace ovms {
namespace test {

// Serialize a Delta to a rapidjson::Document for use in test assertions that compare
// JSON strings.  Re-parses the serializer output so tests can use HasMember() etc.
inline rapidjson::Document deltaToDocument(const Delta& d) {
    RapidJsonDeltaSerializer s;
    rapidjson::Document doc;
    std::string json = std::visit([&](const auto& v) { return s.serialize(v); }, d);
    doc.Parse(json.c_str());
    return doc;
}

// Serialize a Delta directly to the JSON string produced by RapidJsonDeltaSerializer.
inline std::string deltaToJson(const Delta& d) {
    RapidJsonDeltaSerializer s;
    return std::visit([&](const auto& v) { return s.serialize(v); }, d);
}

// Drives a complete token sequence through OVMSTextStreamer and accumulates all
// emitted deltas into a ParsedOutput.  This mirrors exactly what the production
// servable does in unary (non-streaming) mode: push all tokens to the streamer,
// then read the accumulated deltas.
inline ParsedOutput parseWithStreamer(
    const ov::genai::Tokenizer& tokenizer,
    OutputParser& outputParser,
    const std::vector<int64_t>& generatedTokens,
    bool toolsAvailable,
    bool userWantsSpecialTokens = false) {

    outputParser.resetStreamingState();

    ParsedOutput result;
    std::vector<ToolCall> toolCalls;

    auto callback = [&](Delta delta, bool /*isLast*/) {
        std::visit(overloaded{
                       [&](const ContentDelta& d) { result.content.append(d.text); },
                       [&](const ReasoningDelta& d) { result.reasoning.append(d.text); },
                       [&](const ToolCallDelta& d) {
                           if (d.index < 0)
                               return;
                           const auto idx = static_cast<size_t>(d.index);
                           if (idx >= toolCalls.size())
                               toolCalls.resize(idx + 1);
                           auto& tc = toolCalls[idx];
                           if (d.id)
                               tc.id = *d.id;
                           if (d.name)
                               tc.name = *d.name;
                           tc.arguments.append(d.arguments);
                       },
                       [](const FinishDelta&) {},
                       [](const AudioDelta&) {},
                   },
            delta);
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
    ToolCalls_t completedToolCalls;
    completedToolCalls.reserve(toolCalls.size());
    for (auto& tc : toolCalls) {
        if (tc.arguments.empty())
            continue;
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
