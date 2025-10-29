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
#include <random>
#include <string>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "utils.hpp"
#include "../../../logging.hpp"

namespace ovms {
std::string generateRandomId() {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    static constexpr int idLength = 9;
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_int_distribution<> dist(0, sizeof(alphanum) - 2);

    std::string id;
    id.reserve(idLength);
    for (int i = 0; i < idLength; ++i) {
        id += alphanum[dist(rng)];
    }
    return id;
}

bool parseToolCallsFromJsonArray(const rapidjson::Document& toolsDoc, ToolCalls_t& toolCalls) {
    if (!toolsDoc.IsArray()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Expected JSON array for tool calls");
        return false;
    }

    for (const auto& toolVal : toolsDoc.GetArray()) {
        if (!toolVal.IsObject()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call is not a valid JSON object");
            continue;
        }

        ToolCall toolCall;
        toolCall.id = generateRandomId();

        if (toolVal.HasMember("name") && toolVal["name"].IsString()) {
            toolCall.name = toolVal["name"].GetString();
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call does not contain valid name field");
            continue;
        }

        if (toolVal.HasMember("arguments") && toolVal["arguments"].IsObject()) {
            rapidjson::StringBuffer sb;
            rapidjson::Writer<rapidjson::StringBuffer> toolWriter(sb);
            toolVal["arguments"].Accept(toolWriter);
            toolCall.arguments = sb.GetString();
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call does not contain valid parameters object");
            continue;
        }

        toolCalls.push_back(toolCall);
    }

    return !toolCalls.empty();
}
}  // namespace ovms
