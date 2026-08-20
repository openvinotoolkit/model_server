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
#pragma once

#include <string>

#include "src/llm/io_processing/delta.hpp"
#include "src/llm/apis/openai_idelta_serializer.hpp"

namespace ovms {

// Serializes Delta variants to the OpenAI streaming delta JSON schema using rapidjson.
//
// Output shapes:
//   ContentDelta   → {"delta":{"content":"<text>"}}
//   ReasoningDelta → {"delta":{"reasoning_content":"<text>"}}
//   ToolCallDelta  → first delta (id/name present):
//                      {"delta":{"tool_calls":[{"id":"<id>","type":"function","index":<n>,"function":{"name":"<name>"}}]}}
//                    argument delta (id/name nullopt):
//                      {"delta":{"tool_calls":[{"index":<n>,"function":{"arguments":"<args>"}}]}}
//   FinishDelta    → {}
class RapidJsonDeltaSerializer : public IDeltaSerializer {
public:
    std::string serialize(const ContentDelta& delta) const override;
    std::string serialize(const ReasoningDelta& delta) const override;
    std::string serialize(const ToolCallDelta& delta) const override;
    std::string serialize(const FinishDelta& delta) const override;
    std::string serialize(const AudioDelta& delta) const override;
};

}  // namespace ovms
