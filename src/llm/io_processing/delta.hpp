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

#include <optional>
#include <string>
#include <variant>

namespace ovms {

struct ContentDelta {
    std::string text;
};

struct ReasoningDelta {
    std::string text;
};

struct ToolCallDelta {
    int index;
    // Present only on the first delta for a given tool call index; nullopt on argument-streaming deltas.
    std::optional<std::string> id;
    std::optional<std::string> name;
    std::string arguments;
};

// Emitted when generation ends on a swallowed token: no content to carry, but the
// caller still needs to emit a finish_reason chunk.
struct FinishDelta {};

// Audio streaming chunk (omni-model path only): base64-encoded PCM16 audio.
struct AudioDelta {
    std::string base64;
};

using Delta = std::variant<ContentDelta, ReasoningDelta, ToolCallDelta, FinishDelta, AudioDelta>;

// Helper for exhaustive std::visit — CTAD deduction guide included.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...)->overloaded<Ts...>;

}  // namespace ovms
