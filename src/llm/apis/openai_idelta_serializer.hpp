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

namespace ovms {
class IDeltaSerializer {
public:
    virtual ~IDeltaSerializer() = default;

    virtual std::string serialize(const ContentDelta& delta) const = 0;
    virtual std::string serialize(const ReasoningDelta& delta) const = 0;
    virtual std::string serialize(const ToolCallDelta& delta) const = 0;
    virtual std::string serialize(const FinishDelta& delta) const = 0;
    virtual std::string serialize(const AudioDelta& delta) const = 0;

    std::string serialize(const Delta& delta) const {
        return std::visit([this](const auto& d) { return serialize(d); }, delta);
    }
};

}  // namespace ovms
