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
#include <memory>
#include <string>
#include <vector>

#include "src/status.hpp"

namespace mediapipe {
class CalculatorGraphConfig_Node;
}

namespace ovms {
struct GraphSidePackets;
class PythonBackend;

class NodeInitializer {
public:
    virtual ~NodeInitializer() = default;
    virtual bool matches(const std::string& calculatorName) const = 0;
    virtual Status initialize(
        const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig,
        const std::string& graphName,
        const std::string& basePath,
        GraphSidePackets& sidePackets,
        PythonBackend* pythonBackend) = 0;
};

class NodeInitializerRegistry {
public:
    static NodeInitializerRegistry& instance();
    void add(std::unique_ptr<NodeInitializer> initializer);
    const std::vector<std::unique_ptr<NodeInitializer>>& all() const;

private:
    NodeInitializerRegistry() = default;
    std::vector<std::unique_ptr<NodeInitializer>> initializers_;
};
}  // namespace ovms
