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

#include <string>
#include <unordered_map>
#include <vector>

namespace mediapipe {
class CalculatorGraphConfig_Node;
}  // namespace mediapipe

namespace ovms {

// PythonExecutorCalculator node settings extracted by the host, so the Python plugin does not depend on MediaPipe protos.
struct PythonNodeConfig {
    std::string nodeName;
    std::string handlerPath;  // relative paths are resolved against graphPath
    std::string graphPath;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, std::string> outputsNameTagMapping;
};

// Implemented in the host (python_calculators_host).
PythonNodeConfig toPythonNodeConfig(const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig, const std::string& graphPath);

}  // namespace ovms
