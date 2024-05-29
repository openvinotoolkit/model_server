//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include "mediapipegraphexecutor.hpp"

#include <string>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

#if (PYTHON_DISABLE == 0)
#include "../python/python_backend.hpp"
#endif

namespace ovms {

MediapipeGraphExecutor::MediapipeGraphExecutor(
    const std::string& name,
    const std::string& version,
    const ::mediapipe::CalculatorGraphConfig& config,
    stream_types_mapping_t inputTypes,
    stream_types_mapping_t outputTypes,
    std::vector<std::string> inputNames,
    std::vector<std::string> outputNames,
    const PythonNodeResourcesMap& pythonNodeResourcesMap,
    const LLMNodeResourcesMap& llmNodeResourcesMap,
    PythonBackend* pythonBackend) :
    name(name),
    version(version),
    config(config),
    inputTypes(std::move(inputTypes)),
    outputTypes(std::move(outputTypes)),
    inputNames(std::move(inputNames)),
    outputNames(std::move(outputNames)),
    pythonNodeResourcesMap(pythonNodeResourcesMap),
    llmNodeResourcesMap(llmNodeResourcesMap),
    pythonBackend(pythonBackend),
    currentStreamTimestamp(STARTING_TIMESTAMP) {}

const std::string MediapipeGraphExecutor::PYTHON_SESSION_SIDE_PACKET_TAG = "py";
const std::string MediapipeGraphExecutor::LLM_SESSION_SIDE_PACKET_TAG = "llm";
const ::mediapipe::Timestamp MediapipeGraphExecutor::STARTING_TIMESTAMP = ::mediapipe::Timestamp(0);

}  // namespace ovms
