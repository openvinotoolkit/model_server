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

#include <exception>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "graph_executor_constants.hpp"

#pragma warning(push)
#pragma warning(disable : 4324 6001 6385 6386 6326 6011 4309 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/llm/execution_context_utils.hpp"
#include "live_graph_session_impl.hpp"

namespace ovms {

MediapipeGraphExecutor::~MediapipeGraphExecutor() = default;

bool MediapipeGraphExecutor::supportsSession(const std::string& profile) const {
    return sidePacketMaps.liveGraphProfiles.count(profile) != 0;
}

Status MediapipeGraphExecutor::startSession(const std::string& profile, LiveGraphSession::Observer observer,
    std::unique_ptr<LiveGraphSession>& session) {
    const auto prepared = sidePacketMaps.liveGraphProfiles.find(profile);
    if (prepared == sidePacketMaps.liveGraphProfiles.end())
        return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID,
            "Unsupported live graph profile: " + profile);
    std::map<std::string, mediapipe::Packet> packets;
    buildInputSidePackets(packets, sidePacketMaps);
    try {
        if (!liveExecutionLease && activeInferenceGuard) {
            liveExecutionLease = std::make_shared<ActiveInferenceGuard>(std::move(*activeInferenceGuard));
            activeInferenceGuard.reset();
        }
        return createLiveGraphSession(prepared->second, std::move(packets),
            liveExecutionLease, std::move(observer), session);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to start live graph {} profile {}: {}", name, profile, e.what());
        return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, e.what());
    }
}

MediapipeGraphExecutor::MediapipeGraphExecutor(
    const std::string& name,
    const std::string& version,
    const ::mediapipe::CalculatorGraphConfig& config,
    stream_types_mapping_t inputTypes,
    stream_types_mapping_t outputTypes,
    std::vector<std::string> inputNames,
    std::vector<std::string> outputNames,
    const GraphSidePackets& sidePacketMaps,
    PythonBackend* pythonBackend,
    MediapipeServableMetricReporter* mediapipeServableMetricReporter,
    GraphIdGuard&& guard,
    std::shared_ptr<std::atomic<int64_t>> activeInferenceCount,
    std::shared_ptr<std::atomic<int64_t>> lastActivityTimeNs) :
    name(name),
    version(version),
    config(config),
    inputTypes(std::move(inputTypes)),
    outputTypes(std::move(outputTypes)),
    inputNames(std::move(inputNames)),
    outputNames(std::move(outputNames)),
    sidePacketMaps(sidePacketMaps),
    pythonBackend(pythonBackend),
    currentStreamTimestamp(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE)),
    mediapipeServableMetricReporter(mediapipeServableMetricReporter),
    guard(std::move(guard)),
    activeInferenceGuard(std::in_place, std::move(activeInferenceCount), std::move(lastActivityTimeNs)) {}
MediapipeGraphExecutor::MediapipeGraphExecutor(
    const std::string& name,
    const std::string& version,
    const ::mediapipe::CalculatorGraphConfig& config,
    stream_types_mapping_t inputTypes,
    stream_types_mapping_t outputTypes,
    std::vector<std::string> inputNames,
    std::vector<std::string> outputNames,
    const GraphSidePackets& sidePacketMaps,
    PythonBackend* pythonBackend,
    MediapipeServableMetricReporter* mediapipeServableMetricReporter,
    std::shared_ptr<std::atomic<int64_t>> activeInferenceCount,
    std::shared_ptr<std::atomic<int64_t>> lastActivityTimeNs) :
    name(name),
    version(version),
    config(config),
    inputTypes(std::move(inputTypes)),
    outputTypes(std::move(outputTypes)),
    inputNames(std::move(inputNames)),
    outputNames(std::move(outputNames)),
    sidePacketMaps(sidePacketMaps),
    pythonBackend(pythonBackend),
    currentStreamTimestamp(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE)),
    mediapipeServableMetricReporter(mediapipeServableMetricReporter),
    activeInferenceGuard(std::in_place, std::move(activeInferenceCount), std::move(lastActivityTimeNs)) {}

}  // namespace ovms
