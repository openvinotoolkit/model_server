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
#include "graphqueue.hpp"

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "../queue.hpp"
#include "src/python/pythonnoderesources.hpp"
#include "src/llm/servable.hpp"
#include "src/assert.hpp"

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"

#include "outputstreamobserver.hpp"
namespace {
//const ::mediapipe::Timestamp STARTING_TIMESTAMP = ::mediapipe::Timestamp(0);  // TODO @atobisze common
const std::string PYTHON_SESSION_SIDE_PACKET_NAME = "py";
const std::string LLM_SESSION_SIDE_PACKET_NAME = "llm";
}  // namespace
namespace ovms {

std::shared_ptr<GraphHelper> constructGraphHelper(const ::mediapipe::CalculatorGraphConfig& config, PythonNodeResourcesMap& pythonNodeResourcesMap, GenAiServableMap& genAiServableMap) {
    SPDLOG_TRACE("Constructing GraphHelper():{}", (void*)gh.get());
    auto gh = std::make_shared<GraphHelper>();
    auto absStatus = gh->graph->Initialize(config);
    if (!absStatus.ok()) {
        SPDLOG_ERROR("Failed to initialize graph issue:{}", absStatus.ToString());
        // This would mean validation did execute fully
        ASSERT_ALWAYS(true);
    }
    for (auto& name : config.output_stream()) {
        std::string streamName = getStreamName(name);
        gh->outStreamObservers[streamName] = std::shared_ptr<OutputStreamObserverI>(new NullOutputStreamObserver());  // TODO use at() FIXME
        auto& perGraphObserverFunctor = gh->outStreamObservers[streamName];
        SPDLOG_TRACE("Installing output stream observer for output:{}", streamName);
        absStatus = gh->graph->ObserveOutputStream(streamName, [&perGraphObserverFunctor](const ::mediapipe::Packet& packet) -> absl::Status { return perGraphObserverFunctor->handlePacket(packet); });  // TODO FIXME throw?
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Failed to install output stream observer for output:{}; issue:{}", streamName, absStatus.ToString());
            return nullptr;
        }
    }
    gh->initialized = true;
    std::map<std::string, mediapipe::Packet> inputSidePackets;
    inputSidePackets[PYTHON_SESSION_SIDE_PACKET_NAME] = mediapipe::MakePacket<PythonNodeResourcesMap>(pythonNodeResourcesMap)
                                                            .At(STARTING_TIMESTAMP);
    inputSidePackets[LLM_SESSION_SIDE_PACKET_NAME] = mediapipe::MakePacket<GenAiServableMap>(genAiServableMap).At(STARTING_TIMESTAMP);
    for (auto [k, v] : inputSidePackets) {
        SPDLOG_ERROR("k:{} v", k);
    }
    SPDLOG_ERROR("ER");
    absStatus = gh->graph->StartRun(inputSidePackets);
    SPDLOG_ERROR("ER");
    if (!absStatus.ok()) {
        SPDLOG_ERROR("Input sidePackets size:{}, python map size:{} key:{} side packet name:{}", inputSidePackets.size(), pythonNodeResourcesMap.size(), pythonNodeResourcesMap.begin()->first, PYTHON_SESSION_SIDE_PACKET_NAME);
        SPDLOG_ERROR("ER issue:{}", absStatus.ToString());
        throw 42;
    }
    SPDLOG_TRACE("Constructed graph helper");
    return gh;
}
void GraphQueue::restoreStream(int streamId) {
    if (streamId < inferRequests.size()) {
        SPDLOG_ERROR("Cannot restore stream id > queue length");
        ASSERT_ALWAYS(streamId < inferRequests.size());
    }
    SPDLOG_TRACE("Restoring graph helper id:{}", streamId);
    auto gh = constructGraphHelper(*this->config, *this->pythonNodeResourcesMap, *this->genAiServableMap);
    if (gh == nullptr) {
        SPDLOG_ERROR("Failed to restore graph helper: {}", streamId);
        ASSERT_ALWAYS(false);
    }
    inferRequests[streamId] = gh;
}

GraphQueue::GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<PythonNodeResourcesMap> pythonNodeResourcesMap, std::shared_ptr<GenAiServableMap> genAiServableMap, int streamsLength) :
    Queue(streamsLength),
    config(std::make_shared<const ::mediapipe::CalculatorGraphConfig>(config)),
    pythonNodeResourcesMap(pythonNodeResourcesMap),
    genAiServableMap(genAiServableMap) {
    SPDLOG_TRACE("Constructing GraphQueue():{}", (void*)this);
    inferRequests.reserve(streamsLength);
    // TODO FIXME split constructor to init to handle retCodes?
    for (auto i = 0; i < streamsLength; ++i) {
        SPDLOG_ERROR("Constructing GraphHelper id:{}", i);
        auto gh = constructGraphHelper(*this->config, *pythonNodeResourcesMap, *genAiServableMap);
        if (gh == nullptr) {
            SPDLOG_ERROR("Failed to construct GraphHelper");
            throw 42; // FIXME @atobisze factory
        }

        inferRequests.emplace_back(std::move(gh));
    }
}
void GraphHelper::closeGraph() {
    SPDLOG_ERROR("ER");
    ASSERT_ALWAYS(this->graph.get() != nullptr);
    auto absStatus = absl::OkStatus();
    if (this->initialized) {
        SPDLOG_ERROR("Calling wait until idle graph:{}", (void*)this->graph.get());
        absStatus = this->graph->WaitUntilIdle();
    }
    if (!absStatus.ok()) {
        SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
        //        throw 42.2;
    }
    absStatus = this->graph->CloseAllPacketSources();
    if (!absStatus.ok()) {
        SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
        //      throw "as";
    }
    SPDLOG_TRACE("GraphQueue wait until done graph");
    absStatus = this->graph->WaitUntilDone();
    if (!absStatus.ok()) {
        SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
        //    throw 42.2;
    }
    this->graph->Cancel();
    if (!absStatus.ok()) {
        SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
        //    throw 42.2;
    }
    SPDLOG_ERROR("ER");

}

GraphHelper::GraphHelper() :
graph(std::make_shared<::mediapipe::CalculatorGraph>()),
currentTimestamp(::mediapipe::Timestamp(0)) {}

GraphHelper::~GraphHelper() {
    SPDLOG_TRACE("GraphHelper wait until idle graph");
    closeGraph();
    this->graph.reset();
    SPDLOG_ERROR("ER ~GraphHelper:{}", (void*) this);
}

GraphQueue::~GraphQueue() {
    SPDLOG_ERROR("ER ~GraphQueue:{}", (void*)this);
    for (auto& graphHelper : inferRequests) {
        SPDLOG_TRACE("GraphQueue wait until idle graph");
        graphHelper.reset();
        SPDLOG_ERROR("ER");
    }
    SPDLOG_ERROR("ER ~GraphQueue:{}", (void*)this);
}
// TODO FIXME @atobisze move to destructor
}  // namespace ovms
