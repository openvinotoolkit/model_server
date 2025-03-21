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

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"

#include "outputstreamobserver.hpp"
namespace {
//const ::mediapipe::Timestamp STARTING_TIMESTAMP = ::mediapipe::Timestamp(0);  // TODO @atobisze common
const std::string PYTHON_SESSION_SIDE_PACKET_NAME = "py";
const std::string LLM_SESSION_SIDE_PACKET_NAME = "llm";
}  // namespace
namespace ovms {
GraphQueue::GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<PythonNodeResourcesMap> pythonNodeResourcesMap, std::shared_ptr<GenAiServableMap> genAiServableMap, int streamsLength) :
    Queue(streamsLength),
    pythonNodeResourcesMap(pythonNodeResourcesMap),
    genAiServableMap(genAiServableMap) {
    SPDLOG_ERROR("ER Constr graph queue:{}", (void*)this);
    inferRequests.reserve(streamsLength);
    // TODO FIXME split constructor to init to handle retCodes?
    for (auto i = 0; i < streamsLength; ++i) {
        auto gh = std::make_shared<GraphHelper>();
        gh->graph = std::make_shared<::mediapipe::CalculatorGraph>();
        gh->currentTimestamp = ::mediapipe::Timestamp(0);

        auto absStatus = gh->graph->Initialize(config);
        if (!absStatus.ok()) {
            SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
            throw 42;
        }
        for (auto& name : config.output_stream()) {
            std::string streamName = getStreamName(name);
            gh->outStreamObservers[streamName] = std::shared_ptr<OutputStreamObserverI>(new NullOutputStreamObserver());  // TODO use at() FIXME
            auto& perGraphObserverFunctor = gh->outStreamObservers[streamName];
            absStatus = gh->graph->ObserveOutputStream(streamName, [&perGraphObserverFunctor](const ::mediapipe::Packet& packet) -> absl::Status { return perGraphObserverFunctor->handlePacket(packet); });  // TODO FIXME throw?
            if (!absStatus.ok()) {
                SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
                throw 42;
            }
        }
        std::map<std::string, mediapipe::Packet> inputSidePackets;
        inputSidePackets[PYTHON_SESSION_SIDE_PACKET_NAME] = mediapipe::MakePacket<PythonNodeResourcesMap>(*pythonNodeResourcesMap)
                                                                .At(STARTING_TIMESTAMP);
        inputSidePackets[LLM_SESSION_SIDE_PACKET_NAME] = mediapipe::MakePacket<GenAiServableMap>(*genAiServableMap).At(STARTING_TIMESTAMP);
        for (auto [k, v] : inputSidePackets) {
            SPDLOG_ERROR("k:{} v", k);
        }
        SPDLOG_ERROR("ER");
        absStatus = gh->graph->StartRun(inputSidePackets);
        SPDLOG_ERROR("ER");
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Input sidePackets size:{}, python map size:{} key:{} side packet name:{}", inputSidePackets.size(), pythonNodeResourcesMap->size(), pythonNodeResourcesMap->begin()->first, PYTHON_SESSION_SIDE_PACKET_NAME);
            SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
            throw 42;
        }

        SPDLOG_ERROR("ER");
        inferRequests.emplace_back(std::move(gh));
        SPDLOG_ERROR("ER");
    }
}
GraphQueue::~GraphQueue() {
    SPDLOG_ERROR("ER Destroy graph queue:{}", (void*)this);
    for (auto& graphHelper : inferRequests) {
        auto absStatus = graphHelper->graph->WaitUntilIdle();
        if (!absStatus.ok()) {
            SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
            //        throw 42.2;
        }
        absStatus = graphHelper->graph->CloseAllPacketSources();
        if (!absStatus.ok()) {
            SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
            //      throw "as";
        }
        absStatus = graphHelper->graph->WaitUntilDone();
        if (!absStatus.ok()) {
            SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
            //    throw 42.2;
        }
        graphHelper->graph->Cancel();
        if (!absStatus.ok()) {
            SPDLOG_ERROR("ER issue:{} {}", absStatus.ToString(), (void*)this);
            //    throw 42.2;
        }
        SPDLOG_ERROR("ER");
        graphHelper->graph.reset();
        SPDLOG_ERROR("ER");
    }
    SPDLOG_ERROR("ER Destroy graph queue:{}", (void*)this);
}
}  // namespace ovms
