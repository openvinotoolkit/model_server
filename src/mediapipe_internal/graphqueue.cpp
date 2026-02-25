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
#include <map>
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

#pragma warning(push)
#pragma warning(disable : 4324 6001 6385 6386 6326 6011 4309 4005 4456 6246)
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma warning(pop)

#include "graph_executor_constants.hpp"
#include "outputstreamobserver.hpp"
namespace ovms {
GraphQueue::GraphQueue(const ::mediapipe::CalculatorGraphConfig& config, std::shared_ptr<GraphSidePackets> sidePacketMaps, int streamsLength) :
    Queue(streamsLength),
    sidePacketMaps(sidePacketMaps) {
    inferRequests.reserve(streamsLength);
    // TODO FIXME split constructor to init to handle retCodes?
    for (auto i = 0; i < streamsLength; ++i) {
        auto gh = std::make_shared<GraphHelper>();
        gh->graph = std::make_shared<::mediapipe::CalculatorGraph>();
        gh->currentTimestamp = ::mediapipe::Timestamp(0);

        auto absStatus = gh->graph->Initialize(config);
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Graph queue initialization failed: {}", absStatus.ToString());
            throw std::runtime_error(absStatus.ToString());
        }
        for (auto& name : config.output_stream()) {
            std::string streamName = getStreamName(name);
            gh->outStreamObservers[streamName] = std::shared_ptr<OutputStreamObserverI>(new NullOutputStreamObserver());  // TODO use at() FIXME
            auto& perGraphObserverFunctor = gh->outStreamObservers[streamName];
            absStatus = gh->graph->ObserveOutputStream(streamName, [&perGraphObserverFunctor](const ::mediapipe::Packet& packet) -> absl::Status { return perGraphObserverFunctor->handlePacket(packet); });
            if (!absStatus.ok()) {
                SPDLOG_ERROR("Graph queue ObserveOutputStream failed: {}", absStatus.ToString());
                throw std::runtime_error(absStatus.ToString());
            }
        }
        for (const auto& [nodeName, _] : sidePacketMaps->genAiServableMap) {
            gh->genAiExecutionContextMap[nodeName] = std::make_shared<GenAiExecutionContextHolder>();
        }
        std::map<std::string, mediapipe::Packet> inputSidePackets;
#if (PYTHON_DISABLE == 0)
        inputSidePackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(sidePacketMaps->pythonNodeResourcesMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
#endif
        inputSidePackets[LLM_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiServableMap>(sidePacketMaps->genAiServableMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        inputSidePackets[LLM_EXECUTION_CONTEXT_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiExecutionContextMap>(gh->genAiExecutionContextMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        inputSidePackets[IMAGE_GEN_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<ImageGenerationPipelinesMap>(sidePacketMaps->imageGenPipelinesMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        inputSidePackets[EMBEDDINGS_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<EmbeddingsServableMap>(sidePacketMaps->embeddingsServableMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        inputSidePackets[RERANK_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<RerankServableMap>(sidePacketMaps->rerankServableMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        inputSidePackets[STT_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<SttServableMap>(sidePacketMaps->sttServableMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        inputSidePackets[TTS_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<TtsServableMap>(sidePacketMaps->ttsServableMap).At(::mediapipe::Timestamp(STARTING_TIMESTAMP_VALUE));
        absStatus = gh->graph->StartRun(inputSidePackets);
        if (!absStatus.ok()) {
            SPDLOG_ERROR("Graph queue StartRun failed: {}", absStatus.ToString());
            throw std::runtime_error(absStatus.ToString());
        }
        inferRequests.emplace_back(std::move(gh));
    }
}
GraphQueue::~GraphQueue() {
    for (auto& graphHelper : inferRequests) {
        auto absStatus = graphHelper->graph->WaitUntilIdle();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("Graph queue WaitUntilIdle error: {}", absStatus.ToString());
        }
        absStatus = graphHelper->graph->CloseAllPacketSources();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("Graph queue CloseAllPacketSources error: {}", absStatus.ToString());
        }
        absStatus = graphHelper->graph->WaitUntilDone();
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("Graph queue WaitUntilDone error: {}", absStatus.ToString());
        }
        graphHelper->graph->Cancel();
        graphHelper->graph.reset();
    }
}
}  // namespace ovms
