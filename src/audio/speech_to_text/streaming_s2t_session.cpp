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
#include "streaming_s2t_session.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "mediapipe/framework/calculator_graph.h"
#include "src/logging.hpp"
#include "src/mediapipe_internal/streaming_s2t_graph.hpp"
#include "s2t_servable.hpp"

namespace ovms {
struct StreamingS2TSession::Impl {
    mediapipe::CalculatorGraph graph;
    StreamingS2TGraphExecution execution{graph};
    std::mutex inputMutex;
    int64_t timestamp = 0;
    bool cancelled = false;

    absl::Status send(StreamingS2TInput input) {
        std::lock_guard<std::mutex> lock(inputMutex);
        if (cancelled)
            return absl::CancelledError("Streaming STT session is closed");
        if (graph.HasError()) {
            absl::Status status;
            if (graph.GetCombinedErrors(&status))
                return status;
        }
        return graph.AddPacketToInputStream("audio",
            mediapipe::MakePacket<StreamingS2TInput>(std::move(input)).At(mediapipe::Timestamp(++timestamp)));
    }
};

StreamingS2TSession::StreamingS2TSession(std::shared_ptr<SttServable> servable, Observer observer) :
    impl_(std::make_unique<Impl>()) {
    if (!servable || !servable->streamingBackend || !observer)
        throw std::invalid_argument("Streaming STT session requires a streaming servable and observer");
    mediapipe::CalculatorGraphConfig config;
    config.set_max_queue_size(64);
    config.add_input_stream("audio");
    config.add_output_stream("events");
    config.add_input_side_packet("s2t_servable");
    auto* node = config.add_node();
    node->set_name("StreamingS2T");
    node->set_calculator("StreamingS2TCalculator");
    node->add_input_stream("AUDIO:audio");
    node->add_output_stream("EVENT:events");
    node->add_input_side_packet("STT_NODE_RESOURCES:stt_servable");
    auto status = impl_->execution.initialize(config);
    if (!status.ok())
        throw std::runtime_error(status.ToString());
    impl_->graph.SetGraphInputStreamAddMode(mediapipe::CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);
    status = impl_->graph.ObserveOutputStream("events",
        [observer = std::move(observer)](const mediapipe::Packet& packet) {
            try {
                observer(packet.Get<StreamingS2TEvent>());
                return absl::OkStatus();
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Streaming STT output observer failed: {}", e.what());
                return absl::InternalError(e.what());
            } catch (...) {
                SPDLOG_ERROR("Streaming STT output observer failed with an unknown exception");
                return absl::InternalError("Streaming STT output observer failed");
            }
        });
    if (!status.ok())
        throw std::runtime_error(status.ToString());
    std::map<std::string, mediapipe::Packet> packets;
    packets["s2t_servable"] = mediapipe::MakePacket<SttServableMap>(
                                SttServableMap{{"StreamingS2T", std::move(servable)}})
                                  .At(mediapipe::Timestamp(0));
    impl_->execution.addSidePackets(packets);
    status = impl_->graph.StartRun(packets);
    if (!status.ok())
        throw std::runtime_error(status.ToString());
}

StreamingS2TSession::~StreamingS2TSession() {
    cancel();
    const auto status = impl_->graph.WaitUntilDone();
    if (!status.ok())
        SPDLOG_DEBUG("Streaming STT session shutdown: {}", status.ToString());
}

absl::Status StreamingS2TSession::push(const std::string& utteranceId, AudioChunk audio) {
    return impl_->send({StreamingS2TInput::Type::Audio, utteranceId,
        std::make_shared<AudioChunk>(std::move(audio))});
}

absl::Status StreamingS2TSession::finish(const std::string& utteranceId) {
    return impl_->send({StreamingS2TInput::Type::Finish, utteranceId, nullptr});
}

void StreamingS2TSession::cancel() {
    {
        std::lock_guard<std::mutex> lock(impl_->inputMutex);
        impl_->cancelled = true;
    }
    impl_->execution.invalidate();
    impl_->graph.Cancel();
}

}  // namespace ovms
