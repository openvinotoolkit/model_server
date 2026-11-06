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
#include "live_graph_session_impl.hpp"

#include <cstdint>
#include <exception>
#include <mutex>
#include <set>
#include <utility>

#include "mediapipe/framework/calculator_graph.h"
#include "src/logging.hpp"
#include "live_graph_config.hpp"
#include "streaming_s2t_graph.hpp"

namespace ovms {
namespace {
Status executionStatus(const absl::Status& status) {
    return status.ok() ? Status(StatusCode::OK) :
                         Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, status.ToString());
}

class LiveGraphSessionImpl final : public LiveGraphSession {
    std::shared_ptr<void> executionLease_;
    std::shared_ptr<const LiveGraphConfig> config_;
    mediapipe::CalculatorGraph graph_;
    StreamingS2TGraphExecution execution_{graph_};
    std::mutex inputMutex_;
    std::set<std::string> closedInputs_;
    int64_t timestamp_ = 0;
    bool cancelled_ = false;
    bool started_ = false;

public:
    LiveGraphSessionImpl(std::shared_ptr<const LiveGraphConfig> config,
        std::shared_ptr<void> executionLease) :
        executionLease_(std::move(executionLease)), config_(std::move(config)) {}

    Status start(std::map<std::string, mediapipe::Packet> sidePackets, Observer observer) {
        auto status = execution_.initialize(config_->graph);
        if (!status.ok())
            return executionStatus(status);
        graph_.SetGraphInputStreamAddMode(mediapipe::CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);
        const auto sharedObserver = std::make_shared<Observer>(std::move(observer));
        for (const auto& [name, codec] : config_->outputs) {
            status = graph_.ObserveOutputStream(name,
                [sharedObserver, name, codec](const mediapipe::Packet& packet) {
                    try {
                        const auto result = codec.observe(packet, *sharedObserver, name);
                        return result.ok() ? absl::OkStatus() : absl::InternalError(result.string());
                    } catch (const std::exception& e) {
                        SPDLOG_ERROR("Live graph observer failed: {}", e.what());
                        return absl::InternalError(e.what());
                    } catch (...) {
                        SPDLOG_ERROR("Live graph observer threw an unknown exception");
                        return absl::InternalError("Live graph observer threw an unknown exception");
                    }
                });
            if (!status.ok())
                return executionStatus(status);
        }
        execution_.addSidePackets(sidePackets);
        status = graph_.StartRun(sidePackets);
        started_ = status.ok();
        return executionStatus(status);
    }

    ~LiveGraphSessionImpl() override {
        cancel();
        if (started_) {
            const auto status = graph_.WaitUntilDone();
            if (!status.ok())
                SPDLOG_DEBUG("Live graph session shutdown: {}", status.ToString());
        }
    }

    Status send(const std::string& streamName, const LiveGraphPacket& packet) override {
        std::lock_guard<std::mutex> lock(inputMutex_);
        if (cancelled_ || closedInputs_.count(streamName))
            return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Live graph input is closed");
        const auto stream = config_->inputs.find(streamName);
        if (stream == config_->inputs.end())
            return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unknown live graph input: " + streamName);
        if (graph_.HasError()) {
            absl::Status error;
            if (graph_.GetCombinedErrors(&error))
                return executionStatus(error);
        }
        mediapipe::Packet input;
        const auto status = stream->second.encode(packet, input);
        if (!status.ok())
            return status;
        return executionStatus(graph_.AddPacketToInputStream(streamName,
            input.At(mediapipe::Timestamp(++timestamp_))));
    }

    Status closeInput(const std::string& streamName) override {
        std::lock_guard<std::mutex> lock(inputMutex_);
        if (cancelled_ || closedInputs_.count(streamName))
            return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Live graph input is closed");
        if (!config_->inputs.count(streamName))
            return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unknown live graph input: " + streamName);
        const auto status = graph_.CloseInputStream(streamName);
        if (status.ok())
            closedInputs_.insert(streamName);
        return executionStatus(status);
    }

    void cancel() override {
        std::lock_guard<std::mutex> lock(inputMutex_);
        if (cancelled_)
            return;
        cancelled_ = true;
        execution_.invalidate();
        graph_.Cancel();
    }
};
}  // namespace

Status createLiveGraphSession(std::shared_ptr<const LiveGraphConfig> config,
    std::map<std::string, mediapipe::Packet> sidePackets,
    std::shared_ptr<void> executionLease, LiveGraphSession::Observer observer,
    std::unique_ptr<LiveGraphSession>& session) {
    if (session)
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Live graph session output must be empty");
    if (!config || !observer || config->inputs.empty() || config->outputs.empty())
        return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Live graph requires a profile, streams and observer");
    for (const auto& [name, codec] : config->inputs) {
        if (name.empty() || !codec.encode)
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Live graph requires named input codecs");
    }
    for (const auto& [name, codec] : config->outputs) {
        if (name.empty() || !codec.observe)
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Live graph requires named output codecs");
    }
    auto candidate = std::make_unique<LiveGraphSessionImpl>(std::move(config), std::move(executionLease));
    const auto status = candidate->start(std::move(sidePackets), std::move(observer));
    if (status.ok())
        session = std::move(candidate);
    return status;
}
}  // namespace ovms
