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
#pragma once
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../execution_context.hpp"
#include "../metric.hpp"
#include "../status.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#include "mediapipegraphdefinition.hpp"  // for version in response and PythonNodeResourceMap
#include "packettypes.hpp"

namespace ovms {
class PythonBackend;

#define MP_RETURN_ON_FAIL(code, message, errorCode)              \
    {                                                            \
        auto absStatus = code;                                   \
        if (!absStatus.ok()) {                                   \
            const std::string absMessage = absStatus.ToString(); \
            SPDLOG_DEBUG("{} {}", message, absMessage);          \
            return Status(errorCode, std::move(absMessage));     \
        }                                                        \
    }

#define OVMS_RETURN_ON_FAIL(code) \
    {                             \
        auto status = code;       \
        if (!status.ok()) {       \
            return status;        \
        }                         \
    }

#define OVMS_RETURN_MP_ERROR_ON_FAIL(code, message)                     \
    {                                                                   \
        auto status = code;                                             \
        if (!status.ok()) {                                             \
            SPDLOG_DEBUG("{} {}", message, status.string());            \
            return absl::Status(absl::StatusCode::kCancelled, message); \
        }                                                               \
    }

#define OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(code, message)             \
    {                                                                    \
        auto status = code;                                              \
        if (!status.ok()) {                                              \
            std::stringstream ss;                                        \
            ss << status.string() << "; " << message;                    \
            std::lock_guard<std::mutex> lock(sendMutex);                 \
            auto status = sendErrorImpl(ss.str(), res);                  \
            if (!status.ok()) {                                          \
                SPDLOG_DEBUG("Writing error to disconnected client: {}", \
                    status.string());                                    \
            }                                                            \
        }                                                                \
    }

class MediapipeGraphExecutor {
    const std::string name;
    const std::string version;
    const ::mediapipe::CalculatorGraphConfig config;
    stream_types_mapping_t inputTypes;
    stream_types_mapping_t outputTypes;
    const std::vector<std::string> inputNames;
    const std::vector<std::string> outputNames;

    PythonNodeResourcesMap pythonNodeResourcesMap;
    PythonBackend* pythonBackend;

    ::mediapipe::Timestamp currentStreamTimestamp;

public:
    static const std::string PYTHON_SESSION_SIDE_PACKET_TAG;
    static const ::mediapipe::Timestamp STARTING_TIMESTAMP;

    MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
        stream_types_mapping_t inputTypes,
        stream_types_mapping_t outputTypes,
        std::vector<std::string> inputNames, std::vector<std::string> outputNames,
        const PythonNodeResourcesMap& pythonNodeResourcesMap,
        PythonBackend* pythonBackend);

    template <typename RequestType, typename ResponseType>
    Status infer(const RequestType* request, ResponseType* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) {
        SPDLOG_DEBUG("Start unary KServe request mediapipe graph: {} execution", this->name);
        ::mediapipe::CalculatorGraph graph;
        MP_RETURN_ON_FAIL(graph.Initialize(this->config), std::string("failed initialization of MediaPipe graph: ") + this->name, StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
        std::unordered_map<std::string, ::mediapipe::OutputStreamPoller> outputPollers;
        for (auto& name : this->outputNames) {
            if (name.empty()) {
                SPDLOG_DEBUG("Creating Mediapipe graph outputs name failed for: {}", name);
                return StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR;
            }
            auto absStatusOrPoller = graph.AddOutputStreamPoller(name);
            if (!absStatusOrPoller.ok()) {
                const std::string absMessage = absStatusOrPoller.status().ToString();
                SPDLOG_DEBUG("Failed to add mediapipe graph output stream poller: {} with error: {}", this->name, absMessage);
                return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR, std::move(absMessage));
            }
            outputPollers.emplace(name, std::move(absStatusOrPoller).value());
        }
        std::map<std::string, mediapipe::Packet> sideInputPackets;
        OVMS_RETURN_ON_FAIL(deserializeInputSidePacketsFromFirstRequestImpl(sideInputPackets, *request));
#if (PYTHON_DISABLE == 0)
        sideInputPackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(this->pythonNodeResourcesMap).At(STARTING_TIMESTAMP);
#endif
        MP_RETURN_ON_FAIL(graph.StartRun(sideInputPackets), std::string("start MediaPipe graph: ") + this->name, StatusCode::MEDIAPIPE_GRAPH_START_ERROR);

        ::mediapipe::Packet packet;
        std::set<std::string> outputPollersWithReceivedPacket;

        size_t numberOfPacketsCreated = 0;
        OVMS_RETURN_ON_FAIL(
            createAndPushPacketsImpl(
                std::shared_ptr<const RequestType>(request,
                    // Custom deleter to avoid deallocation by custom holder
                    // Conversion to shared_ptr is required for unified deserialization method
                    // for first and subsequent requests
                    [](const RequestType*) {}),
                this->inputTypes,
                this->pythonBackend,
                graph,
                this->currentStreamTimestamp,
                numberOfPacketsCreated));

        // This differs from inferStream - we require user to feed all streams
        if (this->inputNames.size() > numberOfPacketsCreated) {
            SPDLOG_DEBUG("Not all input packets created. Expected: {}, Actual: {}. Aborting execution of mediapipe graph: {}",
                this->inputNames.size(),
                numberOfPacketsCreated,
                this->name);
            return Status(StatusCode::INVALID_NO_OF_INPUTS, "Not all input packets created");
        }

        // we wait idle since some calculators could hold ownership on packet content while nodes further down the graph
        // can be still processing those. Closing packet sources triggers Calculator::Close() on nodes that do not expect
        // new packets
        MP_RETURN_ON_FAIL(graph.WaitUntilIdle(), "graph wait until idle", StatusCode::MEDIAPIPE_EXECUTION_ERROR);
        MP_RETURN_ON_FAIL(graph.CloseAllPacketSources(), "graph close all packet sources", StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR);
        for (auto& [outputStreamName, poller] : outputPollers) {
            size_t receivedOutputs = 0;
            SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
            if (poller.Next(&packet)) {
                SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
                try {
                    OVMS_RETURN_ON_FAIL(
                        onPacketReadySerializeImpl(
                            request->id(),
                            this->name,
                            this->version,
                            outputStreamName,
                            this->outputTypes.at(outputStreamName),
                            packet,
                            *response));
                } catch (...) {
                    return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Exception during packet serialization");
                }
                outputPollersWithReceivedPacket.insert(outputStreamName);
                ++receivedOutputs;
            }
            SPDLOG_TRACE("Received all: {} packets for: {}", receivedOutputs, outputStreamName);
        }
        MP_RETURN_ON_FAIL(graph.WaitUntilDone(), "grap wait until done", StatusCode::MEDIAPIPE_EXECUTION_ERROR);
        if (outputPollers.size() != outputPollersWithReceivedPacket.size()) {
            SPDLOG_DEBUG("Mediapipe failed to execute. Failed to receive all output packets");
            return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unknown error during mediapipe execution");
        }
        SPDLOG_DEBUG("Received all output stream packets for graph: {}", this->name);
        return StatusCode::OK;
    }

    template <typename RequestType, typename ResponseType>
    Status inferStream(const RequestType& req, ResponseType& res) {
        SPDLOG_DEBUG("Start MediapipeGraphExecutor::inferEx mediapipe graph: {} execution", this->name);
        std::mutex sendMutex;
        try {
            // Init
            ::mediapipe::CalculatorGraph graph;
            MP_RETURN_ON_FAIL(graph.Initialize(this->config), "graph initialization", StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);

            // Installing observers
            for (const auto& outputName : this->outputNames) {
                MP_RETURN_ON_FAIL(graph.ObserveOutputStream(outputName, [&res, &sendMutex, &outputName, this](const ::mediapipe::Packet& packet) -> absl::Status {
                    try {
                        // API
                        std::lock_guard<std::mutex> lock(sendMutex);
                        OVMS_RETURN_MP_ERROR_ON_FAIL(onPacketReadySerializeAndSendImpl(
                                                         "" /*no ids for streaming*/,
                                                         this->name,
                                                         this->version,
                                                         outputName,
                                                         this->outputTypes.at(outputName),
                                                         packet,
                                                         res),
                            "error in send packet routine");
                        return absl::OkStatus();
                    } catch (...) {
                        return absl::Status(absl::StatusCode::kCancelled, "error in serialization");
                    }
                }),
                    "output stream observer installation", StatusCode::INTERNAL_ERROR);  // Should never happen for validated graphs
            }

            // API
            std::map<std::string, mediapipe::Packet> inputSidePackets;
            OVMS_RETURN_ON_FAIL(deserializeInputSidePacketsFromFirstRequestImpl(inputSidePackets, req));  // API
#if (PYTHON_DISABLE == 0)
            inputSidePackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(this->pythonNodeResourcesMap)
                                                                   .At(STARTING_TIMESTAMP);
#endif
            MP_RETURN_ON_FAIL(graph.StartRun(inputSidePackets), "graph start", StatusCode::MEDIAPIPE_GRAPH_START_ERROR);

            // Deserialize first request
            // API
            size_t numberOfPacketsCreated = 0;
            OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(
                createAndPushPacketsImpl(
                    std::shared_ptr<const RequestType>(&req,
                        // Custom deleter to avoid deallocation by custom holder
                        // Conversion to shared_ptr is required for unified deserialization method
                        // for first and subsequent requests
                        [](const RequestType*) {}),
                    this->inputTypes,
                    this->pythonBackend,
                    graph,
                    this->currentStreamTimestamp,
                    numberOfPacketsCreated),
                "partial deserialization of first request");

            // Read loop
            // Here we create ModelInferRequest with shared ownership,
            // and move it down to custom packet holder to ensure
            // lifetime is extended to lifetime of deserialized Packets.
            auto newReq = std::make_shared<RequestType>();
            while (waitForNewRequest(res, *newReq)) {
                // API
                auto pstatus = validateSubsequentRequestImpl(
                    *newReq,
                    this->name,
                    this->version,
                    this->inputTypes);
                if (pstatus.ok()) {
                    OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(
                        createAndPushPacketsImpl(
                            newReq,
                            this->inputTypes,
                            this->pythonBackend,
                            graph,
                            this->currentStreamTimestamp,
                            numberOfPacketsCreated),
                        "partial deserialization of subsequent requests");
                } else {
                    OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(pstatus, "validate subsequent requests");
                }

                if (graph.HasError()) {
                    SPDLOG_DEBUG("Graph {}: encountered an error, stopping the execution", this->name);
                    break;
                }

                newReq = std::make_shared<RequestType>();
            }

            SPDLOG_DEBUG("Graph {}: Closing packet sources...", this->name);
            // Close input streams
            MP_RETURN_ON_FAIL(graph.CloseAllPacketSources(), "closing all packet sources", StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR);

            SPDLOG_DEBUG("Graph {}: Closed all packet sources. Waiting untill done...", this->name);
            MP_RETURN_ON_FAIL(graph.WaitUntilDone(), "waiting until done", StatusCode::MEDIAPIPE_EXECUTION_ERROR);
            SPDLOG_DEBUG("Graph {}: Done execution", this->name);
            return StatusCode::OK;
        } catch (...) {
            return Status(StatusCode::UNKNOWN_ERROR, "Exception while processing MediaPipe graph");  // To be displayed in method level above
        }
    }
};
}  // namespace ovms
