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
#include "../logging.hpp"
#include "../model_metric_reporter.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../timer.hpp"
#pragma warning(push)
#pragma warning(disable : 4324 6001 6385 6386 6326 6011 4309 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#include "mediapipe_utils.hpp"
#include "mediapipegraphdefinition.hpp"  // for version in response and PythonNodeResourceMap
#include "packettypes.hpp"

namespace ovms {
class PythonBackend;
class ServableMetricReporter;

inline StatusCode mediapipeAbslToOvmsStatus(absl::StatusCode code) {
    if (code == absl::StatusCode::kFailedPrecondition) {  // ovms session calculator returns this status code when loading model fails
        return StatusCode::MEDIAPIPE_PRECONDITION_FAILED;
    }
    return StatusCode::MEDIAPIPE_EXECUTION_ERROR;
}

#define OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(code, message, isSuccess)  \
    _Pragma("warning(push)")                                             \
        _Pragma("warning(disable : 4456 6246)") {                        \
        auto status = code;                                              \
        if (!status.ok()) {                                              \
            std::stringstream ss;                                        \
            ss << status.string() << "; " << message;                    \
            std::lock_guard<std::mutex> lock(sendMutex);                 \
            auto status = sendErrorImpl(ss.str(), serverReaderWriter);   \
            if (!status.ok()) {                                          \
                SPDLOG_DEBUG("Writing error to disconnected client: {}", \
                    status.string());                                    \
            }                                                            \
            isSuccess = false;                                           \
        } else {                                                         \
            isSuccess = true;                                            \
        }                                                                \
    }                                                                    \
    _Pragma("warning(pop)")

class MediapipeGraphExecutor {
    const std::string name;
    const std::string version;
    const ::mediapipe::CalculatorGraphConfig config;
    stream_types_mapping_t inputTypes;
    stream_types_mapping_t outputTypes;
    const std::vector<std::string> inputNames;
    const std::vector<std::string> outputNames;

    PythonNodeResourcesMap pythonNodeResourcesMap;
    GenAiServableMap llmNodeResourcesMap;
    PythonBackend* pythonBackend;

    ::mediapipe::Timestamp currentStreamTimestamp;

    MediapipeServableMetricReporter* mediapipeServableMetricReporter;

public:
    static const std::string PYTHON_SESSION_SIDE_PACKET_TAG;
    static const std::string LLM_SESSION_SIDE_PACKET_TAG;
    static const ::mediapipe::Timestamp STARTING_TIMESTAMP;

    MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
        stream_types_mapping_t inputTypes,
        stream_types_mapping_t outputTypes,
        std::vector<std::string> inputNames, std::vector<std::string> outputNames,
        const PythonNodeResourcesMap& pythonNodeResourcesMap,
        const GenAiServableMap& llmNodeResourcesMap,
        PythonBackend* pythonBackend,
        MediapipeServableMetricReporter* mediapipeServableMetricReporter);

    template <typename RequestType, typename ResponseType>
    Status infer(const RequestType* request, ResponseType* response, ExecutionContext executionContext) {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_DEBUG("Start unary KServe request mediapipe graph: {} execution", this->name);
        MetricCounterGuard failedRequestsGuard(this->mediapipeServableMetricReporter->getRequestsMetric(executionContext, false));
        MetricGaugeGuard currentGraphsGuard(this->mediapipeServableMetricReporter->currentGraphs.get());
        ::mediapipe::CalculatorGraph graph;
        SPDLOG_ERROR("SetExecutor XXX");
        std::ignore = graph.SetExecutor("", sharedThreadPool);  // TODO FIXME
        SPDLOG_ERROR("Start unary KServe request mediapipe graph: {} initializationXXXbegin", this->name);
        MP_RETURN_ON_FAIL(graph.Initialize(this->config), std::string("failed initialization of MediaPipe graph: ") + this->name, StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
        enum : unsigned int {
            PROCESS,
            TIMER_END2
        };
        Timer<TIMER_END2> timer;
        timer.start(PROCESS);
        SPDLOG_ERROR("Start unary KServe request mediapipe graph: {} initializationXXXend", this->name);
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
        std::map<std::string, mediapipe::Packet> inputSidePackets;
        OVMS_RETURN_ON_FAIL(deserializeInputSidePacketsFromFirstRequestImpl(inputSidePackets, *request));
#if (PYTHON_DISABLE == 0)
        inputSidePackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(this->pythonNodeResourcesMap).At(STARTING_TIMESTAMP);
        inputSidePackets[LLM_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiServableMap>(this->llmNodeResourcesMap).At(STARTING_TIMESTAMP);
#endif
        SPDLOG_ERROR("Start unary KServe request mediapipe graph: {} startRunXXXbegin", this->name);
        MP_RETURN_ON_FAIL(graph.StartRun(inputSidePackets), std::string("start MediaPipe graph: ") + this->name, StatusCode::MEDIAPIPE_GRAPH_START_ERROR);
        SPDLOG_ERROR("Start unary KServe request mediapipe graph: {} startRunXXXend", this->name);

        ::mediapipe::Packet packet;
        std::set<std::string> outputPollersWithReceivedPacket;

        size_t numberOfPacketsCreated = 0;
        auto ovms_status = createAndPushPacketsImpl(
            std::shared_ptr<const RequestType>(request,
                // Custom deleter to avoid deallocation by custom holder
                // Conversion to shared_ptr is required for unified deserialization method
                // for first and subsequent requests
                [](const RequestType*) {}),
            this->inputTypes,
            this->pythonBackend,
            graph,
            this->currentStreamTimestamp,
            numberOfPacketsCreated);
        if (!ovms_status.ok()) {
            INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getGraphErrorMetric(executionContext));
            return ovms_status;
        }

        // This differs from inferStream - we require user to feed all streams
        if (this->inputNames.size() > numberOfPacketsCreated) {
            SPDLOG_DEBUG("Not all input packets created. Expected: {}, Actual: {}. Aborting execution of mediapipe graph: {}",
                this->inputNames.size(),
                numberOfPacketsCreated,
                this->name);
            return Status(StatusCode::INVALID_NO_OF_INPUTS, "Not all input packets created");
        }

        failedRequestsGuard.disable();
        INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getRequestsMetric(executionContext, true));

        // we wait idle since some calculators could hold ownership on packet content while nodes further down the graph
        // can be still processing those. Closing packet sources triggers Calculator::Close() on nodes that do not expect
        // new packets
        auto status = graph.WaitUntilIdle();
        if (!status.ok()) {  // Collect error metric after Open()
            INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getGraphErrorMetric(executionContext));
        }
        MP_RETURN_ON_FAIL(status, "graph wait until idle", mediapipeAbslToOvmsStatus(status.code()));

        MP_RETURN_ON_FAIL(graph.CloseAllPacketSources(), "graph close all packet sources", StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR);
        for (auto& [outputStreamName, poller] : outputPollers) {
            size_t receivedOutputs = 0;
            SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
            if (poller.Next(&packet)) {
                SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
                try {
                    OVMS_RETURN_ON_FAIL(
                        onPacketReadySerializeImpl(
                            getRequestId(*request),
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
        status = graph.WaitUntilDone();
        if (!status.ok()) {  // Collect error metric after Process()
            INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getGraphErrorMetric(executionContext));
        }
        MP_RETURN_ON_FAIL(status, "graph wait until done", mediapipeAbslToOvmsStatus(status.code()));
        if (outputPollers.size() != outputPollersWithReceivedPacket.size()) {
            SPDLOG_DEBUG("Mediapipe failed to execute. Failed to receive all output packets");
            return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unknown error during mediapipe execution");
        }
        timer.stop(PROCESS);
        double processTime = timer.template elapsed<std::chrono::microseconds>(PROCESS);
        OBSERVE_IF_ENABLED(this->mediapipeServableMetricReporter->getProcessingTimeMetric(executionContext), processTime);
        INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getResponsesMetric(executionContext));
        SPDLOG_DEBUG("Received all output stream packets for graph: {}", this->name);
        return StatusCode::OK;
    }

    template <typename RequestType, typename ReaderWriterType>
    Status inferStream(const RequestType& req, ReaderWriterType& serverReaderWriter, ExecutionContext executionContext) {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_DEBUG("Start MediapipeGraphExecutor::inferEx mediapipe graph: {} execution", this->name);
        std::mutex sendMutex;
        try {
            MetricGaugeGuard currentGraphs(this->mediapipeServableMetricReporter->currentGraphs.get());
            ::mediapipe::CalculatorGraph graph;
            {
                OVMS_PROFILE_SCOPE("Mediapipe graph initialization");
                // Init
                SPDLOG_DEBUG("Start unary KServe request mediapipe graph: {} initializationXXX", this->name);
                MP_RETURN_ON_FAIL(graph.Initialize(this->config), "graph initialization", StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
                SPDLOG_DEBUG("Start unary KServe request mediapipe graph: {} initializationXXX ended", this->name);
            }
            enum : unsigned int {
                PROCESS,
                TIMER_END2
            };
            Timer<TIMER_END2> timer;
            timer.start(PROCESS);
            {
                OVMS_PROFILE_SCOPE("Mediapipe graph installing packet observers");
                // Installing observers
                for (const auto& outputName : this->outputNames) {
                    MP_RETURN_ON_FAIL(graph.ObserveOutputStream(outputName, [&serverReaderWriter, &sendMutex, &outputName, &executionContext, this](const ::mediapipe::Packet& packet) -> absl::Status {
                        OVMS_PROFILE_SCOPE("Mediapipe Packet Ready Callback");
                        try {
                            std::lock_guard<std::mutex> lock(sendMutex);
                            OVMS_RETURN_MP_ERROR_ON_FAIL(onPacketReadySerializeAndSendImpl(
                                                             "" /*no ids for streaming*/,
                                                             this->name,
                                                             this->version,
                                                             outputName,
                                                             this->outputTypes.at(outputName),
                                                             packet,
                                                             serverReaderWriter),
                                "error in send packet routine");

                            auto now = std::chrono::system_clock::now();
                            auto currentTimestamp = ::mediapipe::Timestamp(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
                            OBSERVE_IF_ENABLED(this->mediapipeServableMetricReporter->getRequestLatencyMetric(executionContext), (currentTimestamp - packet.Timestamp()).Microseconds());
                            INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getResponsesMetric(executionContext));
                            return absl::OkStatus();
                        } catch (...) {
                            SPDLOG_DEBUG("Error occurred during packet serialization in mediapipe graph: {}", this->name);
                            return absl::Status(absl::StatusCode::kCancelled, "error in serialization");
                        }
                    }),
                        "output stream observer installation", StatusCode::INTERNAL_ERROR);  // Should never happen for validated graphs
                }
            }

            std::map<std::string, mediapipe::Packet> inputSidePackets;
            {
                OVMS_PROFILE_SCOPE("Mediapipe graph creating input side packets");
                OVMS_RETURN_ON_FAIL(deserializeInputSidePacketsFromFirstRequestImpl(inputSidePackets, req));
#if (PYTHON_DISABLE == 0)
                inputSidePackets[PYTHON_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<PythonNodeResourcesMap>(this->pythonNodeResourcesMap)
                                                                       .At(STARTING_TIMESTAMP);
                inputSidePackets[LLM_SESSION_SIDE_PACKET_TAG] = mediapipe::MakePacket<GenAiServableMap>(this->llmNodeResourcesMap).At(STARTING_TIMESTAMP);
#endif
            }

            {
                OVMS_PROFILE_SCOPE("Mediapipe graph start run");
                MP_RETURN_ON_FAIL(graph.StartRun(inputSidePackets), "graph start", StatusCode::MEDIAPIPE_GRAPH_START_ERROR);
            }

            size_t numberOfPacketsCreated = 0;
            {
                OVMS_PROFILE_SCOPE("Mediapipe graph deserializing first request");
                // Deserialize first request
                bool isSuccess = true;
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
                    "partial deserialization of first request", isSuccess);
                INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getRequestsMetric(executionContext, isSuccess));
            }

            // Read loop
            // Here we create ModelInferRequest with shared ownership,
            // and move it down to custom packet holder to ensure
            // lifetime is extended to lifetime of deserialized Packets.
            auto newReq = std::make_shared<RequestType>();
            while (waitForNewRequest(serverReaderWriter, *newReq)) {
                auto pstatus = validateSubsequentRequestImpl(
                    *newReq,
                    this->name,
                    this->version,
                    this->inputTypes);
                bool isSuccess = true;
                if (pstatus.ok()) {
                    OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(
                        createAndPushPacketsImpl(
                            newReq,
                            this->inputTypes,
                            this->pythonBackend,
                            graph,
                            this->currentStreamTimestamp,
                            numberOfPacketsCreated),
                        "partial deserialization of subsequent requests", isSuccess);
                } else {
                    OVMS_WRITE_ERROR_ON_FAIL_AND_CONTINUE(std::move(pstatus), "validate subsequent requests", isSuccess);
                }
                INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getRequestsMetric(executionContext, isSuccess));

                if (graph.HasError()) {
                    INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getGraphErrorMetric(executionContext));
                    SPDLOG_DEBUG("Graph {}: encountered an error, stopping the execution", this->name);
                    break;
                }

                newReq = std::make_shared<RequestType>();
            }
            {
                OVMS_PROFILE_SCOPE("MediaPipe closing all packet sources");
                SPDLOG_DEBUG("Graph {}: Closing packet sources...", this->name);
                // Close input streams
                MP_RETURN_ON_FAIL(graph.CloseAllPacketSources(), "closing all packet sources", StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR);
            }
            {
                OVMS_PROFILE_SCOPE("MediaPipe waiting until done");
                SPDLOG_DEBUG("Graph {}: Closed all packet sources. Waiting until done...", this->name);
                auto status = graph.WaitUntilDone();
                if (!status.ok()) {
                    INCREMENT_IF_ENABLED(this->mediapipeServableMetricReporter->getGraphErrorMetric(executionContext));
                }
                MP_RETURN_ON_FAIL(status, "graph wait until done", mediapipeAbslToOvmsStatus(status.code()));
                SPDLOG_DEBUG("Graph {}: Done execution", this->name);
            }
            timer.stop(PROCESS);
            double processTime = timer.template elapsed<std::chrono::microseconds>(PROCESS);
            OBSERVE_IF_ENABLED(this->mediapipeServableMetricReporter->getProcessingTimeMetric(executionContext), processTime);
            return StatusCode::OK;
        } catch (...) {
            SPDLOG_DEBUG("Graph {}: Exception while processing MediaPipe graph", this->name);
            return Status(StatusCode::UNKNOWN_ERROR, "Exception while processing MediaPipe graph");  // To be displayed in method level above
        }
    }
};
}  // namespace ovms
