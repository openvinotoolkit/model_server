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

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../deserialization.hpp"
#include "../execution_context.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../modelmanager.hpp"
#include "../serialization.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipegraphdefinition.hpp"  // for version in response

namespace ovms {
static Status deserializeTensor(const std::string& requestedName, const std::string& requestedVersion, const KFSRequest* request, ov::Tensor& outTensor) {
    auto requestInputItr = std::find_if(request->inputs().begin(), request->inputs().end(), [&requestedName](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == requestedName; });
    if (requestInputItr == request->inputs().end()) {
        std::stringstream ss;
        ss << "Required input: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Missing input with specific name - {}", request->model_name(), requestedVersion, details);
        return Status(StatusCode::INVALID_MISSING_INPUT, details);
    }
    if (request->raw_input_contents().size() == 0 || request->raw_input_contents().size() != request->inputs().size()) {
        std::stringstream ss;
        ss << "Cannot find data in raw_input_content for input with name: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request->model_name(), requestedVersion, details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    auto inputIndex = requestInputItr - request->inputs().begin();
    auto& bufferLocation = request->raw_input_contents().at(inputIndex);
    try {
        ov::Shape shape;
        for (int i = 0; i < requestInputItr->shape().size(); i++) {
            if (requestInputItr->shape()[i] <= 0) {
                std::stringstream ss;
                ss << "Negative or zero dimension size is not acceptable: " << tensorShapeToString(requestInputItr->shape()) << "; input name: " << requestedName;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", request->model_name(), requestedVersion, details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
            shape.push_back(requestInputItr->shape()[i]);
        }
        ov::element::Type precision = ovmsPrecisionToIE2Precision(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        size_t expectElementsCount = ov::shape_size(shape.begin(), shape.end());
        size_t expectedBytes = precision.size() * expectElementsCount;
        if (expectedBytes <= 0) {
            std::stringstream ss;
            ss << "Invalid precision with expected bytes equal to 0: " << requestInputItr->datatype() << "; input name: " << requestedName;
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] {}", request->model_name(), requestedVersion, details);
            return Status(StatusCode::INVALID_PRECISION, details);
        }
        if (expectedBytes != bufferLocation.size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedBytes << " bytes; Actual: " << bufferLocation.size() << " bytes; input name: " << requestedName;
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", request->model_name(), requestedVersion, details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
        outTensor = ov::Tensor(precision, shape, const_cast<void*>((const void*)bufferLocation.data()));
        return StatusCode::OK;
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("Kserve mediapipe request deserialization failed:{}", e.what());
    } catch (...) {
        SPDLOG_DEBUG("KServe mediapipe request deserialization failed");
    }
    return Status(StatusCode::INTERNAL_ERROR, "Unexpected error during Tensor creation");
}

MediapipeGraphExecutor::MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config) :
    name(name),
    version(version),
    config(config) {}

namespace {
enum : unsigned int {
    INITIALIZE_GRAPH,
    RUN_GRAPH,
    ADD_INPUT_PACKET,
    FETCH_OUTPUT,
    ALL_FETCH,
    TOTAL,
    TIMER_END
};
}  // namespace

static std::map<std::string, mediapipe::Packet> createInputSidePackets(const KFSRequest* request) {
    std::map<std::string, mediapipe::Packet> inputSidePackets;
    for (const auto& [name, valueChoice] : request->parameters()) {
        if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kStringParam) {
            inputSidePackets[name] = mediapipe::MakePacket<std::string>(valueChoice.string_param()).At(mediapipe::Timestamp(0));  // TODO timestamp of side packets
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            inputSidePackets[name] = mediapipe::MakePacket<int64_t>(valueChoice.int64_param()).At(mediapipe::Timestamp(0));  // TODO timestamp of side packets
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            inputSidePackets[name] = mediapipe::MakePacket<bool>(valueChoice.bool_param()).At(mediapipe::Timestamp(0));  // TODO timestamp of side packets
        } else {
            SPDLOG_DEBUG("Handling parameters of different types than: bool, string, int64 is not supported");
        }
    }
    return inputSidePackets;
}

Status MediapipeGraphExecutor::infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const {
    Timer<TIMER_END> timer;
    SPDLOG_DEBUG("Start KServe request mediapipe graph: {} execution", request->model_name());
    ::mediapipe::CalculatorGraph graph;
    auto absStatus = graph.Initialize(this->config);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("KServe request for mediapipe graph: {} initialization failed with message: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
    }
    std::unordered_map<std::string, ::mediapipe::OutputStreamPoller> outputPollers;
    for (auto& name : this->config.output_stream()) {
        auto absStatusOrPoller = graph.AddOutputStreamPoller(name);
        if (!absStatusOrPoller.ok()) {
            const std::string absMessage = absStatusOrPoller.status().ToString();
            return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR, std::move(absMessage));
        }
        outputPollers.emplace(name, std::move(absStatusOrPoller).value());
    }
    auto inputNames = this->config.input_stream();
    std::map<std::string, mediapipe::Packet> inputSidePackets{createInputSidePackets(request)};
    absStatus = graph.StartRun(inputSidePackets);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to start mediapipe graph: {} with error: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_START_ERROR, std::move(absMessage));
    }
    if (inputNames.size() != request->inputs().size()) {
        std::stringstream ss;
        ss << "Expected: " << inputNames.size() << "; Actual: " << request->inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of inputs - {}", request->model_name(), version, details);
        return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
    }
    for (auto name : inputNames) {
        SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
        ov::Tensor input_tensor;
        auto status = deserializeTensor(name, version, request, input_tensor);
        if (!status.ok()) {
            SPDLOG_DEBUG("Failed to deserialize tensor: {}", name);
            return status;
        }
        absStatus = graph.AddPacketToInputStream(
            name, ::mediapipe::MakePacket<ov::Tensor>(std::move(input_tensor)).At(::mediapipe::Timestamp(0)));
        if (!absStatus.ok()) {
            const std::string absMessage = absStatus.ToString();
            SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {} with error: {}",
                name, request->model_name(), absStatus.message(), absStatus.raw_code());
            return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, std::move(absMessage));
        }
        absStatus = graph.CloseInputStream(name);
        if (!absStatus.ok()) {
            const std::string absMessage = absStatus.ToString();
            SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {} with error: {}",
                name, request->model_name(), absMessage);
            return Status(StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR, std::move(absMessage));
        }
    }
    // receive outputs
    ::mediapipe::Packet packet;
    std::set<std::string> outputPollersWithReceivedPacket;
    for (auto& [outputStreamName, poller] : outputPollers) {
        size_t receivedOutputs = 0;
        SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
        while (poller.Next(&packet)) {
            SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
            auto received = packet.Get<ov::Tensor>();
            auto* output = response->add_outputs();
            output->set_name(outputStreamName);
            output->set_datatype(
                ovmsPrecisionToKFSPrecision(
                    ovElementTypeToOvmsPrecision(
                        received.get_element_type())));
            output->clear_shape();
            for (const auto& dim : received.get_shape()) {
                output->add_shape(dim);
            }
            response->add_raw_output_contents()->assign(reinterpret_cast<char*>(received.data()), received.get_byte_size());
            SPDLOG_TRACE("Received packet for: {} {}", outputStreamName, receivedOutputs);
            outputPollersWithReceivedPacket.insert(outputStreamName);
            ++receivedOutputs;
        }
        SPDLOG_TRACE("Received all packets for: {}", outputStreamName);
    }
    absStatus = graph.WaitUntilDone();
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Mediapipe failed to execute: {}", absMessage);
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, absMessage);
    }
    if (outputPollers.size() != outputPollersWithReceivedPacket.size()) {
        SPDLOG_DEBUG("Mediapipe failed to execute. Failed to receive all output packets");
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unknown error during mediapipe execution");
    }
    SPDLOG_DEBUG("Received all output stream packets for graph: {}", request->model_name());
    response->set_model_name(name);
    response->set_id(request->id());
    response->set_model_version(version);
    return StatusCode::OK;
}

}  // namespace ovms
