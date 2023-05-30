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
#include <memory>
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
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipegraphdefinition.hpp"  // for version in response

namespace ovms {
static ov::Tensor bruteForceDeserialize(const std::string& requestedName, const KFSRequest* request) {
    auto requestInputItr = std::find_if(request->inputs().begin(), request->inputs().end(), [&requestedName](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == requestedName; });
    if (requestInputItr == request->inputs().end()) {
        return ov::Tensor();  // TODO
    }
    auto inputIndex = requestInputItr - request->inputs().begin();
    bool deserializeFromSharedInputContents = request->raw_input_contents().size() > 0;
    auto bufferLocation = deserializeFromSharedInputContents ? &request->raw_input_contents()[inputIndex] : nullptr;
    if (!bufferLocation) {
        SPDLOG_ERROR("Reading buffer for Mediapipe KServe is now only supported when using raw_input_contents");
        throw 42;
    }
    ov::Shape shape;
    for (int i = 0; i < requestInputItr->shape().size(); i++) {
        shape.push_back(requestInputItr->shape()[i]);
    }
    ov::element::Type precision = ovmsPrecisionToIE2Precision(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
    // TODO handle both KFS input handling ways
    try {
        auto outTensor = ov::Tensor(precision, shape, const_cast<void*>((const void*)bufferLocation->c_str()));
        return outTensor;
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("Kserve mediapipe request deserialization failed:{}", e.what());
    } catch (...) {
        SPDLOG_DEBUG("KServe mediapipe request deserialization failed");
    }
    return ov::Tensor();
}

MediapipeGraphExecutor::MediapipeGraphExecutor(const std::string& name, const ::mediapipe::CalculatorGraphConfig& config) :
    name(name),
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

Status MediapipeGraphExecutor::infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const {
    Timer<TIMER_END> timer;
    SPDLOG_DEBUG("Start KServe request mediapipe graph:{} execution", request->model_name());
    ::mediapipe::CalculatorGraph graph;
    auto absStatus = graph.Initialize(this->config);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("KServe request for mediapipe graph:{} execution failed with message: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
    }
    std::unordered_map<std::string, ::mediapipe::OutputStreamPoller> outputPollers;
    // TODO validate number of inputs
    // TODO validate input names against input streams
    auto outputNames = this->config.output_stream();
    for (auto name : outputNames) {
        auto absStatusOrPoller = graph.AddOutputStreamPoller(name);
        if (!absStatusOrPoller.ok()) {
            const std::string absMessage = absStatusOrPoller.status().ToString();
            return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR, std::move(absMessage));
        }
        outputPollers.emplace(name, std::move(absStatusOrPoller).value());
    }
    auto inputNames = this->config.input_stream();
    absStatus = graph.StartRun({});
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to start mediapipe graph: {} with error: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_START_ERROR, std::move(absMessage));
    }
    for (auto name : inputNames) {
        SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
        if ( this->)
        ov::Tensor input_tensor = bruteForceDeserialize(name, request);
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
    for (auto& [outputStreamName, poller] : outputPollers) {
        SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
        while (poller.Next(&packet)) {
            SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
            auto received = packet.Get<ov::Tensor>();
            if ((spdlog::default_logger_raw()->level() == spdlog::level::debug) &&
                (received.get_element_type() == ov::element::Type_t::f32)) {
                // TODO remove before release
                float* dataOut = (float*)received.data();
                auto timestamp = packet.Timestamp();
                std::stringstream ss;
                ss << "ServiceImpl Received tensor: [";
                auto shape = received.get_shape();
                size_t elementsCount = 1;
                for (auto& dim : shape) {
                    elementsCount *= dim;
                }
                for (size_t x = 0; x < elementsCount; ++x) {
                    ss << dataOut[x] << " ";
                }
                ss << " ]  timestamp: " << timestamp.DebugString();
                SPDLOG_DEBUG(ss.str());
            }
            auto* output = response->add_outputs();
            output->clear_shape();
            output->set_name(outputStreamName);
            auto* outputContentString = response->add_raw_output_contents();
            auto ovDtype = received.get_element_type();
            auto outputDtype = ovmsPrecisionToKFSPrecision(ovElementTypeToOvmsPrecision(ovDtype));
            output->set_datatype(outputDtype);
            auto shape = received.get_shape();
            for (const auto& dim : shape) {
                output->add_shape(dim);
            }
            if ((spdlog::default_logger_raw()->level() == spdlog::level::debug) &&
                (received.get_element_type() == ov::element::Type_t::f32)) {
                // TODO remove before release
                float* data = (float*)received.data();
                std::stringstream ss2;
                ss2 << "[ ";
                for (size_t i = 0; i < 10; ++i) {
                    ss2 << data[i] << " ";
                }
                ss2 << "]";
                SPDLOG_DEBUG("ServiceImpl OutputData: {}", ss2.str());
            }
            outputContentString->assign((char*)received.data(), received.get_byte_size());
        }
    }
    SPDLOG_DEBUG("Received all output stream packets for graph: {}", request->model_name());
    response->set_model_name(name);
    response->set_id("1");  // TODO later
    response->set_model_version(std::to_string(MediapipeGraphDefinition::VERSION));
    return StatusCode::OK;
}

}  // namespace ovms
