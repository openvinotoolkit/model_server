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
#include "mediapipegraphdefinition.hpp"

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
#include "../mediapipe_internal/mediapipedemo.hpp"
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
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
namespace {
enum : unsigned int {
    TOTAL,
    TIMER_END
};
}

namespace ovms {
static ov::Tensor bruteForceDeserialize(const std::string& requestedName, const KFSRequest* request) {
    auto requestInputItr = std::find_if(request->inputs().begin(), request->inputs().end(), [&requestedName](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == requestedName; });
    if (requestInputItr == request->inputs().end()) {
        return ov::Tensor();  // TODO
    }
    auto inputIndex = requestInputItr - request->inputs().begin();
    bool deserializeFromSharedInputContents = request->raw_input_contents().size() > 0;
    auto bufferLocation = deserializeFromSharedInputContents ? &request->raw_input_contents()[inputIndex] : nullptr;
    if (!bufferLocation)
        throw 42;
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

const tensor_map_t MediapipeGraphExecutor::getInputsInfo() const {
    return this->inputsInfo;
}

const tensor_map_t MediapipeGraphExecutor::getOutputsInfo() const {
    return this->outputsInfo;
}

MediapipeGraphConfig MediapipeGraphExecutor::MGC;

Status MediapipeGraphExecutor::validateForConfigFileExistence() {
    // TODO check for existence only
    std::ifstream ifs(this->mgconfig.graphPath);
    if (!ifs.is_open()) {
        return StatusCode::FILE_INVALID;
    }
    std::string configContent;  // write directly to this @atobiszei

    ifs.seekg(0, std::ios::end);
    configContent.reserve(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    configContent.assign((std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());
    SPDLOG_DEBUG("ER:{}", configContent);
    this->chosenConfig = std::move(configContent);
    return StatusCode::OK;
}
Status MediapipeGraphExecutor::validateForConfigLoadableness() {
    bool res = ::google::protobuf::TextFormat::ParseFromString(chosenConfig, &this->config);
    if (!res) {
        SPDLOG_ERROR("ER");
        return StatusCode::FILE_INVALID;  // TODO @atobiszei error for parsing proto
    }
    return StatusCode::OK;
}
Status MediapipeGraphExecutor::validate(ModelManager& manager) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Started validation of mediapipe: {}", getName());
    ValidationResultNotifier notifier(this->status, this->loadedNotify);
    Status validationResult = validateForConfigFileExistence();
    if (!validationResult.ok()) {
        return validationResult;
    }
    validationResult = validateForConfigLoadableness();
    if (!validationResult.ok()) {
        return validationResult;
    }
    // 1 validate existence of graphdef file
    // 2 validate protoreading capabilities
    // 3 validate 1<= outputs
    // 4 validate 1<= inputs
    // 5 validate no side_packets?
    SPDLOG_ERROR("ER:{}", this->mgconfig.graphPath);
    ::mediapipe::CalculatorGraphConfig proto;
    auto status = createInputsInfo();
    if (!status.ok()) {
        throw 52;  // TODO @atobiszei
    }
    status = createOutputsInfo();
    if (!status.ok()) {
        throw 53;  // TODO @atobiszei
    }
    notifier.passed = true;
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Finished validation of mediapipe: {}", getName());
    return StatusCode::OK;
}  // TODO

MediapipeGraphExecutor::MediapipeGraphExecutor(const std::string name,
    const MediapipeGraphConfig& config,
    MetricRegistry* registry,
    const MetricConfig* metricConfig) :
    name(name),
    status(this->name) {
    mgconfig = config;
}

Status MediapipeGraphExecutor::createInputsInfo() {
    auto outputNames = config.output_stream();
    for (auto name : outputNames) {
        outputsInfo.insert({name, TensorInfo::getUnspecifiedTensorInfo()});
    }
    return StatusCode::OK;
}

Status MediapipeGraphExecutor::createOutputsInfo() {
    auto inputNames = config.input_stream();
    for (auto name : inputNames) {
        inputsInfo.insert({name, TensorInfo::getUnspecifiedTensorInfo()});
    }
    return StatusCode::OK;
}
Status MediapipeGraphExecutor::infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const {
    SPDLOG_DEBUG("Start KServe request mediapipe graph:{} execution", request->model_name());
    ::mediapipe::CalculatorGraph graph;
    auto ret = graph.Initialize(config);
    if (!ret.ok()) {
        SPDLOG_DEBUG("KServe request for mediapipe graph:{} execution failed with message: {}", request->model_name(), ret.message());
    }

    std::unordered_map<std::string, ::mediapipe::OutputStreamPoller> outputPollers;
    // TODO validate number of inputs
    // TODO validate input names against input streams
    auto outputNames = config.output_stream();
    for (auto name : outputNames) {
        auto absStatus = graph.AddOutputStreamPoller(name);
        if (!absStatus.ok()) {
            return StatusCode::NOT_IMPLEMENTED;
        }
        outputPollers.emplace(name, std::move(absStatus).value());
    }
    auto inputNames = config.input_stream();
    auto ret2 = graph.StartRun({});  // TODO retcode
    if (!ret2.ok()) {
        SPDLOG_DEBUG("Failed to start mediapipe graph: {} with error: {}", request->model_name(), ret2.message());
        throw 43;  // TODO retcode
    }
    for (auto name : inputNames) {
        SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
        ov::Tensor input_tensor = bruteForceDeserialize(name, request);
        auto abstatus = graph.AddPacketToInputStream(
            name, ::mediapipe::MakePacket<ov::Tensor>(std::move(input_tensor)).At(::mediapipe::Timestamp(0)));
        if (!abstatus.ok()) {
            SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {}. Error message: {}, error code: {}",
                name, request->model_name(), abstatus.message(), abstatus.raw_code());
            throw 44;
        }
        abstatus = graph.CloseInputStream(name);  // TODO retcode
        if (!abstatus.ok()) {
            SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {}. Error message: {}, error code: {}",
                name, request->model_name(), abstatus.message(), abstatus.raw_code());
            throw 45;
        }
        SPDLOG_ERROR("Tensor to deserialize:\"{}\"", name);
    }
    // receive outputs
    ::mediapipe::Packet packet;
    SPDLOG_ERROR("ER");
    for (auto& [outputStreamName, poller] : outputPollers) {
        SPDLOG_ERROR("ER");
        SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
        while (poller.Next(&packet)) {
            SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
            auto received = packet.Get<ov::Tensor>();
            float* dataOut = (float*)received.data();
            auto timestamp = packet.Timestamp();
            std::stringstream ss;
            ss << "ServiceImpl Received tensor: [";
            for (int x = 0; x < 10; ++x) {
                ss << dataOut[x] << " ";
            }
            ss << " ]  timestamp: " << timestamp.DebugString();
            SPDLOG_DEBUG(ss.str());
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
            float* data = (float*)received.data();
            std::stringstream ss2;
            ss2 << "[ ";
            for (size_t i = 0; i < 10; ++i) {
                ss2 << data[i] << " ";
            }
            ss2 << "]";
            SPDLOG_DEBUG("ServiceImpl OutputData: {}", ss2.str());
            outputContentString->assign((char*)received.data(), received.get_byte_size());
        }
    }
    SPDLOG_DEBUG("Received all output stream packets for graph: {}", request->model_name());
    response->set_model_name(name);
    response->set_id("1");  // TODO later
    response->set_model_version(std::to_string(VERSION));
    return StatusCode::OK;
}
}  // namespace ovms
