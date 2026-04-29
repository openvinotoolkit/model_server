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
#pragma once

#include "entry_node.hpp"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "../deserialization_main.hpp"
#include "../logging.hpp"
#include "../ov_utils.hpp"
#include "../predict_request_validation_utils.hpp"
#include "../profiler.hpp"
#include "../regularovtensorfactory.hpp"
#include "../tensor_conversion.hpp"
#include "../tensorinfo.hpp"
#include "nodesession.hpp"

namespace ovms {

template <typename RequestType>
Status EntryNode<RequestType>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) {
    OVMS_PROFILE_FUNCTION();
    notifyEndQueue.push(NodeSessionKeyPair(*this, sessionId));
    return StatusCode::OK;
}

template <typename RequestType>
Status EntryNode<RequestType>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
    OVMS_PROFILE_FUNCTION();
    TensorWithSourceMap outputs;
    auto status = fetchResults(outputs);
    if (!status.ok()) {
        return status;
    }
    SessionResult metaOutputsPair{nodeSession.getNodeSessionMetadata(), std::move(outputs)};
    auto it = nodeSessionOutputs.emplace(nodeSession.getSessionKey(), std::move(metaOutputsPair));
    if (!it.second) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Failed to set entry node session results.");
        return StatusCode::UNKNOWN_ERROR;
    }
    return StatusCode::OK;
}

template <typename RequestType>
Status EntryNode<RequestType>::fetchResults(TensorWithSourceMap& outputs) {
    auto status = validate();
    if (!status.ok()) {
        return status;
    }
    InputSink<TensorWithSourceMap&> inputSink(outputs);
    bool isPipeline = true;
    return deserializePredictRequest<ConcreteTensorProtoDeserializator, InputSink<TensorWithSourceMap&>>(*request, inputsInfo, outputsInfo, inputSink, isPipeline, factories);
}

template <typename RequestType>
Status EntryNode<RequestType>::createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName) {
    bool nativeFileFormatUsed = false;
    auto status = isNativeFileFormatUsed(*this->request, tensorName, nativeFileFormatUsed);
    if (!status.ok()) {
        return status;
    }
    if (nativeFileFormatUsed) {
        return Node::createShardedTensor(dividedTensor, precision, shape, tensor, i, step, metadata, tensorName);
    }

    // if condition is perf optimization
    // when demultiplying from entry node from tensor content we can skip allocation for sharded tensors
    // and reuse memory from original tensor since its memory is kept for whole duration of predict request
    if ((precision == Precision::FP32) ||
        (precision == Precision::I32) ||
        (precision == Precision::FP64) ||
        (precision == Precision::I64) ||
        (precision == Precision::I8) ||
        (precision == Precision::U8) ||
        (precision == Precision::I16)) {
        dividedTensor = createTensorWithNoDataOwnership(ovmsPrecisionToIE2Precision(precision), shape, (void*)((char*)(tensor.data()) + i * step));
    } else {
        return Node::createShardedTensor(dividedTensor, precision, shape, tensor, i, step, metadata, std::move(tensorName));
    }
    return StatusCode::OK;
}

template <typename RequestType>
const Status EntryNode<RequestType>::validate() {
    static const std::set<std::string> optionalInputNames = {};
    return request_validation_utils::validate(
        *request,
        inputsInfo,
        outputsInfo,
        getRequestServableName(*request),
        1,
        optionalInputNames);  // Pipelines are not versioned and always reports version 1
}

template <>
inline Status InputSink<TensorWithSourceMap&>::give(const std::string& name, ov::Tensor& tensor) {
    requester.emplace(std::make_pair(name, TensorWithSource(tensor)));
    return StatusCode::OK;
}

}  //  namespace ovms
