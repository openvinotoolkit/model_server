//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "entry_node.hpp"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "binaryutils.hpp"
#include "deserialization.hpp"
#include "logging.hpp"
#include "ov_utils.hpp"
#include "predict_request_validation_utils.hpp"
#include "profiler.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

namespace ovms {

template <typename RequestType>
Status EntryNode<RequestType>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) {
    OVMS_PROFILE_FUNCTION();
    // this should be created in EntryNode::SetInputs, or special method for entry node called
    // in event loop can be done in future release while implementing dynamic demultiplexing at
    // entry node
    NodeSessionMetadata metadata;
    auto nodeSession = getNodeSession(metadata);  // call to create session
    if (!nodeSession) {
        notifyEndQueue.push(NodeSessionKeyPair(*this, nodeSession->getSessionKey()));
        return StatusCode::INTERNAL_ERROR;
    }
    notifyEndQueue.push(NodeSessionKeyPair(*this, nodeSession->getSessionKey()));
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
    return deserializePredictRequest<ConcreteTensorProtoDeserializator>(*request, inputsInfo, inputSink, isPipeline);
}

template <>
Status InputSink<TensorWithSourceMap&>::give(const std::string& name, ov::Tensor& tensor) {
    requester.emplace(std::make_pair(name, TensorWithSource(tensor)));
    return StatusCode::OK;
}

template <>
Status EntryNode<tensorflow::serving::PredictRequest>::isInputBinary(const std::string& name, bool& isBinary) const {
    auto it = request->inputs().find(name);
    if (it == request->inputs().end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Error during checking binary input; input: {} does not exist", name);
        return StatusCode::INTERNAL_ERROR;
    }
    isBinary = it->second.string_val_size() > 0;
    return StatusCode::OK;
}
template <>
Status EntryNode<::inference::ModelInferRequest>::isInputBinary(const std::string& name, bool& isBinary) const {
    auto it = request->inputs().begin();
    while (it != request->inputs().end()) {
        if (it->name() == name) {
            break;
        }
        ++it;
    }
    if (it == request->inputs().end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Error during checking binary input; input: {} does not exist", name);
        return StatusCode::INTERNAL_ERROR;
    }
    isBinary = it->contents().bytes_contents_size() > 0;
    return StatusCode::OK;
}

template <typename RequestType>
Status EntryNode<RequestType>::createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName) {
    bool isBinary = false;
    auto status = this->isInputBinary(tensorName, isBinary);
    if (!status.ok()) {
        return status;
    }
    if (isBinary) {
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
        dividedTensor = createSharedTensor(ovmsPrecisionToIE2Precision(precision), shape, (void*)((char*)(tensor.data()) + i * step));
    } else {
        return Node::createShardedTensor(dividedTensor, precision, shape, tensor, i, step, metadata, tensorName);
    }
    return StatusCode::OK;
}

template <>
const Status EntryNode<tensorflow::serving::PredictRequest>::validate() {
    static const std::set<std::string> optionalInputNames = {};
    return request_validation_utils::validate(
        *request,
        inputsInfo,
        request->model_spec().name(),
        1,
        optionalInputNames);  // Pipelines are not versioned and always reports version 1
}
template <>
const Status EntryNode<::inference::ModelInferRequest>::validate() {
    static const std::set<std::string> optionalInputNames = {};
    return request_validation_utils::validate(
        *request,
        inputsInfo,
        request->model_name(),
        1,
        optionalInputNames);  // Pipelines are not versioned and always reports version 1
}

template Status EntryNode<tensorflow::serving::PredictRequest>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status EntryNode<::inference::ModelInferRequest>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status EntryNode<tensorflow::serving::PredictRequest>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status EntryNode<::inference::ModelInferRequest>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status EntryNode<tensorflow::serving::PredictRequest>::fetchResults(TensorWithSourceMap& outputs);
template Status EntryNode<::inference::ModelInferRequest>::fetchResults(TensorWithSourceMap& outputs);
template Status EntryNode<tensorflow::serving::PredictRequest>::isInputBinary(const std::string& name, bool& isBinary) const;
template Status EntryNode<::inference::ModelInferRequest>::isInputBinary(const std::string& name, bool& isBinary) const;
template Status EntryNode<tensorflow::serving::PredictRequest>::createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName);
template Status EntryNode<::inference::ModelInferRequest>::createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName);
template const Status EntryNode<tensorflow::serving::PredictRequest>::validate();
template const Status EntryNode<::inference::ModelInferRequest>::validate();
}  //  namespace ovms
