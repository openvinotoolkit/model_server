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

#include <inference_engine.hpp>

#include "binaryutils.hpp"
#include "deserialization.hpp"
#include "logging.hpp"
#include "ov_utils.hpp"
#include "predict_request_validation_utils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop

namespace ovms {

Status EntryNode::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) {
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

Status EntryNode::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
    TensorMap outputs;
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

Status EntryNode::fetchResults(TensorMap& outputs) {
    auto status = validate();
    if (!status.ok()) {
        return status;
    }
    InputSink_2<TensorMap&> inputSink(outputs);
    bool isPipeline = true;
    return deserializePredictRequest_2<ConcreteTensorProtoDeserializator_2>(*request, inputsInfo, inputSink, isPipeline);
}
template <>  // TODO remove
Status InputSink_2<TensorMap&>::give(const std::string& name, ov::runtime::Tensor& tensor) {
    requester[name] = std::make_shared<ov::runtime::Tensor>(tensor);
    return StatusCode::OK;
}

template <>
Status InputSink_2<TensorMap&>::give(const std::string& name, std::shared_ptr<ov::runtime::Tensor>& tensor) {
    requester[name] = tensor;
    return StatusCode::OK;
}

Status EntryNode::isInputBinary(const std::string& name, bool& isBinary) const {
    auto it = request->inputs().find(name);
    if (it == request->inputs().end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Error during checking binary input; input: {} does not exist", name);
        return StatusCode::INTERNAL_ERROR;
    }
    isBinary = it->second.string_val_size() > 0;
    return StatusCode::OK;
}

Status EntryNode::createShardedBlob(std::shared_ptr<ov::runtime::Tensor>& dividedBlob, Precision precision, const shape_t& shape, std::shared_ptr<ov::runtime::Tensor> tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName) {
    bool isBinary = false;
    auto status = this->isInputBinary(tensorName, isBinary);
    if (!status.ok()) {
        return status;
    }
    if (isBinary) {
        return Node::createShardedBlob(dividedBlob, precision, shape, tensor, i, step, metadata, tensorName);
    }

    // if condition is perf optimization
    // when demultiplying from entry node from tensor content we can skip allocation for sharded blobs
    // and reuse memory from original tensor since its memory is kept for whole duration of predict request
    if ((precision == Precision::FP32) ||
        (precision == Precision::I32) ||
        (precision == Precision::I8) ||
        (precision == Precision::U8) ||
        (precision == Precision::I16)) {
        dividedBlob = createSharedTensor(ovmsPrecisionToIE2Precision(precision), shape, (void*)((char*)(tensor->data()) + i * step));
    } else {
        return Node::createShardedBlob(dividedBlob, precision, shape, tensor, i, step, metadata, tensorName);
    }
    return StatusCode::OK;
}

const Status EntryNode::validate() {
    static const std::set<const char*> optionalInputNames = {};
    return request_validation_utils::validate(
        *request,
        inputsInfo,
        request->model_spec().name(),
        1,
        optionalInputNames);  // Pipelines are not versioned and always reports version 1
}

}  // namespace ovms
