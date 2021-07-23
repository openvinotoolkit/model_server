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
#include <string>
#include <utility>

#include <inference_engine.hpp>

#include "binaryutils.hpp"
#include "deserialization.hpp"
#include "logging.hpp"

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
    BlobMap outputs;
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

Status EntryNode::fetchResults(BlobMap& outputs) {
    InputSink<BlobMap&> inputSink(outputs);
    bool isPipeline = true;
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*request, inputsInfo, inputSink, isPipeline);

    return status;
}

template <>
Status InputSink<BlobMap&>::give(const std::string& name, InferenceEngine::Blob::Ptr blob) {
    requester[name] = blob;
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

Status EntryNode::createShardedBlob(InferenceEngine::Blob::Ptr& dividedBlob, const InferenceEngine::TensorDesc& dividedBlobDesc, InferenceEngine::Blob::Ptr blob, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string blobName) {
    bool isBinary = false;
    auto status = this->isInputBinary(blobName, isBinary);
    if (!status.ok()) {
        return status;
    }
    if (isBinary) {
        return Node::createShardedBlob(dividedBlob, dividedBlobDesc, blob, i, step, metadata, blobName);
    }

    // if condition is perf optimization
    // when demultiplying from entry node from tensor content we can skip allocation for sharded blobs
    // and reuse memory from original blob since its memory is kept for whole duration of predict request
    if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::FP32) {
        dividedBlob = InferenceEngine::make_shared_blob<float>(dividedBlobDesc, (float*)blob->buffer() + i * step / sizeof(float));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::I32) {
        dividedBlob = InferenceEngine::make_shared_blob<int32_t>(dividedBlobDesc, (int32_t*)blob->buffer() + i * step / sizeof(int32_t));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::I8) {
        dividedBlob = InferenceEngine::make_shared_blob<int8_t>(dividedBlobDesc, (int8_t*)blob->buffer() + i * step / sizeof(int8_t));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::U8) {
        dividedBlob = InferenceEngine::make_shared_blob<uint8_t>(dividedBlobDesc, (uint8_t*)blob->buffer() + i * step / sizeof(uint8_t));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::I16) {
        dividedBlob = InferenceEngine::make_shared_blob<int16_t>(dividedBlobDesc, (int16_t*)blob->buffer() + i * step / sizeof(int16_t));
    } else {
        return Node::createShardedBlob(dividedBlob, dividedBlobDesc, blob, i, step, metadata, blobName);
    }
    return StatusCode::OK;
}

}  // namespace ovms
