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
    auto status = validate();
    if (!status.ok()) {
        return status;
    }
    InputSink<BlobMap&> inputSink(outputs);
    bool isPipeline = true;
    return deserializePredictRequest<ConcreteTensorProtoDeserializator>(*request, inputsInfo, inputSink, isPipeline);
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
        dividedBlob = InferenceEngine::make_shared_blob<float>(dividedBlobDesc, InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<float*>() + i * step / sizeof(float));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::I32) {
        dividedBlob = InferenceEngine::make_shared_blob<int32_t>(dividedBlobDesc, InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<int32_t*>() + i * step / sizeof(int32_t));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::I8) {
        dividedBlob = InferenceEngine::make_shared_blob<int8_t>(dividedBlobDesc, InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<int8_t*>() + i * step / sizeof(int8_t));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::U8) {
        dividedBlob = InferenceEngine::make_shared_blob<uint8_t>(dividedBlobDesc, InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>() + i * step / sizeof(uint8_t));
    } else if (dividedBlobDesc.getPrecision() == InferenceEngine::Precision::I16) {
        dividedBlob = InferenceEngine::make_shared_blob<int16_t>(dividedBlobDesc, InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<int16_t*>() + i * step / sizeof(int16_t));
    } else {
        return Node::createShardedBlob(dividedBlob, dividedBlobDesc, blob, i, step, metadata, blobName);
    }
    return StatusCode::OK;
}

const Status EntryNode::validateNumberOfInputs(const tensorflow::serving::PredictRequest* request, const size_t expectedNumberOfInputs) {
    if (request->inputs_size() < 0 || expectedNumberOfInputs != static_cast<size_t>(request->inputs_size())) {
        std::stringstream ss;
        ss << "Expected: " << expectedNumberOfInputs << "; Actual: " << request->inputs_size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid number of inputs - {}", details);
        return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
    }
    return StatusCode::OK;
}

const Status EntryNode::checkIfShapeValuesNegative(const tensorflow::TensorProto& requestInput) {
    for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
        if (requestInput.tensor_shape().dim(i).size() < 0) {
            const std::string details = "Negative dimension size is not acceptable: " + TensorInfo::tensorShapeToString(requestInput.tensor_shape());
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Invalid shape - {}", details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}

const Status EntryNode::validateNumberOfBinaryInputShapeDimensions(const tensorflow::TensorProto& requestInput) {
    if (requestInput.tensor_shape().dim_size() != 1) {
        std::stringstream ss;
        ss << "Expected number of binary input shape dimensions: 1; Actual: " << requestInput.tensor_shape().dim_size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid number of shape dimensions - {}", details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

const bool EntryNode::checkBinaryInputBatchSizeMismatch(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    if (requestInput.string_val_size() <= 0) {
        return true;
    }
    if (networkInput.getEffectiveShape()[0] > 0 && networkInput.getEffectiveShape()[0] != static_cast<size_t>(requestInput.string_val_size())) {
        return true;
    }
    return false;
}

const Status EntryNode::validatePrecision(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    // Network and request must have the same precision
    if (requestInput.dtype() != networkInput.getPrecisionAsDataType()) {
        std::stringstream ss;
        ss << "Expected: " << networkInput.getPrecisionAsString()
           << "; Actual: " << TensorInfo::getDataTypeAsString(requestInput.dtype());
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid precision - {}", details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

const Status EntryNode::validateNumberOfShapeDimensions(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    // Network and request must have the same number of shape dimensions, higher than 0
    auto& shape = networkInput.getEffectiveShape();
    if (requestInput.tensor_shape().dim_size() <= 0 ||
        shape.size() != static_cast<size_t>(requestInput.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << TensorInfo::shapeToString(shape)
           << "; Actual: " << TensorInfo::tensorShapeToString(requestInput.tensor_shape());
        const std::string details = ss.str();
        SPDLOG_DEBUG("Invalid number of shape dimensions - {}", details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

const bool EntryNode::checkBatchSizeMismatch(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    if (networkInput.getEffectiveShape()[0] > 0 && static_cast<size_t>(requestInput.tensor_shape().dim(0).size()) != networkInput.getEffectiveShape()[0])
        return true;
    return false;
}

const bool EntryNode::checkShapeMismatch(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    // Network and request must have the same shape
    auto& shape = networkInput.getEffectiveShape();
    for (size_t i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
        if (requestInput.tensor_shape().dim(i).size() <= 0 ||
            (shape[i] > 0 && shape[i] != static_cast<size_t>(requestInput.tensor_shape().dim(i).size()))) {
            return true;
        }
    }
    return false;
}

const Status EntryNode::validateTensorContentSize(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    size_t expectedValueCount = 1;
    for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
        expectedValueCount *= requestInput.tensor_shape().dim(i).size();
    }

    // Network expects tensor content size or value count
    if (requestInput.dtype() == tensorflow::DataType::DT_UINT16) {
        if (requestInput.int_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(requestInput.int_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << requestInput.int_val_size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid number of values in tensor proto container - {}", details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else if (requestInput.dtype() == tensorflow::DataType::DT_HALF) {
        if (requestInput.half_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(requestInput.half_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << requestInput.half_val_size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid number of values in tensor proto container - {}", details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else {
        size_t expectedContentSize = expectedValueCount * networkInput.getPrecision().size();
        if (expectedContentSize != requestInput.tensor_content().size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedContentSize << " bytes; Actual: " << requestInput.tensor_content().size() << " bytes";
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid content size of tensor proto - {}", details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
    }
    return StatusCode::OK;
}

const Status EntryNode::validate() {
    Status finalStatus = StatusCode::OK;

    // Network and request must have the same amount of inputs.
    // This cannot be unified with model instance due to different requirements.
    // Stateful models contain more numbers of inputs due to additional state inputs.
    auto expectedNumberOfInputs = inputsInfo.size();
    finalStatus = validateNumberOfInputs(request, expectedNumberOfInputs);
    if (!finalStatus.ok())
        return finalStatus;

    for (const auto& pair : inputsInfo) {
        const auto& name = pair.first;
        auto networkInput = pair.second;
        auto it = request->inputs().find(name);

        // Network and request must have the same names of inputs
        if (it == request->inputs().end()) {
            std::stringstream ss;
            ss << "Required input: " << name;
            const std::string details = ss.str();
            SPDLOG_DEBUG("Missing input with specific name - {}", details);
            return Status(StatusCode::INVALID_MISSING_INPUT, details);
        }

        auto& requestInput = it->second;

        auto status = checkIfShapeValuesNegative(requestInput);
        if (!status.ok())
            return status;

        if (requestInput.dtype() == tensorflow::DataType::DT_STRING) {
            // binary inputs will be validated during conversion to blob
            SPDLOG_DEBUG("Received request containing binary inputs");
            status = validateNumberOfBinaryInputShapeDimensions(requestInput);
            if (!status.ok()) {
                return status;
            }

            if (checkBinaryInputBatchSizeMismatch(*networkInput, requestInput)) {
                std::stringstream ss;
                ss << "Expected: " << networkInput->getEffectiveShape()[0] << "; Actual: " << requestInput.string_val_size();
                const std::string details = ss.str();
                SPDLOG_DEBUG("Invalid batch size - {}", details);
                return Status(StatusCode::INVALID_BATCH_SIZE, details);
            }
            continue;
        }

        status = validatePrecision(*networkInput, requestInput);
        if (!status.ok())
            return status;

        status = validateNumberOfShapeDimensions(*networkInput, requestInput);
        if (!status.ok())
            return status;

        if (checkBatchSizeMismatch(*networkInput, requestInput)) {
            std::stringstream ss;
            ss << "Expected: " << networkInput->getEffectiveShape()[0] << "; Actual: " << requestInput.tensor_shape().dim(0).size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid batch size - {}", details);
            return Status(StatusCode::INVALID_BATCH_SIZE, details);
        }

        if (checkShapeMismatch(*networkInput, requestInput)) {
            std::stringstream ss;
            ss << "Expected: " << TensorInfo::shapeToString(networkInput->getEffectiveShape())
               << "; Actual: " << TensorInfo::tensorShapeToString(requestInput.tensor_shape());
            const std::string details = ss.str();
            SPDLOG_DEBUG("Invalid shape - {}", details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }

        status = validateTensorContentSize(*networkInput, requestInput);
        if (!status.ok())
            return status;
    }
    return finalStatus;
}

}  // namespace ovms
