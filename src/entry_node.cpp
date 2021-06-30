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
    // Fill outputs map with tensorflow predict request inputs. Fetch only those that are required in following nodes
    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& outputName = pair.first;
            if (outputs.find(outputName) != outputs.end()) {
                continue;
            }
            auto it = request->inputs().find(outputName);
            if (it == request->inputs().end()) {
                std::stringstream ss;
                ss << "Required input: " << outputName;
                const std::string details = ss.str();
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Missing input with specific name: {}", getName(), details);
                return Status(StatusCode::INVALID_MISSING_INPUT, details);
            }
            const auto& tensorProto = it->second;
            InferenceEngine::Blob::Ptr blob;
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Deserializing input: {}", getName(), outputName);
            auto status = deserialize(tensorProto, blob, inputsInfo.at(outputName));
            if (!status.ok()) {
                return status;
            }
            outputs[outputName] = blob;
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}]: blob with name: {} description: {} has been prepared", getName(), outputName, TensorInfo::tensorDescToString(blob->getTensorDesc()));
        }
    }

    return StatusCode::OK;
}

Status EntryNode::deserialize(const tensorflow::TensorProto& proto, InferenceEngine::Blob::Ptr& blob, const std::shared_ptr<TensorInfo>& tensorInfo) {
    if (proto.dtype() == tensorflow::DataType::DT_STRING) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Request contains binary input: {}", tensorInfo->getName());
        return deserializeBinaryInput(proto, blob, tensorInfo);
    } else {
        return deserializeNumericalInput(proto, blob);
    }
}

Status EntryNode::deserializeBinaryInput(const tensorflow::TensorProto& proto, InferenceEngine::Blob::Ptr& blob, const std::shared_ptr<TensorInfo>& tensorInfo) {
    return convertStringValToBlob(proto, blob, tensorInfo, true);
}

Status EntryNode::deserializeNumericalInput(const tensorflow::TensorProto& proto, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::TensorDesc description;
    if (proto.tensor_content().size() == 0) {
        const std::string details = "Tensor content size can't be 0";
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] {}", getName(), details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }

    // Assuming content is in proto.tensor_content

    InferenceEngine::SizeVector shape;
    for (int i = 0; i < proto.tensor_shape().dim_size(); i++) {
        shape.emplace_back(proto.tensor_shape().dim(i).size());
    }

    description.setDims(shape);

    size_t tensor_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    if (proto.tensor_content().size() != tensor_count * tensorflow::DataTypeSize(proto.dtype())) {
        std::stringstream ss;
        ss << "Expected: " << tensor_count * tensorflow::DataTypeSize(proto.dtype()) << "; Actual: " << proto.tensor_content().size();
        const std::string details = ss.str();
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node {}] Invalid size of tensor proto - {}", getName(), details);
        return Status(StatusCode::INVALID_CONTENT_SIZE, details);
    }

    // description.setLayout();  // Layout info is stored in model instance. If we find out it is required, then need to be set right before inference.
    try {
        switch (proto.dtype()) {
        case tensorflow::DataType::DT_FLOAT:
            description.setPrecision(InferenceEngine::Precision::FP32);
            blob = InferenceEngine::make_shared_blob<float>(description, (float*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_INT32:
            description.setPrecision(InferenceEngine::Precision::I32);
            blob = InferenceEngine::make_shared_blob<int32_t>(description, (int32_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_INT8:
            description.setPrecision(InferenceEngine::Precision::I8);
            blob = InferenceEngine::make_shared_blob<int8_t>(description, (int8_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_UINT8:
            description.setPrecision(InferenceEngine::Precision::U8);
            blob = InferenceEngine::make_shared_blob<uint8_t>(description, (uint8_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_INT16:
            description.setPrecision(InferenceEngine::Precision::I16);
            blob = InferenceEngine::make_shared_blob<int16_t>(description, (int16_t*)proto.tensor_content().data());
            break;
        case tensorflow::DataType::DT_HALF:
        case tensorflow::DataType::DT_UINT16:
        case tensorflow::DataType::DT_INT64:
        default: {
            std::stringstream ss;
            ss << "Actual: " << TensorInfo::getDataTypeAsString(proto.dtype());
            const std::string details = ss.str();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Unsupported deserialization precision - {}", getName(), details);
            return Status(StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, details);
        }
        }
    } catch (const InferenceEngine::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Exception thrown during deserialization from make_shared_blob; {}; exception message: {}",
            getName(), status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Exception thrown during deserialization from make_shared_blob; {}; exception message: {}",
            getName(), status.string(), e.what());
        return status;
    }
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
