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
#include "exit_node.hpp"

#include <string>
#include <utility>

#include "logging.hpp"
#include "ov_utils.hpp"
#include "serialization.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop

#include "exitnodesession.hpp"

namespace ovms {
Status ExitNode::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
    auto& exitNodeSession = static_cast<ExitNodeSession&>(nodeSession);
    return this->fetchResults(exitNodeSession.getInputBlobs());
}

Status ExitNode::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) {
    notifyEndQueue.push(NodeSessionKeyPair(*this, sessionId));
    return StatusCode::OK;
}

template <>
Status OutputGetter<const BlobMap&>::get(const std::string& name, InferenceEngine::Blob::Ptr& blob) {
    auto it = outputSource.find(name);
    if (it == outputSource.end()) {
        SPDLOG_ERROR("DUPA: name not found{}", name);
        return StatusCode::UNKNOWN_ERROR;  // TODO
    }
    SPDLOG_ERROR("ER:{}", (void*)blob.get());
    blob = it->second;
    SPDLOG_ERROR("ER:{}", (void*)blob.get());
    return StatusCode::OK;
}

Status ExitNode::fetchResults(const BlobMap& inputBlobs) {
    OutputGetter<const BlobMap&> outputGetter(inputBlobs);
    return serializePredictResponse(outputGetter, this->outputsInfo, this->response);
}

Status ExitNode::serialize(const InferenceEngine::Blob::Ptr& blob, tensorflow::TensorProto& proto) {
    // TODO to throw away?
    // Set size
    const auto& dims = getEffectiveBlobShape(blob);

    for (size_t dim : dims) {
        proto.mutable_tensor_shape()->add_dim()->set_size(dim);
    }

    // Set precision
    switch (blob->getTensorDesc().getPrecision()) {
    case InferenceEngine::Precision::FP32:
        proto.set_dtype(tensorflow::DataTypeToEnum<float>::value);
        break;
    case InferenceEngine::Precision::I32:
        proto.set_dtype(tensorflow::DataTypeToEnum<int32_t>::value);
        break;
    case InferenceEngine::Precision::I8:
        proto.set_dtype(tensorflow::DataTypeToEnum<int8_t>::value);
        break;
    case InferenceEngine::Precision::U8:
        proto.set_dtype(tensorflow::DataTypeToEnum<uint8_t>::value);
        break;
    case InferenceEngine::Precision::I16:
        proto.set_dtype(tensorflow::DataTypeToEnum<int16_t>::value);
        break;
    case InferenceEngine::Precision::U16:
        proto.set_dtype(tensorflow::DataTypeToEnum<uint32_t>::value);  // 2 byte padding [v1, v0, 0, 0, u1, u0, 0, 0, ...]
        break;
    case InferenceEngine::Precision::FP16:
        proto.set_dtype(tensorflow::DataTypeToEnum<float>::value);  // 2 byte padding [v1, v0, 0, 0, u1, u0, 0, 0, ...]
        break;
    case InferenceEngine::Precision::I64:
        proto.set_dtype(tensorflow::DataTypeToEnum<int32_t>::value);  // Manually tested that OV I64 = TF int32_t
        break;
    default:
        std::stringstream ss;
        ss << "Actual: " << TensorInfo::getPrecisionAsString(blob->getTensorDesc().getPrecision());
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Node: {}] Unsupported serialization precision - {}", getName(), details);
        Status status = Status(StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, details);
        return status;
    }

    // Set content
    proto.mutable_tensor_content()->assign((char*)blob->buffer(), blob->byteSize());

    return StatusCode::OK;
}

std::unique_ptr<NodeSession> ExitNode::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) {
    return std::make_unique<ExitNodeSession>(metadata, getName(), previous.size(), collapsingDetails);
}
}  // namespace ovms
