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
#include "serialization.hpp"

#include "ov_utils.hpp"

namespace ovms {

Status serializeBlobToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& networkOutput,
    InferenceEngine::Blob::Ptr blob) {
    responseOutput.Clear();
    if (networkOutput->getPrecision() != blob->getTensorDesc().getPrecision()) {
        SPDLOG_ERROR("Failed to serialize blob: {}. There is difference in precision expected:{} vs actual:{}",
            networkOutput->getName(), networkOutput->getPrecision(), blob->getTensorDesc().getPrecision());
        return StatusCode::INTERNAL_ERROR;
    }
    switch (networkOutput->getPrecision()) {
    case InferenceEngine::Precision::FP32:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<float>::value);
        break;
    case InferenceEngine::Precision::I32:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<int>::value);
        break;
    case InferenceEngine::Precision::I8:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<int8_t>::value);
        break;
    case InferenceEngine::Precision::U8:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<uint8_t>::value);
        break;
    case InferenceEngine::Precision::I16:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<int16_t>::value);
        break;
    // 2 byte padding [v1, v0, 0, 0, u1, u0, 0, 0, ...]
    case InferenceEngine::Precision::U16:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<uint32_t>::value);
        break;
    case InferenceEngine::Precision::FP16:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<float>::value);
        break;

    case InferenceEngine::Precision::I64:
        responseOutput.set_dtype(tensorflow::DataTypeToEnum<int32_t>::value);
        break;

    case InferenceEngine::Precision::Q78:
    case InferenceEngine::Precision::BIN:
    case InferenceEngine::Precision::BOOL:
    case InferenceEngine::Precision::MIXED:
    case InferenceEngine::Precision::CUSTOM:
    default: {
        Status status = StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION;
        SPDLOG_ERROR(status.string());
        return status;
    }
    }
    responseOutput.mutable_tensor_shape()->Clear();
    auto& effectiveNetworkOutputShape = networkOutput->getEffectiveShape();
    auto& actualBlobShape = getEffectiveBlobShape(blob);
    if (effectiveNetworkOutputShape.size() != actualBlobShape.size()) {
        SPDLOG_ERROR("Failed to serialize blob: {}. There is difference in number of dimensions expected:{} vs actual:{}",
            networkOutput->getName(), effectiveNetworkOutputShape.size(), actualBlobShape.size());
        return StatusCode::INTERNAL_ERROR;
    }
    for (size_t i = 0; i < effectiveNetworkOutputShape.size(); ++i) {
        size_t dim;
        if (effectiveNetworkOutputShape[i] != 0) {
            if (effectiveNetworkOutputShape[i] != actualBlobShape[i]) {
                SPDLOG_ERROR("Failed to serialize blob: {}. There is difference in dimension:{} expected:{} vs actual:{}",
                    networkOutput->getName(), i, effectiveNetworkOutputShape[i], actualBlobShape[i]);
                return StatusCode::INTERNAL_ERROR;
            }
            dim = effectiveNetworkOutputShape[i];
        } else {
            dim = actualBlobShape[i];
        }
        responseOutput.mutable_tensor_shape()->add_dim()->set_size(dim);
    }
    responseOutput.mutable_tensor_content()->assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<char*>(), blob->byteSize());
    return StatusCode::OK;
}

Status serializeBlobToTensorProto_2(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& networkOutput,
    ov::runtime::Tensor& blob) {
    responseOutput.Clear();
    if (networkOutput->getOvPrecision() != blob.get_element_type()) {
        SPDLOG_ERROR("Failed to serialize blob: {}. There is difference in precision expected:{} vs actual:??",  // TODO: Convert OvType to readable string.
            networkOutput->getName(), networkOutput->getPrecision());
        return StatusCode::INTERNAL_ERROR;
    }
    switch (networkOutput->getPrecision_2()) {
    case ovms::Precision::FP32:
    case ovms::Precision::I32:
    case ovms::Precision::I8:
    case ovms::Precision::U8:
    case ovms::Precision::I16:  // 2 byte padding [v1, v0, 0, 0, u1, u0, 0, 0, ...]
    case ovms::Precision::U16:
    case ovms::Precision::FP16:
    case ovms::Precision::I64:
        responseOutput.set_dtype(networkOutput->getPrecisionAsDataType());
        break;

    case ovms::Precision::Q78:  // TODO: This does not exist in OV 2.0
    case ovms::Precision::BIN:  // TODO: This does not exist in OV 2.0
    case ovms::Precision::BOOL:
    case ovms::Precision::MIXED:   // TODO: This does not exist in OV 2.0
    case ovms::Precision::CUSTOM:  // TODO: This does not exist in OV 2.0
    default: {
        Status status = StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION;
        SPDLOG_ERROR(status.string());
        return status;
    }
    }
    responseOutput.mutable_tensor_shape()->Clear();
    auto& effectiveNetworkOutputShape = networkOutput->getShape();
    // TODO: getEffectiveBlobShape(blob);
    ov::Shape actualBlobShape = blob.get_shape();
    if (effectiveNetworkOutputShape.size() != actualBlobShape.size()) {
        SPDLOG_ERROR("Failed to serialize blob: {}. There is difference in number of dimensions expected:{} vs actual:{}",
            networkOutput->getName(), effectiveNetworkOutputShape.size(), actualBlobShape.size());
        return StatusCode::INTERNAL_ERROR;
    }
    for (size_t i = 0; i < effectiveNetworkOutputShape.size(); ++i) {
        size_t dim;
        if (effectiveNetworkOutputShape[i] != 0) {
            if (effectiveNetworkOutputShape[i] != actualBlobShape[i]) {
                SPDLOG_ERROR("Failed to serialize blob: {}. There is difference in dimension:{} expected:{} vs actual:{}",
                    networkOutput->getName(), i, effectiveNetworkOutputShape[i], actualBlobShape[i]);
                return StatusCode::INTERNAL_ERROR;
            }
            dim = effectiveNetworkOutputShape[i];
        } else {
            dim = actualBlobShape[i];
        }
        responseOutput.mutable_tensor_shape()->add_dim()->set_size(dim);
    }
    responseOutput.mutable_tensor_content()->assign((char*)blob.data(), blob.get_byte_size());
    return StatusCode::OK;
}

template <>
Status OutputGetter<InferenceEngine::InferRequest&>::get(const std::string& name, InferenceEngine::Blob::Ptr& blob) {
    try {
        blob = outputSource.GetBlob(name);
    } catch (const InferenceEngine::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_SERIALIZATION_ERROR;
        SPDLOG_ERROR("{}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

template <>
Status OutputGetter_2<ov::runtime::InferRequest&>::get(const std::string& name, ov::runtime::Tensor& blob) {
    try {
        blob = outputSource.get_tensor(name);
    } catch (const ov::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_SERIALIZATION_ERROR;
        SPDLOG_ERROR("{}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

Status serializePredictResponse(
    InferenceEngine::InferRequest& inferRequest,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response) {
    Status status;
    for (const auto& pair : outputMap) {
        auto networkOutput = pair.second;
        InferenceEngine::Blob::Ptr blob;
        OutputGetter<InferenceEngine::InferRequest&> outputGetter(inferRequest);
        status = outputGetter.get(networkOutput->getName(), blob);
        if (!status.ok()) {
            return status;
        }
        auto& tensorProto = (*response->mutable_outputs())[networkOutput->getMappedName()];
        status = serializeBlobToTensorProto(tensorProto, networkOutput, blob);
        if (!status.ok()) {
            return status;
        }
    }

    return status;
}

Status serializePredictResponse_2(
    ov::runtime::InferRequest& inferRequest,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response) {
    Status status;
    for (const auto& pair : outputMap) {
        auto networkOutput = pair.second;
        ov::runtime::Tensor blob;
        OutputGetter_2<ov::runtime::InferRequest&> outputGetter(inferRequest);
        status = outputGetter.get(networkOutput->getName(), blob);
        if (!status.ok()) {
            return status;
        }
        auto& tensorProto = (*response->mutable_outputs())[networkOutput->getMappedName()];
        status = serializeBlobToTensorProto_2(tensorProto, networkOutput, blob);
        if (!status.ok()) {
            return status;
        }
    }

    return status;
}

}  // namespace ovms
