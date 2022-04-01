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

Status serializeTensorToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& servableOutput,
    ov::Tensor& tensor) {
    responseOutput.Clear();
    if (servableOutput->getOvPrecision() != tensor.get_element_type()) {
        SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in precision expected:{} vs actual:{}",
            servableOutput->getName(),
            TensorInfo::getPrecisionAsString(servableOutput->getPrecision()),
            tensor.get_element_type().get_type_name());
        return StatusCode::INTERNAL_ERROR;
    }
    switch (servableOutput->getPrecision()) {
    case ovms::Precision::FP32:
    case ovms::Precision::I32:
    case ovms::Precision::FP64:
    case ovms::Precision::I8:
    case ovms::Precision::U8:
    case ovms::Precision::I16:  // 2 byte padding [v1, v0, 0, 0, u1, u0, 0, 0, ...]
    case ovms::Precision::U16:
    case ovms::Precision::FP16:
    case ovms::Precision::I64:
        responseOutput.set_dtype(servableOutput->getPrecisionAsDataType());
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
    auto& effectiveNetworkOutputShape = servableOutput->getShape();
    ov::Shape actualTensorShape = tensor.get_shape();
    if (effectiveNetworkOutputShape.size() != actualTensorShape.size()) {
        SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in number of dimensions expected:{} vs actual:{}",
            servableOutput->getName(), effectiveNetworkOutputShape.size(), actualTensorShape.size());
        return StatusCode::INTERNAL_ERROR;
    }
    for (size_t i = 0; i < effectiveNetworkOutputShape.size(); ++i) {
        dimension_value_t dim;
        if (!effectiveNetworkOutputShape[i].match(actualTensorShape[i])) {
            SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in dimension:{} expected:{} vs actual:{}",
                servableOutput->getName(), i, effectiveNetworkOutputShape[i].toString(), actualTensorShape[i]);
            return StatusCode::INTERNAL_ERROR;
        }
        dim = actualTensorShape[i];
        responseOutput.mutable_tensor_shape()->add_dim()->set_size(dim);
    }
    responseOutput.mutable_tensor_content()->assign((char*)tensor.data(), tensor.get_byte_size());
    return StatusCode::OK;
}

template <>
Status OutputGetter<ov::InferRequest&>::get(const std::string& name, ov::Tensor& tensor) {
    try {
        tensor = outputSource.get_tensor(name);
    } catch (const ov::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_SERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

template <>
tensorflow::TensorProto& ProtoGetter<tensorflow::serving::PredictResponse*, tensorflow::TensorProto&>::get(const std::string& name) {
    return (*protoStorage->mutable_outputs())[name];
}

const std::string& getTensorInfoName(const std::string& first, const TensorInfo& tensorInfo) {
    return tensorInfo.getName();
}

const std::string& getOutputMapKeyName(const std::string& first, const TensorInfo& tensorInfo) {
    return first;
}
}  // namespace ovms
