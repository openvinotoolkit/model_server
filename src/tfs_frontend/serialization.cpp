//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#include "../logging.hpp"
#include "../precision.hpp"
#include "../status.hpp"
#include "../tensor_conversion.hpp"
#include "../tfs_frontend/tfs_utils.hpp"

namespace ovms {

static Status serializePrecision(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
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
    case ovms::Precision::STRING:
        responseOutput.set_dtype(getPrecisionAsDataType(servableOutput->getPrecision()));
        break;

    case ovms::Precision::Q78:
    case ovms::Precision::BIN:
    case ovms::Precision::BOOL:
    case ovms::Precision::MIXED:
    case ovms::Precision::CUSTOM:
    default: {
        Status status = StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION;
        SPDLOG_ERROR(status.string());
        return status;
    }
    }
    return StatusCode::OK;
}

static Status serializeShape(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    responseOutput.mutable_tensor_shape()->Clear();
    auto& effectiveNetworkOutputShape = servableOutput->getShape();
    ov::Shape actualTensorShape = tensor.get_shape();
    if (effectiveNetworkOutputShape.size() != actualTensorShape.size()) {
        SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in number of dimensions expected:{} vs actual:{}",
            servableOutput->getName(), effectiveNetworkOutputShape.size(), actualTensorShape.size());
        return StatusCode::INTERNAL_ERROR;
    }
    for (size_t i = 0; i < effectiveNetworkOutputShape.size(); ++i) {
        dimension_value_t dim = actualTensorShape[i];
        if (!effectiveNetworkOutputShape[i].match(dim)) {
            SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in dimension:{} expected:{} vs actual:{}",
                servableOutput->getName(), i, effectiveNetworkOutputShape[i].toString(), dim);
            return StatusCode::INTERNAL_ERROR;
        }
        responseOutput.mutable_tensor_shape()->add_dim()->set_size(dim);
    }
    return StatusCode::OK;
}

static void serializeOvTensorStringToTfProtoContent(tensorflow::TensorProto& proto, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    std::string* strings = tensor.data<std::string>();
    for (size_t i = 0; i < tensor.get_shape()[0]; i++) {
        proto.add_string_val(strings[i]);
    }
}
Status serializeTensorToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    if (servableOutput->getPostProcessingHint() == TensorInfo::ProcessingHint::STRING_2D_U8) {
        return convertOVTensor2DToStringResponse(tensor, responseOutput);
    }
    auto status = serializePrecision(responseOutput, servableOutput, tensor);
    if (!status.ok()) {
        return status;
    }
    status = serializeShape(responseOutput, servableOutput, tensor);
    if (!status.ok()) {
        return status;
    }
    if (servableOutput->getPostProcessingHint() == TensorInfo::ProcessingHint::STRING_NATIVE) {
        serializeOvTensorStringToTfProtoContent(responseOutput, tensor);
    } else {
        serializeContent(responseOutput.mutable_tensor_content(), tensor);
    }
    return StatusCode::OK;
}
template <>
tensorflow::TensorProto& ProtoGetter<tensorflow::serving::PredictResponse*, tensorflow::TensorProto&>::createOutput(const std::string& name) {
    OVMS_PROFILE_FUNCTION();
    return (*protoStorage->mutable_outputs())[name];
}
}  // namespace ovms
