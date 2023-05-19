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

#include "kfs_frontend/kfs_utils.hpp"
#include "ov_utils.hpp"
#include "precision.hpp"
#include "status.hpp"
#include "tensor_conversion.hpp"
#include "tfs_frontend/tfs_utils.hpp"

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

static Status serializePrecision(
    ::KFSResponse::InferOutputTensor& responseOutput,
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
    if (servableOutput->getPrecision() == ovms::Precision::U8 && servableOutput->getPostProcessingHint() == TensorInfo::ProcessingHint::STRING_2D_U8) {
        responseOutput.set_datatype("BYTES");
        return StatusCode::OK;
    }
    switch (servableOutput->getPrecision()) {
    case ovms::Precision::FP64:
    case ovms::Precision::FP32:
    case ovms::Precision::FP16:
    case ovms::Precision::I64:
    case ovms::Precision::I32:
    case ovms::Precision::I16:
    case ovms::Precision::I8:
    case ovms::Precision::U64:
    case ovms::Precision::U32:
    case ovms::Precision::U16:
    case ovms::Precision::U8:
    case ovms::Precision::BOOL:
        responseOutput.set_datatype(ovmsPrecisionToKFSPrecision(servableOutput->getPrecision()));
        break;
    case ovms::Precision::UNDEFINED:
    case ovms::Precision::MIXED:
    case ovms::Precision::Q78:
    case ovms::Precision::BIN:
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

static Status serializeShape(
    ::KFSResponse::InferOutputTensor& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    responseOutput.clear_shape();
    auto& effectiveNetworkOutputShape = servableOutput->getShape();
    ov::Shape actualTensorShape = tensor.get_shape();
    if (effectiveNetworkOutputShape.size() != actualTensorShape.size()) {
        SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in number of dimensions expected:{} vs actual:{}",
            servableOutput->getName(), effectiveNetworkOutputShape.size(), actualTensorShape.size());
        return StatusCode::INTERNAL_ERROR;
    }
    if (servableOutput->getPostProcessingHint() == TensorInfo::ProcessingHint::STRING_2D_U8) {
        responseOutput.add_shape(tensor.get_shape()[0]);
        return StatusCode::OK;
    }
    for (size_t i = 0; i < effectiveNetworkOutputShape.size(); ++i) {
        dimension_value_t dim = actualTensorShape[i];
        if (!effectiveNetworkOutputShape[i].match(dim)) {
            SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in dimension:{} expected:{} vs actual:{}",
                servableOutput->getName(), i, effectiveNetworkOutputShape[i].toString(), dim);
            return StatusCode::INTERNAL_ERROR;
        }
        responseOutput.add_shape(dim);
    }
    return StatusCode::OK;
}

static void serializeContent(std::string* content, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    // We only fill if the content is not already filled.
    // It can be filled in gather exit node handler.
    if (content->size() == 0) {
        content->assign((char*)tensor.data(), tensor.get_byte_size());
    }
}

static void serializeStringContent(std::string* content, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    // We only fill if the content is not already filled.
    // It can be filled in gather exit node handler.
    if (!content->empty()) {
        return;
    }

    size_t batchSize = tensor.get_shape()[0];
    size_t maxStringLen = tensor.get_shape()[1];
    for (size_t i = 0; i < batchSize; i++) {
        uint32_t strLen = strnlen((char*)tensor.data() + i * maxStringLen, maxStringLen);
        content->append(reinterpret_cast<const char*>(&strLen), sizeof(strLen));
        content->append((char*)tensor.data() + i * maxStringLen, strLen);
    }
}
#define SERIALIZE_BY_DATATYPE(contents, datatype)                                  \
    for (size_t i = 0; i < tensor.get_byte_size(); i += sizeof(datatype)) {        \
        auto value = responseOutput.mutable_contents()->contents()->Add();         \
        *value = (*(reinterpret_cast<const datatype*>((char*)tensor.data() + i))); \
    }

static void serializeContent(::inference::ModelInferResponse::InferOutputTensor& responseOutput, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    if (responseOutput.datatype() == "FP32") {
        SERIALIZE_BY_DATATYPE(mutable_fp32_contents, float)
    } else if (responseOutput.datatype() == "INT64") {
        SERIALIZE_BY_DATATYPE(mutable_int64_contents, int64_t)
    } else if (responseOutput.datatype() == "INT32") {
        SERIALIZE_BY_DATATYPE(mutable_int_contents, int32_t)
    } else if (responseOutput.datatype() == "INT16") {
        SERIALIZE_BY_DATATYPE(mutable_int_contents, int16_t)
    } else if (responseOutput.datatype() == "INT8") {
        SERIALIZE_BY_DATATYPE(mutable_int_contents, int8_t)
    } else if (responseOutput.datatype() == "UINT64") {
        SERIALIZE_BY_DATATYPE(mutable_uint64_contents, uint64_t)
    } else if (responseOutput.datatype() == "UINT32") {
        SERIALIZE_BY_DATATYPE(mutable_uint_contents, uint32_t)
    } else if (responseOutput.datatype() == "UINT16") {
        SERIALIZE_BY_DATATYPE(mutable_uint_contents, uint16_t)
    } else if (responseOutput.datatype() == "UINT8") {
        SERIALIZE_BY_DATATYPE(mutable_uint_contents, uint8_t)
    } else if (responseOutput.datatype() == "FP64") {
        SERIALIZE_BY_DATATYPE(mutable_fp64_contents, double)
    } else if (responseOutput.datatype() == "BYTES") {
        responseOutput.mutable_contents()->add_bytes_contents((char*)tensor.data(), tensor.get_byte_size());
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
    serializeContent(responseOutput.mutable_tensor_content(), tensor);
    return StatusCode::OK;
}

Status serializeTensorToTensorProtoRaw(
    ::inference::ModelInferResponse::InferOutputTensor& responseOutput,
    std::string* rawOutputContents,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    auto status = serializePrecision(responseOutput, servableOutput, tensor);
    if (!status.ok()) {
        return status;
    }
    status = serializeShape(responseOutput, servableOutput, tensor);
    if (!status.ok()) {
        return status;
    }
    if (servableOutput->getPostProcessingHint() == TensorInfo::ProcessingHint::STRING_2D_U8) {
        serializeStringContent(rawOutputContents, tensor);
    } else {
        serializeContent(rawOutputContents, tensor);
    }
    return StatusCode::OK;
}

Status serializeTensorToTensorProto(
    ::KFSResponse::InferOutputTensor& responseOutput,
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
    serializeContent(responseOutput, tensor);
    return StatusCode::OK;
}

template <>
Status OutputGetter<ov::InferRequest&>::get(const std::string& name, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
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
tensorflow::TensorProto& ProtoGetter<tensorflow::serving::PredictResponse*, tensorflow::TensorProto&>::createOutput(const std::string& name) {
    OVMS_PROFILE_FUNCTION();
    return (*protoStorage->mutable_outputs())[name];
}

template <>
::KFSResponse::InferOutputTensor& ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&>::createOutput(const std::string& name) {
    OVMS_PROFILE_FUNCTION();
    for (int i = 0; i < protoStorage->outputs_size(); i++) {
        auto& tensor = *protoStorage->mutable_outputs(i);
        if (tensor.name() == name) {
            return tensor;
        }
    }
    auto* output = protoStorage->add_outputs();
    output->set_name(name);
    return *output;
}

template <>
std::string* ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&>::createContent(const std::string& name) {
    OVMS_PROFILE_FUNCTION();
    for (int i = 0; i < protoStorage->outputs_size(); i++) {
        auto& tensor = *protoStorage->mutable_outputs(i);
        if (tensor.name() == name) {
            if (protoStorage->raw_output_contents_size() <= i) {
                return protoStorage->add_raw_output_contents();
            }
            return protoStorage->mutable_raw_output_contents(i);
        }
    }
    return protoStorage->add_raw_output_contents();
}

const std::string& getTensorInfoName(const std::string& first, const TensorInfo& tensorInfo) {
    return tensorInfo.getName();
}

const std::string& getOutputMapKeyName(const std::string& first, const TensorInfo& tensorInfo) {
    return first;
}
}  // namespace ovms
