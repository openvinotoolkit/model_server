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

#include "kfs_utils.hpp"
#include "../logging.hpp"
#include "../precision.hpp"
#include "../status.hpp"
#include "../tensor_conversion.hpp"

namespace ovms {
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
    case ovms::Precision::STRING:
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
        serializeStringContentFrom2DU8(rawOutputContents, tensor);
    } else if (servableOutput->getPostProcessingHint() == TensorInfo::ProcessingHint::STRING_NATIVE) {
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
    if (servableOutput->getPostProcessingHint() == TensorInfo::ProcessingHint::STRING_NATIVE) {
        return StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION;
    }
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

template <> // TODO @Atobisze for other type outputgetter
Status serializePredictResponse<ov::InferRequest&, KFSRequest, KFSResponse>(
    OutputGetter<ov::InferRequest&>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const ::KFSRequest* request,
    ::KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent) {
    return serializePredictResponse(outputGetter, servableName, servableVersion, outputMap, response, outputNameChooser, useSharedOutputContent);
}
template Status serializePredictResponse<ov::InferRequest&, KFSRequest, KFSResponse>(
    OutputGetter<ov::InferRequest&>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const ::KFSRequest* request,
    ::KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent);
}  // namespace ovms
