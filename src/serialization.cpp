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

#include <stdexcept>

#include "inferenceresponse.hpp"
#include "inferencetensor.hpp"
#include "kfs_frontend/kfs_utils.hpp"
#include "ov_utils.hpp"
#include "pocapiinternal.hpp"
#include "tfs_frontend/tfs_utils.hpp"

namespace ovms {

static Status serializePrecision(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& servableOutput,
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
    const std::shared_ptr<TensorInfo>& servableOutput,
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

static Status serializePrecision(
    InferenceTensor& responseOutput,
    const std::shared_ptr<TensorInfo>& servableOutput,
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
        responseOutput.setDataType(getPrecisionAsOVMSDataType(servableOutput->getPrecision()));
        break;
    case ovms::Precision::BF16:
    case ovms::Precision::U4:
    case ovms::Precision::U1:
    case ovms::Precision::BOOL:
    case ovms::Precision::CUSTOM:
    case ovms::Precision::UNDEFINED:
    case ovms::Precision::DYNAMIC:
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
    const std::shared_ptr<TensorInfo>& servableOutput,
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
    const std::shared_ptr<TensorInfo>& servableOutput,
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

static Status serializeShape(
    InferenceTensor& responseOutput,
    const std::shared_ptr<TensorInfo>& servableOutput,
    ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    auto& effectiveNetworkOutputShape = servableOutput->getShape();
    ov::Shape actualTensorShape = tensor.get_shape();
    if (effectiveNetworkOutputShape.size() != actualTensorShape.size()) {
        SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in number of dimensions expected:{} vs actual:{}",
            servableOutput->getName(), effectiveNetworkOutputShape.size(), actualTensorShape.size());
        return StatusCode::INTERNAL_ERROR;
    }
    shape_t finalShape;
    for (size_t i = 0; i < effectiveNetworkOutputShape.size(); ++i) {
        dimension_value_t dim = actualTensorShape[i];
        if (!effectiveNetworkOutputShape[i].match(dim)) {
            SPDLOG_ERROR("Failed to serialize tensor: {}. There is difference in dimension:{} expected:{} vs actual:{}",
                servableOutput->getName(), i, effectiveNetworkOutputShape[i].toString(), dim);
            return StatusCode::INTERNAL_ERROR;
        }
        finalShape.push_back(dim);
    }
    responseOutput.setShape(finalShape);
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

static void serializeContent(InferenceTensor& responseOutput, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    // We only fill if the content is not already filled.
    // It can be filled in gather exit node handler.
    if (responseOutput.getBuffer() == nullptr)
        responseOutput.setBuffer(tensor.data(), tensor.get_byte_size(), OVMS_BUFFERTYPE_CPU, std::nullopt, true);
}

Status serializeTensorToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& servableOutput,
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
    serializeContent(responseOutput.mutable_tensor_content(), tensor);
    return StatusCode::OK;
}

Status serializeTensorToTensorProto(
    ::KFSResponse::InferOutputTensor& responseOutput,
    std::string* rawOutputContents,
    const std::shared_ptr<TensorInfo>& servableOutput,
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
    serializeContent(rawOutputContents, tensor);
    return StatusCode::OK;
}

Status serializeTensorToTensorProto(
    InferenceTensor& responseOutput,
    const std::shared_ptr<TensorInfo>& servableOutput,
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

template <>
InferenceTensor& ProtoGetter<InferenceResponse*, InferenceTensor&>::createOutput(const std::string& name) {
    OVMS_PROFILE_FUNCTION();
    InferenceTensor* t = nullptr;
    if (!protoStorage->hasOutput(name.c_str())) {
        auto status = protoStorage->addOutput(name);
        if (!status.ok()) {
            // Should never happen since we check for its existence above
            throw std::logic_error("internal error: cannot add output to InferenceResponse");
        }
    }
    auto status = protoStorage->getOutput(name.c_str(), &t);
    if (!status.ok()) {
        // Should never happen since we add the output above
        throw std::logic_error("internal error: cannot get output by name");
    } else {
        return *t;
    }
}

const std::string& getTensorInfoName(const std::string& first, const TensorInfo& tensorInfo) {
    return tensorInfo.getName();
}

const std::string& getOutputMapKeyName(const std::string& first, const TensorInfo& tensorInfo) {
    return first;
}
}  // namespace ovms
