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
#pragma once
#include "serialization.hpp"
#include <memory>
#include <string>

#include <openvino/openvino.hpp>

#include "buffer.hpp"
#include "capi_utils.hpp"
#include "inferencerequest.hpp"
#include "inferenceresponse.hpp"
#include "inferencetensor.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../serialization_common.hpp"
#include "../tensorinfo.hpp"

namespace ovms {
template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    InferenceResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    uint32_t outputId = 0;
    for (const auto& [outputName, outputInfo] : outputMap) {
        ov::Tensor tensor;
        status = outputGetter.get(outputNameChooser(outputName, *outputInfo), tensor);
        if (!status.ok()) {
            return status;
        }
        auto servableMetaPrecision = outputInfo->getPrecision();
        auto actualPrecision = ovElementTypeToOvmsPrecision(tensor.get_element_type());
        if (servableMetaPrecision != actualPrecision) {
            return StatusCode::INTERNAL_ERROR;
        }
        if (!outputInfo->getShape().match(tensor.get_shape())) {
            return StatusCode::INTERNAL_ERROR;
        }
        switch (servableMetaPrecision) {
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
            break;
        case ovms::Precision::BF16:
        case ovms::Precision::U4:
        case ovms::Precision::U1:
        case ovms::Precision::BOOL:  // ?
        case ovms::Precision::CUSTOM:
        case ovms::Precision::UNDEFINED:
        case ovms::Precision::DYNAMIC:
        case ovms::Precision::MIXED:
        case ovms::Precision::Q78:
        case ovms::Precision::BIN:
        case ovms::Precision::STRING:
        default: {
            Status status = StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION;
            SPDLOG_ERROR(status.string());
            return status;
        }
        }
        // Mapped name for single model result serialization: possible mapping_config.json setting
        // For DAG: setting in pipeline output configuration
        status = response->addOutput(
            outputInfo->getMappedName(),
            getPrecisionAsOVMSDataType(actualPrecision),
            reinterpret_cast<const int64_t*>(tensor.get_shape().data()),
            tensor.get_shape().size());
        if (status == StatusCode::DOUBLE_TENSOR_INSERT) {
            // DAG demultiplexer CAPI handling
            // there is performance optimization so that during gather stage we do not double copy nodes
            // outputs first to intermediate shard tensors and then to gathered tensor in response
            return StatusCode::OK;
        }
        if (!status.ok()) {
            SPDLOG_ERROR("Cannot serialize output with name:{} for servable name:{}; version:{}; error: duplicate output name",
                outputName, response->getServableName(), response->getServableVersion());
            return StatusCode::INTERNAL_ERROR;
        }
        const std::string* outputNameFromCapiTensor = nullptr;
        InferenceTensor* outputTensorFromResponse{nullptr};
        status = response->getOutput(outputId, &outputNameFromCapiTensor, &outputTensorFromResponse);
        ++outputId;
        if (!status.ok()) {
            SPDLOG_ERROR("Cannot serialize output with name:{} for servable name:{}; version:{}; error: cannot find inserted input",
                outputName, response->getServableName(), response->getServableVersion());
            return StatusCode::INTERNAL_ERROR;
        }
        outputTensorFromResponse->setBuffer(
            tensor.data(),
            tensor.get_byte_size(),
            OVMS_BUFFERTYPE_CPU,
            std::nullopt,
            true);
        OV_LOGGER("ov::Tensor:{} data():{}, ov::Tensor::get_byte_size():{}", (void*)&tensor, tensor.data(), tensor.get_byte_size());
    }
    return StatusCode::OK;
}

template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const InferenceRequest* request,
    InferenceResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {  // does not apply for C-API frontend
    OVMS_PROFILE_FUNCTION();
    Status status;
    uint32_t outputId = 0;
    for (const auto& [outputName, outputInfo] : outputMap) {
        ov::Tensor tensor;
        OV_LOGGER("ov::Tensor(): {}", (void*)&tensor);
        status = outputGetter.get(outputNameChooser(outputName, *outputInfo), tensor);
        if (!status.ok()) {
            return status;
        }
        OV_LOGGER("ov::Tensor: {}, tensor.get_element_type()", (void*)&tensor);
        auto servableMetaPrecision = outputInfo->getPrecision();
        auto actualPrecision = ovElementTypeToOvmsPrecision(tensor.get_element_type());
        if (servableMetaPrecision != actualPrecision) {
            return StatusCode::INTERNAL_ERROR;
        }
        OV_LOGGER("ov::Tensor: {}, tensor.get_shape()", (void*)&tensor);
        if (!outputInfo->getShape().match(tensor.get_shape())) {
            return StatusCode::INTERNAL_ERROR;
        }
        switch (servableMetaPrecision) {
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
        case ovms::Precision::STRING:
            break;
        case ovms::Precision::BF16:
        case ovms::Precision::U4:
        case ovms::Precision::U1:
        case ovms::Precision::BOOL:  // ?
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
        // Mapped name for single model result serialization: possible mapping_config.json setting
        OV_LOGGER("ov::Tensor: {}, tensor.get_shape()", (void*)&tensor);
        // For DAG: setting in pipeline output configuration
        status = response->addOutput(
            outputInfo->getMappedName(),
            getPrecisionAsOVMSDataType(actualPrecision),
            reinterpret_cast<const int64_t*>(tensor.get_shape().data()),
            tensor.get_shape().size());
        if (status == StatusCode::DOUBLE_TENSOR_INSERT) {
            // DAG demultiplexer CAPI handling
            // there is performance optimization so that during gather stage we do not double copy nodes
            // outputs first to intermediate shard tensors and then to gathered tensor in response
            return StatusCode::OK;
        }
        if (!status.ok()) {
            SPDLOG_ERROR("Cannot serialize output with name:{} for servable name:{}; version:{}; error: duplicate output name",
                outputName, response->getServableName(), response->getServableVersion());
            return StatusCode::INTERNAL_ERROR;
        }
        const std::string* outputNameFromCapiTensor = nullptr;
        InferenceTensor* outputTensorFromResponse{nullptr};
        status = response->getOutput(outputId, &outputNameFromCapiTensor, &outputTensorFromResponse);
        if (!status.ok()) {
            SPDLOG_ERROR("Cannot serialize output with name:{} for servable name:{}; version:{}; error: cannot find inserted input",
                outputName, response->getServableName(), response->getServableVersion());
            return StatusCode::INTERNAL_ERROR;
        }
        const InferenceTensor* outputTensorFromRequest{nullptr};
        status = request->getOutput(outputInfo->getMappedName().c_str(), &outputTensorFromRequest);
        bool copyBuffer = true;
        const void* bufferAddr{nullptr};
        OVMS_BufferType bufferType;
        if (!status.ok()) {
            OV_LOGGER("ov::Tensor: {}, tensor.data(): {}", (void*)&tensor, tensor.data());
            bufferAddr = tensor.data();
            bufferType = OVMS_BUFFERTYPE_CPU;
        } else {  // output is in request
            SPDLOG_TRACE("Will serialize output with name:{} for servable name:{}; version:{} with buffer from request",
                outputName, response->getServableName(), response->getServableVersion());
            copyBuffer = false;
            const Buffer* requestOutputBuffer = outputTensorFromRequest->getBuffer();
            if (!requestOutputBuffer) {  // this should be rejected in validation
                SPDLOG_ERROR("Cannot serialize output with name:{} for servable name:{}; version:{}; error: cannot find inserted output",
                    outputName, response->getServableName(), response->getServableVersion());
                return Status(StatusCode::INTERNAL_ERROR, "tried to use tensor with no buffer!");
            }
            bufferType = requestOutputBuffer->getBufferType();
            bufferAddr = requestOutputBuffer->data();
        }
        OV_LOGGER("ov::Tensor: {}, tensor.get_byt_size()", (void*)&tensor);
        outputTensorFromResponse->setBuffer(
            bufferAddr,
            tensor.get_byte_size(),  // here we pass the actual content bytesize not the original buffer size passed in request TODO TBD
            bufferType,
            std::nullopt,  // TODO TBD
            copyBuffer);
        SPDLOG_TRACE("Serialized output with name:{}; for servable name:{}; version:{}; with buffer copy:{}",
            outputName, response->getServableName(), response->getServableVersion(), copyBuffer);
        ++outputId;
    }
    return StatusCode::OK;
}
//template class OutputGetter<ov::InferRequest&>;
}  // namespace ovms
