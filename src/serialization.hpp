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
#pragma once

#include <memory>
#include <string>

#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "capi_frontend/capi_utils.hpp"
#include "capi_frontend/inferenceresponse.hpp"
#include "capi_frontend/inferencetensor.hpp"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "profiler.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

template <typename T>
class OutputGetter {
public:
    OutputGetter(T t) :
        outputSource(t) {}
    Status get(const std::string& name, ov::Tensor& tensor);

private:
    T outputSource;
};

template <typename ProtoStorage, typename ProtoType>
class ProtoGetter {
    ProtoStorage protoStorage;

public:
    ProtoGetter(ProtoStorage protoStorage) :
        protoStorage(protoStorage) {}
    ProtoType createOutput(const std::string& name);
    std::string* createContent(const std::string& name);
};

Status serializeTensorToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor);

Status serializeTensorToTensorProto(
    ::KFSResponse::InferOutputTensor& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor);

Status serializeTensorToTensorProtoRaw(
    ::inference::ModelInferResponse::InferOutputTensor& responseOutput,
    std::string* rawOutputContents,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor);

Status serializeTensorToTensorProto(
    InferenceTensor& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor);

typedef const std::string& (*outputNameChooser_t)(const std::string&, const TensorInfo&);
const std::string& getTensorInfoName(const std::string& first, const TensorInfo& tensorInfo);
const std::string& getOutputMapKeyName(const std::string& first, const TensorInfo& tensorInfo);

template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {  // does not apply for TFS frontend
    OVMS_PROFILE_FUNCTION();
    Status status;
    ProtoGetter<tensorflow::serving::PredictResponse*, tensorflow::TensorProto&> protoGetter(response);
    for (const auto& [outputName, outputInfo] : outputMap) {
        ov::Tensor tensor;
        status = outputGetter.get(outputNameChooser(outputName, *outputInfo), tensor);
        if (!status.ok()) {
            return status;
        }
        auto& tensorProto = protoGetter.createOutput(outputInfo->getMappedName());
        status = serializeTensorToTensorProto(tensorProto, outputInfo, tensor);
        if (!status.ok()) {
            return status;
        }
    }
    return status;
}

template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    ::KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    response->set_model_name(servableName);
    response->set_model_version(std::to_string(servableVersion));
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(response);
    for (const auto& [outputName, outputInfo] : outputMap) {
        ov::Tensor tensor;
        status = outputGetter.get(outputNameChooser(outputName, *outputInfo), tensor);
        if (!status.ok()) {
            return status;
        }
        auto& inferOutputTensor = protoGetter.createOutput(outputInfo->getMappedName());
        if (useSharedOutputContent) {
            status = serializeTensorToTensorProtoRaw(inferOutputTensor, protoGetter.createContent(outputInfo->getMappedName()), outputInfo, tensor);
        } else {
            status = serializeTensorToTensorProto(inferOutputTensor, outputInfo, tensor);
        }

        if (!status.ok()) {
            return status;
        }
    }
    return status;
}
template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    InferenceResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {  // does not apply for C-API frontend
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
        default: {
            Status status = StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION;
            SPDLOG_ERROR(status.string());
            return status;
        }
        }
        InferenceTensor* outputTensor{nullptr};
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
        status = response->getOutput(outputId, &outputNameFromCapiTensor, &outputTensor);
        ++outputId;
        if (!status.ok()) {
            SPDLOG_ERROR("Cannot serialize output with name:{} for servable name:{}; version:{}; error: cannot find inserted input",
                outputName, response->getServableName(), response->getServableVersion());
            return StatusCode::INTERNAL_ERROR;
        }
        outputTensor->setBuffer(
            tensor.data(),
            tensor.get_byte_size(),
            OVMS_BUFFERTYPE_CPU,
            std::nullopt,
            true);
    }
    return StatusCode::OK;
}

}  // namespace ovms
