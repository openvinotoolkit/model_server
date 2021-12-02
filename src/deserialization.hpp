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

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "binaryutils.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

InferenceEngine::TensorDesc getFinalTensorDesc(const ovms::TensorInfo& servableInfo, const tensorflow::TensorProto& requestInput, bool isPipeline);

template <typename T>
InferenceEngine::Blob::Ptr makeBlob(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    return InferenceEngine::make_shared_blob<T>(
        getFinalTensorDesc(*tensorInfo, requestInput, isPipeline),
        const_cast<T*>(reinterpret_cast<const T*>(requestInput.tensor_content().data())));
}

ov::runtime::Tensor makeBlob_2(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline);

class ConcreteTensorProtoDeserializator {
public:
    static InferenceEngine::Blob::Ptr deserializeTensorProto(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
        switch (tensorInfo->getPrecision()) {
        case InferenceEngine::Precision::FP32:
            return makeBlob<float>(requestInput, tensorInfo, isPipeline);
        case InferenceEngine::Precision::I32:
            return makeBlob<int32_t>(requestInput, tensorInfo, isPipeline);
        case InferenceEngine::Precision::I8:
            return makeBlob<int8_t>(requestInput, tensorInfo, isPipeline);
        case InferenceEngine::Precision::U8:
            return makeBlob<uint8_t>(requestInput, tensorInfo, isPipeline);
        case InferenceEngine::Precision::I16:
            return makeBlob<int16_t>(requestInput, tensorInfo, isPipeline);
        case InferenceEngine::Precision::FP16: {
            InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint16_t>(getFinalTensorDesc(*tensorInfo, requestInput, isPipeline));
            blob->allocate();
            // Needs conversion due to zero padding for each value:
            // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
            uint16_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
            auto size = static_cast<size_t>(requestInput.half_val_size());
            for (size_t i = 0; i < size; i++) {
                ptr[i] = requestInput.half_val(i);
            }
            return blob;
        }
        case InferenceEngine::Precision::U16: {
            InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<uint16_t>(getFinalTensorDesc(*tensorInfo, requestInput, isPipeline));
            blob->allocate();
            // Needs conversion due to zero padding for each value:
            // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
            uint16_t* ptr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
            auto size = static_cast<size_t>(requestInput.int_val_size());
            for (size_t i = 0; i < size; i++) {
                ptr[i] = requestInput.int_val(i);
            }
            return blob;
        }
        case InferenceEngine::Precision::I64:
        case InferenceEngine::Precision::MIXED:
        case InferenceEngine::Precision::Q78:
        case InferenceEngine::Precision::BIN:
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::CUSTOM:
        default:
            return nullptr;
        }
    }
};

class ConcreteTensorProtoDeserializator_2 {
public:
    static ov::runtime::Tensor deserializeTensorProto_2(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
        switch (tensorInfo->getPrecision_2()) {
        case ovms::Precision::FP32:
        case ovms::Precision::I32:
        case ovms::Precision::U8:
        case ovms::Precision::I16:
            return makeBlob_2(requestInput, tensorInfo, isPipeline);
        case ovms::Precision::I8: {
            return makeBlob_2(requestInput, tensorInfo, isPipeline);
        }
        case ovms::Precision::FP16: {
            ov::runtime::Tensor tensor(ov::element::f16, tensorInfo->getShape());
            // Needs conversion due to zero padding for each value:
            // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
            uint16_t* ptr = (uint16_t*)tensor.data();
            auto size = static_cast<size_t>(requestInput.half_val_size());
            for (size_t i = 0; i < size; i++) {
                ptr[i] = requestInput.half_val(i);
            }
            return tensor;
        }
        case ovms::Precision::U16: {
            ov::runtime::Tensor tensor(ov::element::u16, tensorInfo->getShape());
            // Needs conversion due to zero padding for each value:
            // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
            uint16_t* ptr = (uint16_t*)tensor.data();
            auto size = static_cast<size_t>(requestInput.int_val_size());
            for (size_t i = 0; i < size; i++) {
                ptr[i] = requestInput.int_val(i);
            }
            return tensor;
        }
        default:
            return ov::runtime::Tensor();
        }
    }
};

template <class TensorProtoDeserializator>
InferenceEngine::Blob::Ptr deserializeTensorProto(
    const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    return TensorProtoDeserializator::deserializeTensorProto(requestInput, tensorInfo, isPipeline);
}

template <class TensorProtoDeserializator>
ov::runtime::Tensor deserializeTensorProto_2(
    const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo, bool isPipeline) {
    return TensorProtoDeserializator::deserializeTensorProto_2(requestInput, tensorInfo, isPipeline);
}

template <class Requester>
class InputSink {
    Requester requester;

public:
    InputSink(Requester requester) :
        requester(requester) {}
    Status give(const std::string& name, InferenceEngine::Blob::Ptr blob);
};

template <class Requester>
class InputSink_2 {
    Requester requester;

public:
    InputSink_2(Requester requester) :
        requester(requester) {}
    Status give(const std::string& name, ov::runtime::Tensor& blob); // TODO replace with one below
    Status give(const std::string& name, std::shared_ptr<ov::runtime::Tensor>& tensor);
};

template <class TensorProtoDeserializator, class Sink>
Status deserializePredictRequest(
    const tensorflow::serving::PredictRequest& request,
    const tensor_map_t& inputMap,
    Sink& inputSink, bool isPipeline) {
    Status status;
    for (const auto& pair : inputMap) {
        try {
            const auto& name = pair.first;
            auto tensorInfo = pair.second;
            auto requestInputItr = request.inputs().find(name);
            if (requestInputItr == request.inputs().end()) {
                SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
                return Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request");
            }
            auto& requestInput = requestInputItr->second;
            InferenceEngine::Blob::Ptr blob;

            if (requestInput.dtype() == tensorflow::DataType::DT_STRING) {
                SPDLOG_DEBUG("Request contains binary input: {}", name);
                status = convertStringValToBlob(requestInput, blob, tensorInfo, isPipeline);
                if (!status.ok()) {
                    SPDLOG_DEBUG("Binary inputs conversion failed.");
                    return status;
                }
            } else {
                blob = deserializeTensorProto<TensorProtoDeserializator>(
                    requestInput, tensorInfo, isPipeline);
            }

            if (blob == nullptr) {
                status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION;
                SPDLOG_DEBUG(status.string());
                return status;
            }
            const std::string ovBlobName = isPipeline ? name : tensorInfo->getName();
            status = inputSink.give(ovBlobName, blob);
            if (!status.ok()) {
                SPDLOG_DEBUG("Feeding inputs to inference performer failed:{}", status.string());
                return status;
            }
            // OV implementation the InferenceEngine::Exception is not
            // a base class for all other exceptions thrown from OV.
            // OV can throw exceptions derived from std::logic_error.
        } catch (const InferenceEngine::Exception& e) {
            status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
            SPDLOG_DEBUG("{}: {}", status.string(), e.what());
            return status;
        } catch (std::logic_error& e) {
            status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
            SPDLOG_DEBUG("{}: {}", status.string(), e.what());
            return status;
        }
    }
    return status;
}

template <class TensorProtoDeserializator, class Sink>
Status deserializePredictRequest_2(
    const tensorflow::serving::PredictRequest& request,
    const tensor_map_t& inputMap,
    Sink& inputSink, bool isPipeline) {
    Status status;
    for (const auto& pair : inputMap) {
        try {
            const auto& name = pair.first;
            auto tensorInfo = pair.second;
            auto requestInputItr = request.inputs().find(name);
            if (requestInputItr == request.inputs().end()) {
                SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
                return Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request");
            }
            auto& requestInput = requestInputItr->second;
            ov::runtime::Tensor blob;

            if (requestInput.dtype() == tensorflow::DataType::DT_STRING) {
                SPDLOG_DEBUG("Request contains binary input: {}", name);
                status = convertStringValToBlob_2(requestInput, blob, tensorInfo, isPipeline);
                if (!status.ok()) {
                    SPDLOG_DEBUG("Binary inputs conversion failed.");
                    return status;
                }
            } else {
                blob = deserializeTensorProto_2<TensorProtoDeserializator>(
                    requestInput, tensorInfo, isPipeline);
            }

            if (!blob) {
                status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION;
                SPDLOG_DEBUG(status.string());
                return status;
            }
            const std::string ovBlobName = isPipeline ? name : tensorInfo->getName();
            status = inputSink.give(ovBlobName, blob);
            if (!status.ok()) {
                SPDLOG_DEBUG("Feeding input:{} to inference performer failed:{}", ovBlobName, status.string());
                return status;
            }
            // OV implementation the InferenceEngine::Exception is not
            // a base class for all other exceptions thrown from OV.
            // OV can throw exceptions derived from std::logic_error.
        } catch (const ov::Exception& e) {
            status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
            SPDLOG_DEBUG("{}: {}", status.string(), e.what());
            return status;
        } catch (std::logic_error& e) {
            status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
            SPDLOG_DEBUG("{}: {}", status.string(), e.what());
            return status;
        }
    }
    return status;
}
}  // namespace ovms
