//*****************************************************************************
// Copyright 2020-2022 Intel Corporation
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

#include "capi_frontend/inferencerequest.hpp"
#include "capi_frontend/inferencetensor.hpp"
#include "kfs_frontend/kfs_utils.hpp"
#include "profiler.hpp"
#include "status.hpp"
#include "tensor_conversion.hpp"
#include "tensorinfo.hpp"
#include "tfs_frontend/tfs_utils.hpp"

namespace ovms {

#define RETURN_IF_ERR(X)   \
    {                      \
        auto status = (X); \
        if (!status.ok())  \
            return status; \
    }

ov::Tensor makeTensor(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo);

ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo,
    const std::string& buffer);
ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo);

ov::Tensor makeTensor(const InferenceTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo);

class ConcreteTensorProtoDeserializator {
public:
    static ov::Tensor deserializeTensorProto(
        const ::KFSRequest::InferInputTensor& requestInput,
        const std::shared_ptr<const TensorInfo>& tensorInfo,
        const std::string* buffer) {
        OVMS_PROFILE_FUNCTION();
        if (nullptr != buffer) {
            switch (tensorInfo->getPrecision()) {
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
            case ovms::Precision::BOOL:
            case ovms::Precision::U8: {
                return makeTensor(requestInput, tensorInfo, *buffer);
            }
            case ovms::Precision::U1:
            case ovms::Precision::CUSTOM:
            case ovms::Precision::UNDEFINED:
            case ovms::Precision::DYNAMIC:
            case ovms::Precision::MIXED:
            case ovms::Precision::Q78:
            default:
                return ov::Tensor();
            }
        } else {
            switch (tensorInfo->getPrecision()) {
                // bool_contents
            case ovms::Precision::BOOL: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                bool* ptr = reinterpret_cast<bool*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().bool_contents()) {
                    ptr[i++] = *(const_cast<bool*>(reinterpret_cast<const bool*>(&number)));
                }
                return tensor;
                break;
            }
                /// int_contents
            case ovms::Precision::I8: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                int8_t* ptr = reinterpret_cast<int8_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().int_contents()) {
                    ptr[i++] = *(const_cast<int8_t*>(reinterpret_cast<const int8_t*>(&number)));
                }
                return tensor;
                break;
            }
            case ovms::Precision::I16: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                int16_t* ptr = reinterpret_cast<int16_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().int_contents()) {
                    ptr[i++] = *(const_cast<int16_t*>(reinterpret_cast<const int16_t*>(&number)));
                }
                return tensor;
                break;
            }
            case ovms::Precision::I32: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                int32_t* ptr = reinterpret_cast<int32_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().int_contents()) {
                    ptr[i++] = *(const_cast<int32_t*>(reinterpret_cast<const int32_t*>(&number)));
                }
                return tensor;
                break;
            }
                /// int64_contents
            case ovms::Precision::I64: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                int64_t* ptr = reinterpret_cast<int64_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().int64_contents()) {
                    ptr[i++] = *(const_cast<int64_t*>(reinterpret_cast<const int64_t*>(&number)));
                }
                return tensor;
                break;
            }
                // uint_contents
            case ovms::Precision::U8: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                uint8_t* ptr = reinterpret_cast<uint8_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().uint_contents()) {
                    ptr[i++] = *(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&number)));
                }
                return tensor;
                break;
            }
            case ovms::Precision::U16: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                uint16_t* ptr = reinterpret_cast<uint16_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().uint_contents()) {
                    ptr[i++] = *(const_cast<uint16_t*>(reinterpret_cast<const uint16_t*>(&number)));
                }
                return tensor;
                break;
            }
            case ovms::Precision::U32: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                uint32_t* ptr = reinterpret_cast<uint32_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().uint_contents()) {
                    ptr[i++] = *(const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(&number)));
                }
                return tensor;
                break;
            }
                // uint64_contents
            case ovms::Precision::U64: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                uint64_t* ptr = reinterpret_cast<uint64_t*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().uint64_contents()) {
                    ptr[i++] = *(const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(&number)));
                }
                return tensor;
                break;
            }
                // fp32_contents
            case ovms::Precision::FP32: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                float* ptr = reinterpret_cast<float*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().fp32_contents()) {
                    ptr[i++] = *(const_cast<float*>(reinterpret_cast<const float*>(&number)));
                }
                return tensor;
                break;
            }
                // fp64_contentes
            case ovms::Precision::FP64: {
                ov::Tensor tensor = makeTensor(requestInput, tensorInfo);
                double* ptr = reinterpret_cast<double*>(tensor.data());
                size_t i = 0;
                for (auto& number : requestInput.contents().fp64_contents()) {
                    ptr[i++] = *(const_cast<double*>(reinterpret_cast<const double*>(&number)));
                }
                return tensor;
                break;
            }
            case ovms::Precision::FP16:
            case ovms::Precision::U1:
            case ovms::Precision::CUSTOM:
            case ovms::Precision::UNDEFINED:
            case ovms::Precision::DYNAMIC:
            case ovms::Precision::MIXED:
            case ovms::Precision::Q78:
            case ovms::Precision::BIN:
            default:
                return ov::Tensor();
            }
        }
    }

    static ov::Tensor deserializeTensorProto(
        const InferenceTensor& requestInput,
        const std::shared_ptr<const TensorInfo>& tensorInfo) {
        OVMS_PROFILE_FUNCTION();
        switch (tensorInfo->getPrecision()) {
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
        case ovms::Precision::BOOL:
        case ovms::Precision::U1:
        case ovms::Precision::U8: {
            return makeTensor(requestInput, tensorInfo);
        }
        case ovms::Precision::CUSTOM:
        case ovms::Precision::UNDEFINED:
        case ovms::Precision::DYNAMIC:
        case ovms::Precision::MIXED:
        case ovms::Precision::Q78:
        case ovms::Precision::BIN:
        default:
            return ov::Tensor();
        }
    }
    static ov::Tensor deserializeTensorProto(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<const TensorInfo>& tensorInfo) {
        OVMS_PROFILE_FUNCTION();
        switch (tensorInfo->getPrecision()) {
        case ovms::Precision::FP32:
        case ovms::Precision::I32:
        case ovms::Precision::FP64:
        case ovms::Precision::I64:
        case ovms::Precision::U8:
        case ovms::Precision::I16:
        case ovms::Precision::I8: {
            return makeTensor(requestInput, tensorInfo);
        }
        case ovms::Precision::FP16: {
            ov::Shape shape;
            for (std::int64_t i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
                shape.push_back(requestInput.tensor_shape().dim(i).size());
            }
            ov::Tensor tensor(ov::element::f16, shape);
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
            ov::Shape shape;
            for (std::int64_t i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
                shape.push_back(requestInput.tensor_shape().dim(i).size());
            }
            ov::Tensor tensor(ov::element::u16, shape);
            // Needs conversion due to zero padding for each value:
            // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
            uint16_t* ptr = (uint16_t*)tensor.data();
            auto size = static_cast<size_t>(requestInput.int_val_size());
            for (size_t i = 0; i < size; i++) {
                ptr[i] = requestInput.int_val(i);
            }
            return tensor;
        }
        case ovms::Precision::U32:
        case ovms::Precision::U64:
        default:
            return ov::Tensor();
        }
    }
};

template <class TensorProtoDeserializator>
ov::Tensor deserializeTensorProto(
    const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo) {
    return TensorProtoDeserializator::deserializeTensorProto(requestInput, tensorInfo);
}

template <class TensorProtoDeserializator>
ov::Tensor deserializeTensorProto(
    const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo,
    const std::string* buffer) {
    return TensorProtoDeserializator::deserializeTensorProto(requestInput, tensorInfo, buffer);
}

template <class TensorProtoDeserializator>
ov::Tensor deserializeTensorProto(
    const InferenceTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo) {
    return TensorProtoDeserializator::deserializeTensorProto(requestInput, tensorInfo);
}

template <class Requester>
class InputSink {
    Requester requester;

public:
    InputSink(Requester requester) :
        requester(requester) {}
    Status give(const std::string& name, ov::Tensor& tensor);
};

template <class TensorProtoDeserializator, class Sink>
Status deserializePredictRequest(
    const tensorflow::serving::PredictRequest& request,
    const tensor_map_t& inputMap,
    Sink& inputSink, bool isPipeline) {
    OVMS_PROFILE_FUNCTION();
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
            ov::Tensor tensor;

            if (requiresPreProcessing(requestInput)) {
                switch (tensorInfo->getPreProcessingHint()) {
                case TensorInfo::ProcessingHint::STRING_1D_U8:
                    SPDLOG_DEBUG("Request contains input in 1D string format: {}", name);
                    RETURN_IF_ERR(convertStringRequestToOVTensor1D(requestInput, tensor, nullptr));
                    break;
                case TensorInfo::ProcessingHint::STRING_2D_U8:
                    SPDLOG_DEBUG("Request contains input in 2D string format: {}", name);
                    RETURN_IF_ERR(convertStringRequestToOVTensor2D(requestInput, tensor, nullptr));
                    break;
                case TensorInfo::ProcessingHint::IMAGE:
                    SPDLOG_DEBUG("Request contains input in native file format: {}", name);
                    RETURN_IF_ERR(convertNativeFileFormatRequestTensorToOVTensor(requestInput, tensor, tensorInfo, nullptr));
                    break;
                default:
                    SPDLOG_DEBUG("Request input: {} requires conversion but endpoint specifies no processing hint. Number of dimensions: {}; precision: {}; demultiplexer: {}",
                        name, tensorInfo->getShape().size(), toString(tensorInfo->getPrecision()), tensorInfo->isInfluencedByDemultiplexer());
                    return StatusCode::NOT_IMPLEMENTED;
                }
            } else {
                // Data Array Format
                tensor = deserializeTensorProto<TensorProtoDeserializator>(
                    requestInput, tensorInfo);
            }

            if (!tensor) {
                status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION;
                SPDLOG_DEBUG(status.string());
                return status;
            }
            const std::string ovTensorName = isPipeline ? name : tensorInfo->getName();
            status = inputSink.give(ovTensorName, tensor);
            if (!status.ok()) {
                SPDLOG_DEBUG("Feeding input:{} to inference performer failed:{}", ovTensorName, status.string());
                return status;
            }
            // OV implementation the ov::Exception is not
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

template <class TensorProtoDeserializator, class Sink>
Status deserializePredictRequest(
    const ::KFSRequest& request,
    const tensor_map_t& inputMap,
    Sink& inputSink, bool isPipeline) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    bool deserializeFromSharedInputContents = request.raw_input_contents().size() > 0;
    for (const auto& pair : inputMap) {
        try {
            const auto& name = pair.first;
            auto tensorInfo = pair.second;
            auto requestInputItr = std::find_if(request.inputs().begin(), request.inputs().end(), [&name](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == name; });
            if (requestInputItr == request.inputs().end()) {
                SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
                return Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request");
            }
            ov::Tensor tensor;

            auto inputIndex = requestInputItr - request.inputs().begin();
            auto bufferLocation = deserializeFromSharedInputContents ? &request.raw_input_contents()[inputIndex] : nullptr;

            if (requiresPreProcessing(*requestInputItr)) {
                switch (tensorInfo->getPreProcessingHint()) {
                case TensorInfo::ProcessingHint::STRING_1D_U8:
                    SPDLOG_DEBUG("Request contains input in 1D string format: {}", name);
                    RETURN_IF_ERR(convertStringRequestToOVTensor1D(*requestInputItr, tensor, bufferLocation));
                    break;
                case TensorInfo::ProcessingHint::STRING_2D_U8:
                    SPDLOG_DEBUG("Request contains input in 2D string format: {}", name);
                    RETURN_IF_ERR(convertStringRequestToOVTensor2D(*requestInputItr, tensor, bufferLocation));
                    break;
                case TensorInfo::ProcessingHint::IMAGE:
                    SPDLOG_DEBUG("Request contains input in native file format: {}", name);
                    RETURN_IF_ERR(convertNativeFileFormatRequestTensorToOVTensor(*requestInputItr, tensor, tensorInfo, bufferLocation));
                    break;
                default:
                    SPDLOG_DEBUG("Request input: {} requires conversion but endpoint specifies no processing hint. Number of dimensions: {}; precision: {}; demultiplexer: {}",
                        name, tensorInfo->getShape().size(), toString(tensorInfo->getPrecision()), tensorInfo->isInfluencedByDemultiplexer());
                    return StatusCode::NOT_IMPLEMENTED;
                }
            } else {
                tensor = deserializeTensorProto<TensorProtoDeserializator>(*requestInputItr, tensorInfo, bufferLocation);
                if (!tensor) {
                    status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION;
                    SPDLOG_DEBUG(status.string());
                    return status;
                }
            }

            const std::string ovTensorName = isPipeline ? name : tensorInfo->getName();
            status = inputSink.give(ovTensorName, tensor);
            if (!status.ok()) {
                SPDLOG_DEBUG("Feeding input:{} to inference performer failed:{}", ovTensorName, status.string());
                return status;
            }
            // OV implementation the ov::Exception is not
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
template <class TensorProtoDeserializator, class Sink>
Status deserializePredictRequest(
    const InferenceRequest& request,
    const tensor_map_t& inputMap,
    Sink& inputSink, bool isPipeline) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    for (const auto& [name, tensorInfo] : inputMap) {
        try {
            const InferenceTensor* requestInputPtr{nullptr};
            auto status = request.getInput(name.c_str(), &requestInputPtr);
            if (!status.ok() || requestInputPtr == nullptr) {
                SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
                return Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request");
            }
            ov::Tensor tensor;
            // binary input handling
            /* if (requestInputPtr->getDataType() == "BYTES") {
                SPDLOG_DEBUG("Request contains binary input: {}", name);
                return StatusCode::NOT_IMPLEMENTED;
            } else { */
            tensor = deserializeTensorProto<TensorProtoDeserializator>(*requestInputPtr, tensorInfo);
            if (!tensor) {
                status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION;
                SPDLOG_DEBUG(status.string());
                return status;
            }

            const std::string ovTensorName = isPipeline ? name : tensorInfo->getName();
            status = inputSink.give(ovTensorName, tensor);
            if (!status.ok()) {
                SPDLOG_DEBUG("Feeding input:{} to inference performer failed:{}", ovTensorName, status.string());
                return status;
            }
            // OV implementation the ov::Exception is not
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
