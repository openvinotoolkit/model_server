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
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

template <typename T>
InferenceEngine::Blob::Ptr makeBlob(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo) {
    return InferenceEngine::make_shared_blob<T>(
        tensorInfo->getTensorDesc(),
        const_cast<T*>(reinterpret_cast<const T*>(requestInput.tensor_content().data())));
}

class ConcreteTensorProtoDeserializator {
public:
    static InferenceEngine::Blob::Ptr deserializeTensorProto(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<TensorInfo>& tensorInfo) {
        switch (tensorInfo->getPrecision()) {
        case InferenceEngine::Precision::FP32:
            return makeBlob<float>(requestInput, tensorInfo);
        case InferenceEngine::Precision::FP16: {
            auto blob = InferenceEngine::make_shared_blob<uint16_t>(tensorInfo->getTensorDesc());
            blob->allocate();
            // Needs conversion due to zero padding for each value:
            // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
            uint16_t* ptr = blob->buffer().as<uint16_t*>();
            auto size = static_cast<size_t>(requestInput.half_val_size());
            for (size_t i = 0; i < size; i++) {
                ptr[i] = requestInput.half_val(i);
            }
            return blob;
        }
        case InferenceEngine::Precision::U8:
            return makeBlob<uint8_t>(requestInput, tensorInfo);
        case InferenceEngine::Precision::I8:
            return makeBlob<int8_t>(requestInput, tensorInfo);
        case InferenceEngine::Precision::U16: {
            auto blob = InferenceEngine::make_shared_blob<uint16_t>(tensorInfo->getTensorDesc());
            blob->allocate();
            // Needs conversion due to zero padding for each value:
            // https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/core/framework/tensor.proto#L55
            uint16_t* ptr = blob->buffer().as<uint16_t*>();
            auto size = static_cast<size_t>(requestInput.int_val_size());
            for (size_t i = 0; i < size; i++) {
                ptr[i] = requestInput.int_val(i);
            }
            return blob;
        }
        case InferenceEngine::Precision::I16:
            return makeBlob<int16_t>(requestInput, tensorInfo);
        case InferenceEngine::Precision::I32:
            return makeBlob<int32_t>(requestInput, tensorInfo);
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

template <class TensorProtoDeserializator>
InferenceEngine::Blob::Ptr deserializeTensorProto(
    const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo) {
    return TensorProtoDeserializator::deserializeTensorProto(requestInput, tensorInfo);
}

template <class TensorProtoDeserializator>
Status deserializePredictRequest(
    const tensorflow::serving::PredictRequest& request,
    const tensor_map_t& inputMap,
    InferenceEngine::InferRequest& inferRequest) {
    try {
        for (const auto& pair : inputMap) {
            const auto& name = pair.first;
            auto tensorInfo = pair.second;
            auto requestInputItr = request.inputs().find(name);
            if (requestInputItr == request.inputs().end()) {
                SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
                return Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request");
            }
            auto& requestInput = requestInputItr->second;

            InferenceEngine::Blob::Ptr blob =
                deserializeTensorProto<TensorProtoDeserializator>(
                    requestInput, tensorInfo);

            if (blob == nullptr) {
                Status status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION;
                SPDLOG_DEBUG(status.string());
                return status;
            }
            inferRequest.SetBlob(tensorInfo->getName(), blob);
        }
        // OV implementation the InferenceEngineException is not
        // a base class for all other exceptions thrown from OV.
        // OV can throw exceptions derived from std::logic_error.
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    }

    return StatusCode::OK;
}
}  // namespace ovms
