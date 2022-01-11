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

#include "binaryutils.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

ov::runtime::Tensor makeBlob_2(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo);

class ConcreteTensorProtoDeserializator_2 {
public:
    static ov::runtime::Tensor deserializeTensorProto_2(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<TensorInfo>& tensorInfo) {
        switch (tensorInfo->getPrecision()) {
        case ovms::Precision::FP32:
        case ovms::Precision::I32:
        case ovms::Precision::U8:
        case ovms::Precision::I16:
            return makeBlob_2(requestInput, tensorInfo);
        case ovms::Precision::I8: {
            return makeBlob_2(requestInput, tensorInfo);
        }
        case ovms::Precision::FP16: {
            ov::Shape shape;
            for (std::int64_t i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
                shape.push_back(requestInput.tensor_shape().dim(i).size());
            }
            ov::runtime::Tensor tensor(ov::element::f16, shape);
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
            ov::runtime::Tensor tensor(ov::element::u16, shape);
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
ov::runtime::Tensor deserializeTensorProto_2(
    const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo) {
    return TensorProtoDeserializator::deserializeTensorProto_2(requestInput, tensorInfo);
}

template <class Requester>
class InputSink_2 {
    Requester requester;

public:
    InputSink_2(Requester requester) :
        requester(requester) {}
    Status give(const std::string& name, ov::runtime::Tensor& blob);  // TODO replace with one below
    Status give(const std::string& name, std::shared_ptr<ov::runtime::Tensor>& tensor);
};

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
                    requestInput, tensorInfo);
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
