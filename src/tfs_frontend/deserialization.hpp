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

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wall"

#pragma warning(push)
#pragma warning(disable : 6269 6294 6201 4624 6385 6386 6011 6001 6326)
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma warning(pop)
#pragma GCC diagnostic pop

#include "../deserialization_common.hpp"
#include "../logging.hpp"
//#include "../prediction_service_utils.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../tensor_conversion.hpp"
#include "../tensorinfo.hpp"
#include "tfs_utils.hpp"

namespace ovms {

//////
//
// Move to tfs
//
//////
ov::Tensor makeTensor(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo);

template <>
struct RequestTraits<::TFSPredictRequest> {
    using TensorType = ::tensorflow::TensorProto;
};

template <>
class ConcreteTensorProtoDeserializator<tensorflow::TensorProto> {
public:
    static ov::Tensor deserializeTensor(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<const TensorInfo>& tensorInfo,
        const std::unordered_map<int, std::shared_ptr<IOVTensorFactory>>& factories,
        const std::string* bufferLocation) {
        OVMS_PROFILE_FUNCTION();
        switch (tensorInfo->getPrecision()) {
        case ovms::Precision::FP32:
        case ovms::Precision::U32:
        case ovms::Precision::I32:
        case ovms::Precision::FP64:
        case ovms::Precision::I64:
        case ovms::Precision::U8:
        case ovms::Precision::I16:
        case ovms::Precision::I8: {
            return makeTensor(requestInput, tensorInfo);
        }
#pragma warning(push)
#pragma warning(disable : 4244 4267)
        case ovms::Precision::FP16: {
            OV_LOGGER("ov::Shape()");
            ov::Shape shape;
            for (std::int64_t i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
                OV_LOGGER("ov::Shape::push_back({})", requestInput.tensor_shape().dim(i).size());
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
#pragma warning(pop)
        case ovms::Precision::U64:
        default:
            return ov::Tensor();
        }
    }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>  // TODO separate for different choice
std::tuple<ovms::Status, const typename RequestTraits<::TFSPredictRequest>::TensorType*, const std::string*> getRequestTensorPtr(const ::TFSPredictRequest& request, const std::string& name, ExtractChoice extractChoice) {
    if (ExtractChoice::EXTRACT_OUTPUT == extractChoice) {
        SPDLOG_TRACE("Defining output in TFS is not implemented");
        return std::make_tuple(Status(StatusCode::NOT_IMPLEMENTED, "Failed to deserialize output in request"), nullptr, nullptr);
    }
    auto requestInputItr = request.inputs().find(name);
    if (requestInputItr == request.inputs().end()) {
        SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
        return std::make_tuple(Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize input in request"), nullptr, nullptr);
    }
    return std::make_tuple(Status(StatusCode::OK), &requestInputItr->second, nullptr);
}
#pragma GCC diagnostic pop
}  // namespace ovms
