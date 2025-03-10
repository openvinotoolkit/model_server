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

#include "../deserialization_common.hpp"
#include "kfs_utils.hpp"
#include "../logging.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../tensor_conversion.hpp"
#include "../tensorinfo.hpp"

namespace ovms {
ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo,
    const std::string& buffer);
ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo);

template <>
struct RequestTraits<::KFSRequest> {
    using TensorType = ::KFSRequest::InferInputTensor;
};

template <>
class ConcreteTensorProtoDeserializator<::KFSRequest::InferInputTensor> {
public:
    static ov::Tensor deserializeTensor(
        const ::KFSRequest::InferInputTensor& requestInput,
        const std::shared_ptr<const TensorInfo>& tensorInfo,
        const std::unordered_map<int, std::shared_ptr<IOVTensorFactory>>& factories,
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
#define COPY_EACH_VALUE(OVMS_ENUM_TYPE, C_TYPE, CONTENT_PREFIX)                          \
    case ovms::Precision::OVMS_ENUM_TYPE: {                                              \
        ov::Tensor tensor = makeTensor(requestInput, tensorInfo);                        \
        C_TYPE* ptr = reinterpret_cast<C_TYPE*>(tensor.data());                          \
        size_t i = 0;                                                                    \
        for (auto& number : requestInput.contents().CONTENT_PREFIX()) {                  \
            ptr[i++] = *(const_cast<C_TYPE*>(reinterpret_cast<const C_TYPE*>(&number))); \
        }                                                                                \
        return tensor;                                                                   \
        break;                                                                           \
    }
                COPY_EACH_VALUE(BOOL, bool, bool_contents)
                COPY_EACH_VALUE(I8, int8_t, int_contents)
                COPY_EACH_VALUE(I16, int16_t, int_contents)
                COPY_EACH_VALUE(I32, int32_t, int_contents)
                COPY_EACH_VALUE(I64, int64_t, int64_contents)
                COPY_EACH_VALUE(U8, uint8_t, uint_contents)
                COPY_EACH_VALUE(U16, uint16_t, uint_contents)
                COPY_EACH_VALUE(U32, uint32_t, uint_contents)
                COPY_EACH_VALUE(U64, uint64_t, uint64_contents)
                COPY_EACH_VALUE(FP32, float, fp32_contents)
                COPY_EACH_VALUE(FP64, double, fp64_contents)
            case ovms::Precision::FP16:
            case ovms::Precision::U1:
            case ovms::Precision::CUSTOM:
            case ovms::Precision::UNDEFINED:
            case ovms::Precision::DYNAMIC:
            case ovms::Precision::MIXED:
            case ovms::Precision::Q78:
            case ovms::Precision::BIN:
            default:
                OV_LOGGER("ov::Tensor()");
                return ov::Tensor();
            }
        }
    }
};

// due to header included in many places function below is not used in all cpp files ...
#pragma warning(push)
#pragma warning(disable : 4505)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>  // TODO separate for different choice
std::tuple<ovms::Status, const typename RequestTraits<::KFSRequest>::TensorType*, const std::string*> getRequestTensorPtr(const ::KFSRequest& request, const std::string& name, ExtractChoice extractChoice) {
    if (ExtractChoice::EXTRACT_OUTPUT == extractChoice) {
        SPDLOG_TRACE("Defining output in KFS is not implemented");
        return std::make_tuple(Status(StatusCode::NOT_IMPLEMENTED, "Failed to deserialize output in request"), nullptr, nullptr);
    }
    bool deserializeFromSharedInputContents = request.raw_input_contents().size() > 0;
    auto requestInputItr = std::find_if(request.inputs().begin(), request.inputs().end(), [&name](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == name; });
    if (requestInputItr == request.inputs().end()) {
        SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
        return std::make_tuple(Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request"), nullptr, nullptr);
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto bufferLocation = deserializeFromSharedInputContents ? &request.raw_input_contents()[inputIndex] : nullptr;
    return std::make_tuple(Status(StatusCode::OK), &*requestInputItr, bufferLocation);
}
#pragma GCC diagnostic pop
#pragma warning(pop)

#define RETURN_IF_EMPTY_TENSOR()                                           \
    do {                                                                   \
        if (!tensor) {                                                     \
            status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION; \
            SPDLOG_DEBUG(status.string());                                 \
            return status;                                                 \
        }                                                                  \
    } while (0)

#define RETURN_IF_NOT_OK(fmt, ...)            \
    do {                                      \
        if (!status.ok()) {                   \
            SPDLOG_DEBUG(fmt, ##__VA_ARGS__); \
            return status;                    \
        }                                     \
    } while (0)

}  // namespace ovms
