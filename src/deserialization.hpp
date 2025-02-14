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
#include <tuple>
#include <unordered_map>

#include <openvino/openvino.hpp>

#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "capi_frontend/capi_utils.hpp"
#include "capi_frontend/inferencerequest.hpp"
#include "capi_frontend/inferencetensor.hpp"
#include "kfs_frontend/kfs_utils.hpp"
#include "logging.hpp"
#include "prediction_service_utils.hpp"
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

class IOVTensorFactory;

template <typename Request>
struct RequestTraits {
    using TensorType = void;
};

template <typename RequestTensorType>
class ConcreteTensorProtoDeserializator {
public:
    static ov::Tensor deserializeTensor(
        const RequestTensorType& requestInput,
        const std::shared_ptr<const TensorInfo>& tensorInfo,
        const std::unordered_map<int, std::shared_ptr<IOVTensorFactory>>& factories,
        const std::string* bufferLocation) {
        static_assert(!std::is_same<RequestTensorType, void>::value,
            "Tried to deserialize of yet unsupported frontend");
        return ov::Tensor();
    }
};

template <class Requester>
class InputSink {
    Requester requester;

public:
    InputSink(Requester requester) :
        requester(requester) {}
    Status give(const std::string& name, ov::Tensor& tensor);
};

template <typename RequestType>
static std::tuple<ovms::Status, const typename RequestTraits<RequestType>::TensorType*, const std::string*> getRequestTensorPtr(const RequestType& request, const std::string& name, ExtractChoice extractChoice) {
    static_assert(!std::is_same<typename RequestTraits<RequestType>::TensorType, void>::value,
        "RequestType is not supported. Please provide a specialization for RequestTraits with getRequestTensorPtr.");
    return std::make_tuple(Status(StatusCode::NOT_IMPLEMENTED, "Failed to deserialize request"),
        nullptr, nullptr);
}

template <typename Request, typename Tensor>
Status getTensor(const Request& request, const std::string& name, const Tensor tensor);

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
//////
//
// Move to kfs
//
//////
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

//////
//
// Move to capi
//
//////
ov::Tensor makeTensor(const InferenceTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo, const std::unordered_map<int, std::shared_ptr<IOVTensorFactory>>& factories);

template <>
struct RequestTraits<ovms::InferenceRequest> {
    using TensorType = ovms::InferenceTensor;
};

template <>
class ConcreteTensorProtoDeserializator<InferenceTensor> {
public:
    static ov::Tensor deserializeTensor(
        const InferenceTensor& requestInput,
        const std::shared_ptr<const TensorInfo>& tensorInfo,
        const std::unordered_map<int, std::shared_ptr<IOVTensorFactory>>& factories,
        const std::string*) {
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
        case ovms::Precision::STRING:
        case ovms::Precision::U1:
        case ovms::Precision::U8: {
            return makeTensor(requestInput, tensorInfo, factories);
        }
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
};

// due to header included in many places function below is not used in all cpp files ...
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>  // TODO separate for different choice
std::tuple<ovms::Status, const typename RequestTraits<ovms::InferenceRequest>::TensorType*, const std::string*> getRequestTensorPtr(const ovms::InferenceRequest& request, const std::string& name, ExtractChoice extractChoice) {
    const InferenceTensor* requestTensorPtr{nullptr};
    ovms::Status status;
    switch (extractChoice) {
    case ExtractChoice::EXTRACT_INPUT: {
        status = RequestTensorExtractor<InferenceRequest, InferenceTensor, ExtractChoice::EXTRACT_INPUT>::extract(request, name, &requestTensorPtr);
        break;
    }
    case ExtractChoice::EXTRACT_OUTPUT: {
        status = RequestTensorExtractor<InferenceRequest, InferenceTensor, ExtractChoice::EXTRACT_OUTPUT>::extract(request, name, &requestTensorPtr);
        break;
    }
    }
    if ((!status.ok() || requestTensorPtr == nullptr)) {
        if (extractChoice == ExtractChoice::EXTRACT_INPUT) {
            SPDLOG_DEBUG("Failed to deserialize request. Validation of request failed");
        }
        return std::make_tuple(Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request"), nullptr, nullptr);
    }
    return std::make_tuple(Status(StatusCode::OK), requestTensorPtr, nullptr);
}
#pragma GCC diagnostic pop

#define RETURN_IF_EMPTY_TENSOR()                                           \
    do {                                                                   \
        if (!tensor) {                                                     \
            status = StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION; \
            SPDLOG_DEBUG("{}", status.string());                           \
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

// OV implementation the ov::Exception is not
// a base class for all other exceptions thrown from OV.
// OV can throw exceptions derived from std::logic_error.

#define CATCH_AND_RETURN_ERROR()                                \
    catch (const ov::Exception& e) {                            \
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR; \
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());      \
        return status;                                          \
    }                                                           \
    catch (std::logic_error & e) {                              \
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR; \
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());      \
        return status;                                          \
    }

template <template <typename> class TensorDeserializator, class Sink, class RequestType>
static Status deserializePredictRequest(
    const RequestType& request,
    const tensor_map_t& inputMap,
    const tensor_map_t& outputMap,
    Sink& tensorSink, bool isPipeline, const std::unordered_map<int, std::shared_ptr<IOVTensorFactory>>& factories) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    ov::Tensor tensor;
    for (const auto& [name, tensorInfo] : inputMap) {
        try {
            auto [status, requestInputItr, bufferLocation] = getRequestTensorPtr(request, name, ExtractChoice::EXTRACT_INPUT);
            if (!status.ok() || !requestInputItr) {
                SPDLOG_ERROR("Failed to deserialize request. Validation of request failed");
                return Status(StatusCode::INTERNAL_ERROR, "Failed to deserialize request");
            }
            // TODO move preprocessing to deserializeTensor
            if (requiresPreProcessing(*requestInputItr)) {
                switch (tensorInfo->getPreProcessingHint()) {
                case TensorInfo::ProcessingHint::STRING_NATIVE:
                    SPDLOG_DEBUG("Request contains input in native string format: {}", name);
                    RETURN_IF_ERR(convertStringRequestToOVTensor(*requestInputItr, tensor, bufferLocation));
                    break;
                case TensorInfo::ProcessingHint::STRING_2D_U8:
                    SPDLOG_DEBUG("Request contains input in 2D string format: {}", name);
                    RETURN_IF_ERR(convertStringRequestToOVTensor2D(*requestInputItr, tensor, bufferLocation));
                    break;
                case TensorInfo::ProcessingHint::IMAGE:
                    SPDLOG_DEBUG("Request contains input in native file format: {}", name);
                    RETURN_IF_ERR(convertNativeFileFormatRequestTensorToOVTensor(*requestInputItr, tensor, *tensorInfo, bufferLocation));
                    break;
                default:
                    SPDLOG_DEBUG("Request input: {} requires conversion but endpoint specifies no processing hint. Number of dimensions: {}; precision: {}; demultiplexer: {}",
                        name, tensorInfo->getShape().size(), toString(tensorInfo->getPrecision()), tensorInfo->isInfluencedByDemultiplexer());
                    return StatusCode::NOT_IMPLEMENTED;
                }
            } else {
                using TensorType = typename RequestTraits<RequestType>::TensorType;
                tensor = TensorDeserializator<TensorType>::deserializeTensor(*requestInputItr, tensorInfo, factories, bufferLocation);
            }
            RETURN_IF_EMPTY_TENSOR();
            const std::string ovTensorName = isPipeline ? name : tensorInfo->getName();
            status = tensorSink.give(ovTensorName, tensor);
            RETURN_IF_NOT_OK("Feeding input:{} to inference performer failed:{}", ovTensorName, status.string());
        }
        CATCH_AND_RETURN_ERROR();
    }
    for (const auto& [name, tensorInfo] : outputMap) {
        try {
            auto [status, requestInputItr, bufferLocation] = getRequestTensorPtr(request, name, ExtractChoice::EXTRACT_OUTPUT);
            if (!status.ok() || !requestInputItr) {
                // TODO impose limits on what can be processed in deserialization on output eg. no binary handling
                SPDLOG_TRACE("Skipping output name:{}", name);
                // TODO possibly we could have passed here filtered output map
                // instead of searching for each output and skipping
                continue;
            }
            using TensorType = typename RequestTraits<RequestType>::TensorType;
            tensor = TensorDeserializator<TensorType>::deserializeTensor(*requestInputItr, tensorInfo, factories, nullptr);
            RETURN_IF_EMPTY_TENSOR();
            const std::string ovTensorName = isPipeline ? name : tensorInfo->getName();
            status = tensorSink.give(ovTensorName, tensor);
            RETURN_IF_NOT_OK("Feeding input:{} to inference performer failed:{}", ovTensorName, status.string());
        }
        CATCH_AND_RETURN_ERROR();
    }
    return status;
}
template class ConcreteTensorProtoDeserializator<InferenceTensor>;
}  // namespace ovms
