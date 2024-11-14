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

#include "extractchoice.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"
#include "tensor_conversion.hpp"
#include "deserialization_common.hpp"

namespace ovms {

#define RETURN_IF_ERR(X)   \
    {                      \
        auto status = (X); \
        if (!status.ok())  \
            return status; \
    }

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

class IOVTensorFactory;
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
}  // namespace ovms
