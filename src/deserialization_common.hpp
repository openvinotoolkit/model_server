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

namespace ovms {
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
}  // namespace ovms
