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

//#include "capi_utils.hpp"
#include "../deserialization_common.hpp"
#include "capi_request_utils.hpp"
#include "inferencerequest.hpp"
#include "inferencetensor.hpp"
#include "../logging.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../tensorinfo.hpp"

namespace ovms {
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

template class ConcreteTensorProtoDeserializator<InferenceTensor>;
}  // namespace ovms
