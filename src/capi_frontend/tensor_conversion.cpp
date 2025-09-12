//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "../tensor_conversion.hpp"
#include "../tensor_conversion_common.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "inferencetensor.hpp"
#include "../logging.hpp"
#include "../deps/opencv.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../tensor_conversion_after.hpp"

namespace ovms {

template <>
Status convertNativeFileFormatRequestTensorToOVTensor(const ovms::InferenceTensor& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer) {
    SPDLOG_ERROR("String conversion is not implemented for C-API");
    return StatusCode::NOT_IMPLEMENTED;
}

template <>
Status convertStringRequestToOVTensor2D(
    const ovms::InferenceTensor& src,
    ov::Tensor& tensor,
    const std::string* buffer) {
    SPDLOG_ERROR("String conversion is not implemented for C-API");
    return StatusCode::NOT_IMPLEMENTED;
}

template <>
Status convertStringRequestToOVTensor(const ovms::InferenceTensor& src, ov::Tensor& tensor, const std::string* buffer) {
    SPDLOG_ERROR("Tensor conversion is not supported for C-API");
    return StatusCode::NOT_IMPLEMENTED;
}

template <>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, ovms::InferenceTensor& dst) {
    SPDLOG_ERROR("Tensor conversion is not supported for C-API");
    return StatusCode::NOT_IMPLEMENTED;
}
template Status convertOVTensor2DToStringResponse<ovms::InferenceTensor>(const ov::Tensor& tensor, ovms::InferenceTensor& dst);

}  // namespace ovms
