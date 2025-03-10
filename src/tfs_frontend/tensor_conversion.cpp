//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "tfs_utils.hpp"
#include "../tensor_conversion.hpp"
#include "../tensor_conversion_common.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#pragma warning(push)
#pragma warning(disable : 6269 6294 6201)
#include <opencv2/opencv.hpp>
#pragma warning(pop)

#include "../logging.hpp"
#include "../profiler.hpp"
#include "../status.hpp"

namespace ovms {

Status convertStringRequestFromBufferToOVTensor2D(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::string* buffer) {
    SPDLOG_ERROR("String conversion is not implemented for TFS-API");
    return StatusCode::NOT_IMPLEMENTED;
}
}  // namespace ovms
// we need to see declarations before
#include "../tensor_conversion_after.hpp"
namespace ovms {
template Status convertStringRequestToOVTensor2D<TFSInputTensorType>(const TFSInputTensorType& src, ov::Tensor& tensor, const std::string* buffer);
}  // namespace ovms
