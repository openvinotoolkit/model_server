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
#include "tfs_request_utils.hpp"
#include <map>
#include <memory>
#include <string>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include "../extractchoice.hpp"
#include "../requesttensorextractor.hpp"
#include "../logging.hpp"
#include "../shape.hpp"
#include "../status.hpp"

namespace ovms {
template class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_INPUT>;
template class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_OUTPUT>;

// Assuming the request is already validated, therefore no need to check for negative values or zeros
std::optional<Dimension> getRequestBatchSize(const tensorflow::serving::PredictRequest* request, const size_t batchSizeIndex) {
    auto requestInputItr = request->inputs().begin();
    if (requestInputItr == request->inputs().end()) {
        SPDLOG_DEBUG("Failed to get batch size of a request. Validation of request failed");
        return std::nullopt;
    }
    auto& requestInput = requestInputItr->second;  // assuming same batch size for all inputs

    if (requestInput.tensor_shape().dim_size() < 0) {
        SPDLOG_DEBUG("Failed to get batch size of a request. Input shape size cannot be a negative number. Validation of request failed");
        return std::nullopt;
    }

    if (static_cast<size_t>(requestInput.tensor_shape().dim_size()) < batchSizeIndex + 1) {
        SPDLOG_DEBUG("Failed to get batch size of a request. Batch size index out of shape range. Validation of request failed");
        return std::nullopt;
    }
    return Dimension(requestInput.tensor_shape().dim(batchSizeIndex).size());
}
// Assuming the request is already validated, therefore no need to check for negative values or zeros
std::map<std::string, shape_t> getRequestShapes(const tensorflow::serving::PredictRequest* request) {
    std::map<std::string, shape_t> requestShapes;
    for (auto& it : request->inputs()) {
        shape_t requestShape;
        std::string name = it.first;
        auto& requestInput = it.second;
        for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
            requestShape.push_back(requestInput.tensor_shape().dim(i).size());
        }
        requestShapes[name] = std::move(requestShape);
    }
    return requestShapes;
}
bool useSharedOutputContentFn(const tensorflow::serving::PredictRequest* request) {
    // does not apply for TFS frontend
    return false;
}

Status RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_INPUT>::extract(const tensorflow::serving::PredictRequest& request, const std::string& name, const tensorflow::TensorProto** tensor, size_t* bufferId) {
    if (bufferId == nullptr) {
        return StatusCode::INTERNAL_ERROR;
    }
    auto it = request.inputs().find(name);
    if (it == request.inputs().end()) {
        return StatusCode::NONEXISTENT_TENSOR;
    }
    *tensor = &it->second;
    return StatusCode::OK;
}
template <>
class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_OUTPUT>;
template <>
class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_INPUT>;
}  // namespace ovms
