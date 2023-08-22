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
#include "prediction_service_utils.hpp"

#include <map>

#include "capi_frontend/inferencerequest.hpp"
#include "capi_frontend/inferencetensor.hpp"
#include "deserialization.hpp"
#include "executingstreamidguard.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "serialization.hpp"
#include "stringutils.hpp"
#include "timer.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

namespace ovms {

// Assuming the request is already validated, therefore no need to check for negative values or zeros
std::optional<Dimension> getRequestBatchSize(const ::KFSRequest* request, const size_t batchSizeIndex) {
    auto requestInputItr = request->inputs().begin();
    if (requestInputItr == request->inputs().end()) {
        SPDLOG_DEBUG("Failed to get batch size of a request. Validation of request failed");
        return std::nullopt;
    }
    auto& requestInput = requestInputItr;  // assuming same batch size for all inputs
    if (requestInput->shape().size() < 0) {
        SPDLOG_DEBUG("Failed to get batch size of a request. Input shape size cannot be a negative number. Validation of request failed");
        return std::nullopt;
    }
    if (static_cast<size_t>(requestInput->shape().size()) < batchSizeIndex + 1) {
        SPDLOG_DEBUG("Failed to get batch size of a request. Batch size index out of shape range. Validation of request failed");
        return std::nullopt;
    }
    return Dimension(requestInput->shape()[batchSizeIndex]);
}

// Assuming the request is already validated, therefore no need to check for negative values or zeros
std::map<std::string, shape_t> getRequestShapes(const ::KFSRequest* request) {
    std::map<std::string, shape_t> requestShapes;
    for (auto& it : request->inputs()) {
        shape_t requestShape;
        std::string name = it.name();
        auto& requestInput = it;
        for (int i = 0; i < requestInput.shape().size(); i++) {
            requestShape.push_back(requestInput.shape()[i]);
        }
        requestShapes[name] = std::move(requestShape);
    }
    return requestShapes;
}

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
std::optional<Dimension> getRequestBatchSize(const InferenceRequest* request, const size_t batchSizeIndex) {
    size_t bs = 0;
    auto status = request->getBatchSize(bs, batchSizeIndex);
    if (!status.ok()) {
        return std::nullopt;
    }
    return bs;
}

std::map<std::string, shape_t> getRequestShapes(const InferenceRequest* request) {
    return request->getRequestShapes();
}

bool useSharedOutputContentFn(const tensorflow::serving::PredictRequest* request) {
    // does not apply for TFS frontend
    return false;
}

bool useSharedOutputContentFn(const ::KFSRequest* request) {
    return true;
}

bool useSharedOutputContentFn(const InferenceRequest* request) {
    // does not apply for C-API frontend
    return false;
}

}  // namespace ovms
