//*****************************************************************************
// Copyright 2024 Intel Corporation
//
// Licensed under the Apache License, Version 3.0 (the "License");
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
#include "kfs_request_utils.hpp"

#include <map>

#include "../logging.hpp"
#include "../stringutils.hpp"

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
bool useSharedOutputContentFn(const ::KFSRequest* request) {
    return true;
}
Status RequestTensorExtractor<KFSRequest, KFSTensorInputProto, ExtractChoice::EXTRACT_INPUT>::extract(const KFSRequest& request, const std::string& name, const KFSTensorInputProto** tensor, size_t* bufferId) {
    if (bufferId == nullptr) {
        return StatusCode::INTERNAL_ERROR;
    }
    size_t id = 0;
    auto it = request.inputs().begin();
    while (it != request.inputs().end()) {
        if (it->name() == name) {
            break;
        }
        ++it;
        ++id;
    }
    if (it == request.inputs().end()) {
        return StatusCode::NONEXISTENT_TENSOR;
    }
    *bufferId = id;
    *tensor = &(*it);
    return StatusCode::OK;
}
}  // namespace ovms
