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
#pragma once
#include <map>
#include <memory>
#include <string>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include "capi_frontend/inferencerequest.hpp"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "extractchoice.hpp"
#include "requesttensorextractor.hpp"
#include "logging.hpp"
#include "shape.hpp"
#include "status.hpp"

namespace ovms {
class InferenceRequest;

template <typename RequestType>
std::optional<Dimension> getRequestBatchSize(const RequestType* request, const size_t batchSizeIndex);
template <typename RequestType>
std::map<std::string, shape_t> getRequestShapes(const RequestType* request);
std::optional<Dimension> getRequestBatchSize(const ::KFSRequest* request, const size_t batchSizeIndex);
template <typename RequestType>
std::map<std::string, shape_t> getRequestShapes(const ::KFSRequest* request);

std::optional<Dimension> getRequestBatchSize(const tensorflow::serving::PredictRequest* request, const size_t batchSizeIndex);
std::map<std::string, shape_t> getRequestShapes(const tensorflow::serving::PredictRequest* request);

template <>
class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_OUTPUT> {
public:
    static Status extract(const tensorflow::serving::PredictRequest& request, const std::string& name, const tensorflow::TensorProto** tensor, size_t* bufferId) {
        return StatusCode::NOT_IMPLEMENTED;
    }
};

template <>
class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_INPUT> {
public:
    static Status extract(const tensorflow::serving::PredictRequest& request, const std::string& name, const tensorflow::TensorProto** tensor, size_t* bufferId) {
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
};

template <>
class RequestTensorExtractor<KFSRequest, KFSTensorInputProto, ExtractChoice::EXTRACT_OUTPUT> {
public:
    static Status extract(const KFSRequest& request, const std::string& name, const KFSTensorInputProto** tensor, size_t* bufferId) {
        return StatusCode::NOT_IMPLEMENTED;
    }
};

template <>
class RequestTensorExtractor<KFSRequest, KFSTensorInputProto, ExtractChoice::EXTRACT_INPUT> {
public:
    static Status extract(const KFSRequest& request, const std::string& name, const KFSTensorInputProto** tensor, size_t* bufferId) {
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
};

/**
 * This is specific check required for passing KFS API related info
 * which informs how response should be formatted. Therefore return value should not have an impact for
 * any other frontend.
 */
bool useSharedOutputContentFn(const tensorflow::serving::PredictRequest* request);
bool useSharedOutputContentFn(const ::KFSRequest* request);
bool useSharedOutputContentFn(const InferenceRequest* request);
}  // namespace ovms
