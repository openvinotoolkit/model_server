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
#include "kfs_grpc_inference_service.hpp"
#include "shape.hpp"

namespace ovms {

std::optional<Dimension> getRequestBatchSize(const ::inference::ModelInferRequest* request, const size_t batchSizeIndex);

std::map<std::string, shape_t> getRequestShapes(const ::inference::ModelInferRequest* request);

std::optional<Dimension> getRequestBatchSize(const tensorflow::serving::PredictRequest* request, const size_t batchSizeIndex);
std::map<std::string, shape_t> getRequestShapes(const tensorflow::serving::PredictRequest* request);

}  // namespace ovms
