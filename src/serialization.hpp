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

#include <memory>
#include <string>

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

template <typename T>
class OutputGetter {
public:
    OutputGetter(T t) :
        outputSource(t) {}
    Status get(const std::string& name, InferenceEngine::Blob::Ptr& blob);

private:
    T outputSource;
};
template <typename T>
class OutputGetter_2 {
public:
    OutputGetter_2(T t) :
        outputSource(t) {}
    Status get(const std::string& name, ov::runtime::Tensor& blob);

private:
    T outputSource;
};

Status serializeBlobToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& networkOutput,
    InferenceEngine::Blob::Ptr blob);

Status serializeBlobToTensorProto_2(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& networkOutput,
    ov::runtime::Tensor& blob);

Status serializePredictResponse(
    InferenceEngine::InferRequest& inferRequest,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response);

Status serializePredictResponse_2(
    ov::runtime::InferRequest& inferRequest,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response);

template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response) {
    Status status;
    for (const auto& [outputName, outputInfo] : outputMap) {
        InferenceEngine::Blob::Ptr blob;
        status = outputGetter.get(outputName, blob);
        if (!status.ok()) {
            return status;
        }
        auto& tensorProto = (*response->mutable_outputs())[outputInfo->getMappedName()];
        status = serializeBlobToTensorProto(tensorProto, outputInfo, blob);
        if (!status.ok()) {
            return status;
        }
    }
    return status;
}
}  // namespace ovms
