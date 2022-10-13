//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <functional>
#include <string>

#include <openvino/openvino.hpp>

#include "gathernodeinputhandler.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"
#pragma GCC diagnostic pop

#include "logging.hpp"
#include "profiler.hpp"
#include "status.hpp"

namespace ovms {

Status prepareConsolidatedTensorImpl(tensorflow::serving::PredictResponse* response, char*& tensorOut, const std::string& name, size_t size);
Status prepareConsolidatedTensorImpl(::inference::ModelInferResponse* response, char*& tensorOut, const std::string& name, size_t size);

template <class ResponseType>
class GatherExitNodeInputHandler : public GatherNodeInputHandler {
    ResponseType* response;

    Status prepareConsolidatedTensor(ov::Tensor& tensorOut, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape) const override {
        OVMS_PROFILE_FUNCTION();
        auto numOfElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<ov::Shape::value_type>());
        size_t numOfBytes = numOfElements * ov::element::Type(precision).size();
        char* buffer = nullptr;
        auto status = prepareConsolidatedTensorImpl(response, buffer, name, numOfBytes);
        if (!status.ok()) {
            return status;
        }
        if (!buffer) {
            return StatusCode::INTERNAL_ERROR;
        }
        tensorOut = ov::Tensor(precision, shape, buffer);
        return StatusCode::OK;
    }

public:
    GatherExitNodeInputHandler(uint32_t inputsMissingCount, const CollapseDetails& collapsingDetails, ResponseType* response) :
        GatherNodeInputHandler(inputsMissingCount, collapsingDetails),
        response(response) {}
};

}  // namespace ovms
