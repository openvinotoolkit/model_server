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

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "gathernodeinputhandler.hpp"
#include "kfs_grpc_inference_service.hpp"
#include "logging.hpp"

namespace ovms {

class CollapseDetails;

char* prepareBuffer(tensorflow::serving::PredictResponse* response, const std::string& name, size_t size) {
    OVMS_PROFILE_FUNCTION();
    tensorflow::TensorProto& tensorProto = (*response->mutable_outputs())[name];
    tensorProto.mutable_tensor_content()->resize(size);
    return tensorProto.mutable_tensor_content()->data();
}

char* prepareBuffer(::inference::ModelInferResponse* response, const std::string& name, size_t size) {
    return 0;
}

template <class ResponseType>
class GatherExitNodeInputHandler : public GatherNodeInputHandler {
    ResponseType* response;

public:
    GatherExitNodeInputHandler(uint32_t inputsMissingCount, const CollapseDetails& collapsingDetails, ResponseType* response) :
        GatherNodeInputHandler(inputsMissingCount, collapsingDetails),
        response(response) {}
    //  Status notifyFinishedDependency() override {
    //    SPDLOG_INFO("GatherExitNodeInputHandler XXXXXXXXXXXXXXX");
    //    return GatherNodeInputHandler::notifyFinishedDependency();
    //  }

    ~GatherExitNodeInputHandler() {
    }

    ov::Tensor makeTensorForWrite(const std::string& name, ov::element::Type_t precision, const ov::Shape& shape) override {
        OVMS_PROFILE_FUNCTION();
        size_t size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            size *= shape[i];
        }
        size *= ov::element::Type(precision).size();
        auto* buf = prepareBuffer(response, name, size);
        return ov::Tensor(precision, shape, buf);
    }
};

}  // namespace ovms
