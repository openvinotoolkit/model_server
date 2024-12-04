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

#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "profiler.hpp"
#include "status.hpp"
#include "serialization_common.hpp"
#include "tensorinfo.hpp"

namespace ovms {

Status serializeTensorToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor);

template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {  // does not apply for TFS frontend
    return serializePredictResponse(outputGetter, servableName, servableVersion, outputMap, response, outputNameChooser, useSharedOutputContent);
}

template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {  // does not apply for TFS frontend
    OVMS_PROFILE_FUNCTION();
    Status status;
    ProtoGetter<tensorflow::serving::PredictResponse*, tensorflow::TensorProto&> protoGetter(response);
    for (const auto& [outputName, outputInfo] : outputMap) {
        ov::Tensor tensor;
        status = outputGetter.get(outputNameChooser(outputName, *outputInfo), tensor);
        if (!status.ok()) {
            return status;
        }
        auto& tensorProto = protoGetter.createOutput(outputInfo->getMappedName());
        status = serializeTensorToTensorProto(tensorProto, outputInfo, tensor);
        if (!status.ok()) {
            return status;
        }
    }
    return status;
}
}  // namespace ovms
