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
#pragma once

#include <memory>
#include <string>

#include <openvino/openvino.hpp>

#include "kfs_utils.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../serialization_common.hpp"
#include "../tensorinfo.hpp"

namespace ovms {
Status serializeTensorToTensorProto(
    ::KFSResponse::InferOutputTensor& responseOutput,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor);

template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const ::KFSRequest* request,
    ::KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {
    return serializePredictResponse(outputGetter, servableName, servableVersion, outputMap, response, outputNameChooser, useSharedOutputContent);
}
Status serializeTensorToTensorProtoRaw(
    ::inference::ModelInferResponse::InferOutputTensor& responseOutput,
    std::string* rawOutputContents,
    const std::shared_ptr<const TensorInfo>& servableOutput,
    ov::Tensor& tensor);
template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    ::KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    response->set_model_name(servableName);
    response->set_model_version(std::to_string(servableVersion));
    ProtoGetter<::KFSResponse*, ::KFSResponse::InferOutputTensor&> protoGetter(response);
    for (const auto& [outputName, outputInfo] : outputMap) {
        ov::Tensor tensor;
        status = outputGetter.get(outputNameChooser(outputName, *outputInfo), tensor);
        if (!status.ok()) {
            return status;
        }
        auto& inferOutputTensor = protoGetter.createOutput(outputInfo->getMappedName());
        if (useSharedOutputContent) {
            status = serializeTensorToTensorProtoRaw(inferOutputTensor, protoGetter.createContent(outputInfo->getMappedName()), outputInfo, tensor);
        } else {
            status = serializeTensorToTensorProto(inferOutputTensor, outputInfo, tensor);
        }

        if (!status.ok()) {
            return status;
        }
    }
    return status;
}
}  // namespace ovms
