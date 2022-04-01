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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "kfs_grpc_inference_service.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

template <typename T>
class OutputGetter {
public:
    OutputGetter(T t) :
        outputSource(t) {}
    Status get(const std::string& name, ov::Tensor& tensor);

private:
    T outputSource;
};

template <typename ProtoStorage, typename ProtoType>
class ProtoGetter {
    ProtoStorage protoStorage;

public:
    ProtoGetter(ProtoStorage protoStorage) :
        protoStorage(protoStorage) {}
    ProtoType get(const std::string& name);
};

Status serializeTensorToTensorProto(
    tensorflow::TensorProto& responseOutput,
    const std::shared_ptr<TensorInfo>& servableOutput,
    ov::Tensor& tensor);

typedef const std::string& (*outputNameChooser_t)(const std::string&, const TensorInfo&);
const std::string& getTensorInfoName(const std::string& first, const TensorInfo& tensorInfo);
const std::string& getOutputMapKeyName(const std::string& first, const TensorInfo& tensorInfo);

template <typename T, typename ResponseType>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const tensor_map_t& outputMap,
    ResponseType* response,
    outputNameChooser_t outputNameChooser);
// partial template specialization
template <typename T>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const tensor_map_t& outputMap,
    tensorflow::serving::PredictResponse* response,
    outputNameChooser_t outputNameChooser) {
    Status status;
    ProtoGetter<tensorflow::serving::PredictResponse*, tensorflow::TensorProto&> protoGetter(response);
    for (const auto& [outputName, outputInfo] : outputMap) {
        ov::Tensor tensor;
        status = outputGetter.get(outputNameChooser(outputName, *outputInfo), tensor);
        if (!status.ok()) {
            return status;
        }
        auto& tensorProto = protoGetter.get(outputInfo->getMappedName());
        status = serializeTensorToTensorProto(tensorProto, outputInfo, tensor);
        if (!status.ok()) {
            return status;
        }
    }
    return status;
}
}  // namespace ovms
