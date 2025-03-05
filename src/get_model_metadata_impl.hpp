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

#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011 4457 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "tensorinfo.hpp"

namespace ovms {

using proto_signature_map_t = google::protobuf::Map<std::string, tensorflow::TensorInfo>;

struct ExecutionContext;
class ModelInstance;
class ModelManager;
class PipelineDefinition;
class Server;
class Status;

class GetModelMetadataImpl {
    ModelManager& modelManager;

public:
    GetModelMetadataImpl(ovms::Server& ovmsServer);
    static Status validate(
        const tensorflow::serving::GetModelMetadataRequest* request);

    static void convert(
        const tensor_map_t& from,
        proto_signature_map_t* to);

    static Status buildResponse(
        std::shared_ptr<ModelInstance> instance,
        tensorflow::serving::GetModelMetadataResponse* response);
    static Status buildResponse(
        PipelineDefinition& pipelineDefinition,
        tensorflow::serving::GetModelMetadataResponse* response,
        const ModelManager& manager);

    Status getModelStatus(
        const tensorflow::serving::GetModelMetadataRequest* request,
        tensorflow::serving::GetModelMetadataResponse* response, ExecutionContext context) const;

    static Status getModelStatus(
        const tensorflow::serving::GetModelMetadataRequest* request,
        tensorflow::serving::GetModelMetadataResponse* response,
        ModelManager& manager,
        ExecutionContext context);
    static Status createGrpcRequest(const std::string& model_name, std::optional<int64_t> model_version, tensorflow::serving::GetModelMetadataRequest* request);
    static Status serializeResponse2Json(const tensorflow::serving::GetModelMetadataResponse* response, std::string* output);
};

}  // namespace ovms
