//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include <string>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#pragma GCC diagnostic pop

#include "modelversion.hpp"

namespace ovms {
class ExecutionContext;
class ModelManager;
class ModelVersionStatus;
class Server;
class Status;

void addStatusToResponse(tensorflow::serving::GetModelStatusResponse* response, model_version_t version, const ModelVersionStatus& model_version_status);

class ModelServiceImpl final : public tensorflow::serving::ModelService::Service {
    ovms::ModelManager& modelManager;

public:
    ModelServiceImpl(ovms::Server& ovmsServer);
    ::grpc::Status GetModelStatus(::grpc::ServerContext* context,
        const tensorflow::serving::GetModelStatusRequest* request,
        tensorflow::serving::GetModelStatusResponse* response) override;
    ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext* context,
        const tensorflow::serving::ReloadConfigRequest* request,
        tensorflow::serving::ReloadConfigResponse* response) override;
};

class GetModelStatusImpl {
    ovms::Server& ovmsServer;

public:
    GetModelStatusImpl(ovms::Server& ovmsServer);
    static Status getModelStatus(const tensorflow::serving::GetModelStatusRequest* request, tensorflow::serving::GetModelStatusResponse* response, ModelManager& manager, ExecutionContext context);
    static Status createGrpcRequest(std::string model_name, const std::optional<int64_t> model_version, tensorflow::serving::GetModelStatusRequest* request);
    static Status serializeResponse2Json(const tensorflow::serving::GetModelStatusResponse* response, std::string* output);

    static Status getAllModelsStatuses(std::map<std::string, tensorflow::serving::GetModelStatusResponse>& models_versions, ModelManager& manager, ExecutionContext context);
    static Status serializeModelsStatuses2Json(const std::map<std::string, tensorflow::serving::GetModelStatusResponse>& models_versions, std::string& output);
};

}  // namespace ovms
