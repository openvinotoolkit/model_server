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

#include <memory>
#include <string>
#include <utility>

#include <grpcpp/server_context.h>

#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"
#include "status.hpp"

namespace ovms {

using inference::GRPCInferenceService;
class Model;
class ModelManager;
class ModelInstance;
class ModelInstanceUnloadGuard;
class Pipeline;
class Server;
class Status;
class TensorInfo;
class PipelineDefinition;

class KFSInferenceServiceImpl final : public GRPCInferenceService::Service {
    const Server& ovmsServer;

public:
    KFSInferenceServiceImpl(const Server& server);
    ::grpc::Status ServerLive(::grpc::ServerContext* context, const ::inference::ServerLiveRequest* request, ::inference::ServerLiveResponse* response) override;
    ::grpc::Status ServerReady(::grpc::ServerContext* context, const ::inference::ServerReadyRequest* request, ::inference::ServerReadyResponse* response) override;
    ::grpc::Status ModelReady(::grpc::ServerContext* context, const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response) override;
    ::grpc::Status ServerMetadata(::grpc::ServerContext* context, const ::inference::ServerMetadataRequest* request, ::inference::ServerMetadataResponse* response) override;
    ::grpc::Status ModelMetadata(::grpc::ServerContext* context, const ::inference::ModelMetadataRequest* request, ::inference::ModelMetadataResponse* response) override;
    ::grpc::Status ModelInfer(::grpc::ServerContext* context, const ::inference::ModelInferRequest* request, ::inference::ModelInferResponse* response) override;
    static Status buildResponse(Model& model, ModelInstance& instance, ::inference::ModelMetadataResponse* response);
    static Status buildResponse(PipelineDefinition& pipelineDefinition, ::inference::ModelMetadataResponse* response);
    static Status buildResponse(std::shared_ptr<ModelInstance> instance, ::inference::ModelReadyResponse* response);
    static Status buildResponse(PipelineDefinition& pipelineDefinition, ::inference::ModelReadyResponse* response);
    static void convert(const std::pair<std::string, std::shared_ptr<TensorInfo>>& from, ::inference::ModelMetadataResponse::TensorMetadata* to);
    static Status getModelReady(const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response, const ModelManager& manager);

protected:
    Status getModelInstance(const ::inference::ModelInferRequest* request,
        std::shared_ptr<ovms::ModelInstance>& modelInstance,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr);
    Status getPipeline(const ::inference::ModelInferRequest* request,
        ::inference::ModelInferResponse* response,
        std::unique_ptr<ovms::Pipeline>& pipelinePtr);
};

}  // namespace ovms
