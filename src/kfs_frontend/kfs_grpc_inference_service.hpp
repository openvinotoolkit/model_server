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

using inference::GRPCInferenceService;
using KFSServerMetadataRequest = inference::ServerMetadataRequest;
using KFSServerMetadataResponse = inference::ServerMetadataResponse;
using KFSModelMetadataRequest = inference::ModelMetadataRequest;
using KFSModelMetadataResponse = inference::ModelMetadataResponse;
using KFSRequest = inference::ModelInferRequest;
using KFSResponse = inference::ModelInferResponse;
using KFSTensorInputProto = inference::ModelInferRequest::InferInputTensor;
using KFSTensorOutputProto = inference::ModelInferResponse::InferOutputTensor;
using KFSShapeType = google::protobuf::RepeatedField<int64_t>;
using KFSGetModelStatusRequest = inference::ModelReadyRequest;
using KFSGetModelStatusResponse = inference::ModelReadyResponse;
using KFSDataType = std::string;
using KFSInputTensorIteratorType = google::protobuf::internal::RepeatedPtrIterator<const ::inference::ModelInferRequest_InferInputTensor>;
using KFSOutputTensorIteratorType = google::protobuf::internal::RepeatedPtrIterator<const ::inference::ModelInferResponse_InferOutputTensor>;

namespace ovms {
class ExecutionContext;
class MediapipeGraphDefinition;
class Model;
class ModelInstance;
class ModelInstanceUnloadGuard;
class ModelManager;
class ServableMetricReporter;
class Pipeline;
class Server;
class Status;
class TensorInfo;
class PipelineDefinition;

class KFSInferenceServiceImpl : public GRPCInferenceService::Service {
protected:
    const Server& ovmsServer;
    ModelManager& modelManager;

public:
    Status ModelReadyImpl(::grpc::ServerContext* context, const KFSGetModelStatusRequest* request, KFSGetModelStatusResponse* response, ExecutionContext executionContext);
    Status ServerMetadataImpl(::grpc::ServerContext* context, const KFSServerMetadataRequest* request, KFSServerMetadataResponse* response);
    Status ModelMetadataImpl(::grpc::ServerContext* context, const KFSModelMetadataRequest* request, KFSModelMetadataResponse* response, ExecutionContext executionContext);
    Status ModelInferImpl(::grpc::ServerContext* context, const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut);
    KFSInferenceServiceImpl(const Server& server);
    ::grpc::Status ServerLive(::grpc::ServerContext* context, const ::inference::ServerLiveRequest* request, ::inference::ServerLiveResponse* response) override;
    ::grpc::Status ServerReady(::grpc::ServerContext* context, const ::inference::ServerReadyRequest* request, ::inference::ServerReadyResponse* response) override;
    ::grpc::Status ModelReady(::grpc::ServerContext* context, const KFSGetModelStatusRequest* request, KFSGetModelStatusResponse* response) override;
    ::grpc::Status ServerMetadata(::grpc::ServerContext* context, const KFSServerMetadataRequest* request, KFSServerMetadataResponse* response) override;
    ::grpc::Status ModelMetadata(::grpc::ServerContext* context, const KFSModelMetadataRequest* request, KFSModelMetadataResponse* response) override;
    ::grpc::Status ModelInfer(::grpc::ServerContext* context, const KFSRequest* request, KFSResponse* response) override;
    ::grpc::Status ModelStreamInfer(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::inference::ModelStreamInferResponse, ::inference::ModelInferRequest>* stream) override;
    static Status buildResponse(Model& model, ModelInstance& instance, KFSModelMetadataResponse* response);
    static Status buildResponse(PipelineDefinition& pipelineDefinition, KFSModelMetadataResponse* response);
    static Status buildResponse(std::shared_ptr<ModelInstance> instance, KFSGetModelStatusResponse* response);
    static Status buildResponse(PipelineDefinition& pipelineDefinition, KFSGetModelStatusResponse* response);
    static Status buildResponse(MediapipeGraphDefinition& pipelineDefinition, KFSGetModelStatusResponse* response);
    static Status buildResponse(MediapipeGraphDefinition& mediapipeGraphDefinition, KFSModelMetadataResponse* response);
    static void convert(const std::pair<std::string, std::shared_ptr<const TensorInfo>>& from, KFSModelMetadataResponse::TensorMetadata* to);
    static Status getModelReady(const KFSGetModelStatusRequest* request, KFSGetModelStatusResponse* response, const ModelManager& manager, ExecutionContext executionContext);

protected:
    Status getModelInstance(const KFSRequest* request,
        std::shared_ptr<ovms::ModelInstance>& modelInstance,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr);
    Status getPipeline(const KFSRequest* request,
        KFSResponse* response,
        std::unique_ptr<ovms::Pipeline>& pipelinePtr);
};

}  // namespace ovms
