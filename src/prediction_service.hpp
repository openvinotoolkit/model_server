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

#include <grpcpp/server_context.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
//#include "kfserving_api/grpc_predict_v2.grpc.pb.h"
// TODO
#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"
#pragma GCC diagnostic pop

namespace ovms {

class PredictionServiceImpl final : public tensorflow::serving::PredictionService::Service {
    grpc::Status Predict(
        grpc::ServerContext* context,
        const tensorflow::serving::PredictRequest* request,
        tensorflow::serving::PredictResponse* response) override;

    grpc::Status GetModelMetadata(
        grpc::ServerContext* context,
        const tensorflow::serving::GetModelMetadataRequest* request,
        tensorflow::serving::GetModelMetadataResponse* response) override;
};

using inference::GRPCInferenceService;

class KFSInferenceServiceImpl final : public GRPCInferenceService::Service {
::grpc::Status ServerLive(::grpc::ServerContext* context, const ::inference::ServerLiveRequest* request, ::inference::ServerLiveResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}
::grpc::Status ServerReady(::grpc::ServerContext* context, const ::inference::ServerReadyRequest* request, ::inference::ServerReadyResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}
::grpc::Status ModelReady(::grpc::ServerContext* context, const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}
::grpc::Status ServerMetadata(::grpc::ServerContext* context, const ::inference::ServerMetadataRequest* request, ::inference::ServerMetadataResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}
::grpc::Status ModelMetadata(::grpc::ServerContext* context, const ::inference::ModelMetadataRequest* request, ::inference::ModelMetadataResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}
::grpc::Status ModelInfer(::grpc::ServerContext* context, const ::inference::ModelInferRequest* request, ::inference::ModelInferResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}
};

}  // namespace ovms
