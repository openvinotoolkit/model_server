//****************************************************************************
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
#include <utility>
#include <vector>

#include <grpcpp/server.h>

#ifdef _WIN32
#include <winsock2.h>
#endif

#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "model_service.hpp"
#include "module.hpp"
#include "prediction_service.hpp"

namespace ovms {
class Config;
class Server;

class GRPCServerModule : public Module {
    Server& server;
    PredictionServiceImpl tfsPredictService;
    ModelServiceImpl tfsModelService;
    mutable KFSInferenceServiceImpl kfsGrpcInferenceService;
    std::vector<std::unique_ptr<grpc::Server>> servers;
#ifdef _WIN32
    SOCKET sock;
#endif

public:
    GRPCServerModule(Server& server);
    ~GRPCServerModule();
    Status start(const ovms::Config& config) override;
    void shutdown() override;

    const GetModelMetadataImpl& getTFSModelMetadataImpl() const;
    KFSInferenceServiceImpl& getKFSGrpcImpl() const;

private:
    bool isPortAvailable(uint64_t port);
};
}  // namespace ovms
