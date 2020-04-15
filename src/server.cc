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
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include <iostream>
#include <vector>

#include "config.h"
#include "modelmanager.h"
#include "prediction_service.hpp"

using grpc::Server;
using grpc::ServerBuilder;

using namespace ovms;

int main(int argc, char** argv)
{
    const int GIGABYTE = 1024 * 1024 * 1024;

    auto& config = ovms::Config::instance().parse(argc, argv);
    auto& manager = ModelManager::getInstance();

    Status status = manager.start();

    if (status != Status::OK) {
        std::cout << "ovms::ModelManager::Start() Error: " << int(status) << std::endl;
        return 1;
    }

    PredictionServiceImpl service;
    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(GIGABYTE);
    builder.SetMaxSendMessageSize(GIGABYTE);
    builder.AddListeningPort("0.0.0.0:" + std::to_string(config.port()), grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::vector<std::unique_ptr<Server>> servers;
    for (int i = 0; i < config.grpcWorkers(); i++) {
        servers.push_back(std::unique_ptr<Server>(builder.BuildAndStart()));
    }

    std::cout << "Server started on port " << config.port() << std::endl;
    servers[0]->Wait();

    return 0;
}
