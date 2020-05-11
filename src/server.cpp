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
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include "config.hpp"
#include "modelmanager.hpp"
#include "prediction_service.hpp"
#include "model_service.hpp"

using grpc::Server;
using grpc::ServerBuilder;

using namespace ovms;

uint getGRPCServersCount() {
    const char* environmentVariableBuffer = std::getenv("GRPC_SERVERS");
    if (environmentVariableBuffer) {
        return std::atoi(environmentVariableBuffer);
    }

    auto& config = ovms::Config::instance();
    uint configGRPCServersCount = config.grpcWorkers();
    return configGRPCServersCount;
}

void configure_logger() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    if (const char* log_path = std::getenv("LOG_PATH")) {
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path));
    }
    auto serving_logger = std::make_shared<spdlog::logger>("serving", begin(sinks), end(sinks));

    // auto serving_logger = spdlog::stdout_logger_mt("serving");
    serving_logger->set_level(spdlog::level::info);
    if (const char* log_level = std::getenv("LOG_LEVEL")) {
        if (std::strcmp(log_level, "DEBUG") == 0) {
            serving_logger->set_level(spdlog::level::debug);
        } else if (std::strcmp(log_level, "ERROR") == 0) {
            serving_logger->set_level(spdlog::level::err);
        }
    }
    spdlog::set_default_logger(serving_logger);
}

int server_main(int argc, char** argv) {
    const int GIGABYTE = 1024 * 1024 * 1024;
    configure_logger();
    auto& config = ovms::Config::instance().parse(argc, argv);
    auto& manager = ModelManager::getInstance();

    Status status = manager.start();

    if (status != Status::OK) {
        spdlog::error("ovms::ModelManager::Start() Error: {}", StatusDescription::getError(status));
        return 1;
    }

    PredictionServiceImpl service;
    ModelServiceImpl model_service;
    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(GIGABYTE);
    builder.SetMaxSendMessageSize(GIGABYTE);
    builder.AddListeningPort("0.0.0.0:" + std::to_string(config.port()), grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.RegisterService(&model_service);

    std::vector<std::unique_ptr<Server>> servers;
    uint grpcServersCount = getGRPCServersCount();
    spdlog::debug("Starting grpcservers: {}", grpcServersCount);

    for (uint i = 0; i < grpcServersCount; ++i) {
        servers.push_back(std::unique_ptr<Server>(builder.BuildAndStart()));
    }
    spdlog::info("Server started on port {}", config.port() );
    servers[0]->Wait();

    return 0;
}

