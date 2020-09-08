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
#include <sstream>
#include <string>
#include <vector>

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <signal.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "avx_check.hpp"
#include "config.hpp"
#include "http_server.hpp"
#include "model_service.hpp"
#include "modelmanager.hpp"
#include "prediction_service.hpp"
#include "stringutils.hpp"

using grpc::Server;
using grpc::ServerBuilder;

using namespace ovms;

uint getGRPCServersCount() {
    const char* environmentVariableBuffer = std::getenv("GRPC_SERVERS");
    if (environmentVariableBuffer) {
        auto result = stou32(environmentVariableBuffer);
        if (result && result.value() > 0) {
            return result.value();
        }
    }

    return std::max<uint>(1, ovms::Config::instance().grpcWorkers());
}

struct GrpcChannelArgument {
    std::string key;
    std::string value;
};

// Parses a comma separated list of gRPC channel arguments into list of
// ChannelArgument.
Status parseGrpcChannelArgs(const std::string& channel_arguments_str, std::vector<GrpcChannelArgument>& result) {
    const std::vector<std::string> channel_arguments = tokenize(channel_arguments_str, ',');

    for (const std::string& channel_argument : channel_arguments) {
        std::vector<std::string> key_val = tokenize(channel_argument, '=');
        if (key_val.size() != 2) {
            return StatusCode::GRPC_CHANNEL_ARG_WRONG_FORMAT;
        }
        erase_spaces(key_val[0]);
        erase_spaces(key_val[1]);
        result.push_back({key_val[0], key_val[1]});
    }

    return StatusCode::OK;
}

void configure_logger(const std::string log_level, const std::string log_path) {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    if (!log_path.empty()) {
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path));
    }
    auto serving_logger = std::make_shared<spdlog::logger>("serving", begin(sinks), end(sinks));
    serving_logger->set_level(spdlog::level::info);
    if (!log_level.empty()) {
        if (log_level == "DEBUG") {
            serving_logger->set_level(spdlog::level::debug);
            serving_logger->flush_on(spdlog::level::trace);
        } else if (log_level == "ERROR") {
            serving_logger->set_level(spdlog::level::err);
            serving_logger->flush_on(spdlog::level::err);
        }
    }
    spdlog::set_default_logger(serving_logger);
}

void logConfig(Config& config) {
    spdlog::debug("CLI parameters passed to ovms server");
    if (config.configPath().empty()) {
        spdlog::debug("model_path: {}", config.modelPath());
        spdlog::debug("model_name: {}", config.modelName());
        spdlog::debug("batch_size: {}", config.batchSize());
        spdlog::debug("shape: {}", config.shape());
        spdlog::debug("model_version_policy: {}", config.modelVersionPolicy());
        spdlog::debug("nireq: {}", config.nireq());
        spdlog::debug("target_device: {}", config.targetDevice());
        spdlog::debug("plugin_config: {}", config.pluginConfig());
    } else {
        spdlog::debug("config_path: {}", config.configPath());
    }
    spdlog::debug("gRPC port: {}", config.port());
    spdlog::debug("REST port: {}", config.restPort());
    spdlog::debug("REST workers: {}", config.restWorkers());
    spdlog::debug("gRPC workers: {}", config.grpcWorkers());
    spdlog::debug("gRPC channel arguments: {}", config.grpcChannelArguments());
    spdlog::debug("log level: {}", config.logLevel());
    spdlog::debug("log path: {}", config.logPath());
}

void onInterrupt(int status) {
    spdlog::info("Program interrupted with signal: {}", status);
    spdlog::default_logger()->flush();
    exit(0);
}

void onTerminate(int status) {
    spdlog::info("Program terminated with signal: {}", status);
    spdlog::default_logger()->flush();
    exit(0);
}

void installSignalHandlers() {
    static struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = onInterrupt;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    static struct sigaction sigTermHandler;
    sigTermHandler.sa_handler = onTerminate;
    sigemptyset(&sigTermHandler.sa_mask);
    sigTermHandler.sa_flags = 0;
    sigaction(SIGTERM, &sigTermHandler, NULL);
}

std::vector<std::unique_ptr<Server>> startGRPCServer(
    PredictionServiceImpl& predict_service,
    ModelServiceImpl& model_service) {
    const int GIGABYTE = 1024 * 1024 * 1024;

    std::vector<GrpcChannelArgument> channel_arguments;
    auto& config = ovms::Config::instance();
    auto status = parseGrpcChannelArgs(config.grpcChannelArguments(), channel_arguments);
    if (!status.ok()) {
        spdlog::error("grpc channel arguments passed in wrong format: {}", config.grpcChannelArguments());
        exit(1);
    }

    logConfig(config);
    auto& manager = ModelManager::getInstance();
    status = manager.start();
    if (!status.ok()) {
        spdlog::error("ovms::ModelManager::Start() Error: {}", status.string());
        exit(1);
    }

    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(GIGABYTE);
    builder.SetMaxSendMessageSize(GIGABYTE);
    builder.AddListeningPort("0.0.0.0:" + std::to_string(config.port()), grpc::InsecureServerCredentials());
    builder.RegisterService(&predict_service);
    builder.RegisterService(&model_service);
    for (const GrpcChannelArgument& channel_argument : channel_arguments) {
        // gRPC accept arguments of two types, int and string. We will attempt to
        // parse each arg as int and pass it on as such if successful. Otherwise we
        // will pass it as a string. gRPC will log arguments that were not accepted.
        spdlog::debug("setting grpc channel argument {}: {}", channel_argument.key, channel_argument.value);
        try {
            int i = std::stoi(channel_argument.value);
            builder.AddChannelArgument(channel_argument.key, i);
        } catch (std::invalid_argument const& e) {
            builder.AddChannelArgument(channel_argument.key, channel_argument.value);
        } catch (std::out_of_range const& e) {
            spdlog::error("Out of range parameter {} : {}", channel_argument.key, channel_argument.value);
        }
    }

    std::vector<std::unique_ptr<Server>> servers;
    uint grpcServersCount = getGRPCServersCount();
    servers.reserve(grpcServersCount);
    spdlog::debug("Starting grpc servers: {}", grpcServersCount);

    for (uint i = 0; i < grpcServersCount; ++i) {
        std::unique_ptr<Server> server = builder.BuildAndStart();
        if (server == nullptr) {
            throw std::runtime_error("Failed to start GRPC server at " + std::to_string(config.port()));
        }
        servers.push_back(std::move(server));
    }
    spdlog::info("Server started on port {}", config.port());

    return servers;
}

std::unique_ptr<ovms::http_server> startRESTServer() {
    const int REST_TIMEOUT = 5000;

    auto& config = ovms::Config::instance();
    if (config.restPort() != 0) {
        const std::string server_address = "localhost:" + std::to_string(config.restPort());

        int workers = config.restWorkers() ? config.restWorkers() : 10;
        spdlog::info("workers {}", workers);

        std::unique_ptr<ovms::http_server> restServer = ovms::createAndStartHttpServer(config.restPort(), workers, REST_TIMEOUT);
        if (restServer != nullptr) {
            spdlog::info("Started REST server at {}", server_address);
        } else {
            throw std::runtime_error("Failed to start REST server at " + server_address);
        }

        return restServer;
    }

    return nullptr;
}

int server_main(int argc, char** argv) {
    try {
        auto& config = ovms::Config::instance().parse(argc, argv);
        configure_logger(config.logLevel(), config.logPath());
        installSignalHandlers();

        if (!ovms::check_4th_gen_intel_core_features()) {
            spdlog::error("CPU with AVX support required, current CPU has no such support.");
            return 1;
        }
        PredictionServiceImpl predict_service;
        ModelServiceImpl model_service;

        auto grpc = startGRPCServer(predict_service, model_service);
        auto rest = startRESTServer();

        grpc[0]->Wait();
        if (rest != nullptr) {
            rest->WaitForTermination();
        }
    } catch (std::exception& e) {
        SPDLOG_ERROR("Exception catch: {} - will now terminate.", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        SPDLOG_ERROR("Unknown exception catch - will now terminate.");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
