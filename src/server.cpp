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
#include <netinet/in.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

#include "config.hpp"
#include "http_server.hpp"
#include "logging.hpp"
#include "model_service.hpp"
#include "modelmanager.hpp"
#include "prediction_service.hpp"
#include "stringutils.hpp"

using grpc::Server;
using grpc::ServerBuilder;

using namespace ovms;

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

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

bool isPortAvailable(uint64_t port) {
    struct sockaddr_in addr;
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == -1) {
        return false;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(s);
        return false;
    }
    close(s);
    return true;
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

void logConfig(Config& config) {
    SPDLOG_DEBUG("CLI parameters passed to ovms server");
    if (config.configPath().empty()) {
        SPDLOG_DEBUG("model_path: {}", config.modelPath());
        SPDLOG_DEBUG("model_name: {}", config.modelName());
        SPDLOG_DEBUG("batch_size: {}", config.batchSize());
        SPDLOG_DEBUG("shape: {}", config.shape());
        SPDLOG_DEBUG("model_version_policy: {}", config.modelVersionPolicy());
        SPDLOG_DEBUG("nireq: {}", config.nireq());
        SPDLOG_DEBUG("target_device: {}", config.targetDevice());
        SPDLOG_DEBUG("plugin_config: {}", config.pluginConfig());
        SPDLOG_DEBUG("stateful: {}", config.stateful());
        SPDLOG_DEBUG("sequence_timeout_seconds: {}", config.sequenceTimeoutSeconds());
        SPDLOG_DEBUG("max_sequence_number: {}", config.maxSequenceNumber());
        SPDLOG_DEBUG("low_latency_transformation: {}", config.lowLatencyTransformation());
    } else {
        SPDLOG_DEBUG("config_path: {}", config.configPath());
    }
    SPDLOG_DEBUG("gRPC port: {}", config.port());
    SPDLOG_DEBUG("REST port: {}", config.restPort());
    SPDLOG_DEBUG("gRPC bind address: {}", config.grpcBindAddress());
    SPDLOG_DEBUG("REST bind address: {}", config.restBindAddress());
    SPDLOG_DEBUG("REST workers: {}", config.restWorkers());
    SPDLOG_DEBUG("gRPC workers: {}", config.grpcWorkers());
    SPDLOG_DEBUG("gRPC channel arguments: {}", config.grpcChannelArguments());
    SPDLOG_DEBUG("log level: {}", config.logLevel());
    SPDLOG_DEBUG("log path: {}", config.logPath());
    SPDLOG_DEBUG("file system poll wait seconds: {}", config.filesystemPollWaitSeconds());
}

void onInterrupt(int status) {
    shutdown_request = 1;
}

void onTerminate(int status) {
    shutdown_request = 1;
}

void onIllegal(int status) {
    shutdown_request = 2;
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

    static struct sigaction sigIllHandler;
    sigIllHandler.sa_handler = onIllegal;
    sigemptyset(&sigIllHandler.sa_mask);
    sigIllHandler.sa_flags = 0;
    sigaction(SIGILL, &sigIllHandler, NULL);
}

std::vector<std::unique_ptr<Server>> startGRPCServer(
    PredictionServiceImpl& predict_service,
    ModelServiceImpl& model_service) {
    const int GIGABYTE = 1024 * 1024 * 1024;

    std::vector<GrpcChannelArgument> channel_arguments;
    auto& config = ovms::Config::instance();
    auto status = parseGrpcChannelArgs(config.grpcChannelArguments(), channel_arguments);
    if (!status.ok()) {
        SPDLOG_ERROR("grpc channel arguments passed in wrong format: {}", config.grpcChannelArguments());
        exit(1);
    }

    logConfig(config);
    auto& manager = ModelManager::getInstance();
    status = manager.start();
    if (!status.ok()) {
        SPDLOG_ERROR("ovms::ModelManager::Start() Error: {}", status.string());
        exit(1);
    }

    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(GIGABYTE);
    builder.SetMaxSendMessageSize(GIGABYTE);
    builder.AddListeningPort(config.grpcBindAddress() + ":" + std::to_string(config.port()), grpc::InsecureServerCredentials());
    builder.RegisterService(&predict_service);
    builder.RegisterService(&model_service);
    for (const GrpcChannelArgument& channel_argument : channel_arguments) {
        // gRPC accept arguments of two types, int and string. We will attempt to
        // parse each arg as int and pass it on as such if successful. Otherwise we
        // will pass it as a string. gRPC will log arguments that were not accepted.
        SPDLOG_DEBUG("setting grpc channel argument {}: {}", channel_argument.key, channel_argument.value);
        try {
            int i = std::stoi(channel_argument.value);
            builder.AddChannelArgument(channel_argument.key, i);
        } catch (std::invalid_argument const& e) {
            builder.AddChannelArgument(channel_argument.key, channel_argument.value);
        } catch (std::out_of_range const& e) {
            SPDLOG_WARN("Out of range parameter {} : {}", channel_argument.key, channel_argument.value);
        }
    }

    std::vector<std::unique_ptr<Server>> servers;
    uint grpcServersCount = getGRPCServersCount();
    servers.reserve(grpcServersCount);
    SPDLOG_DEBUG("Starting grpc servers: {}", grpcServersCount);

    if (!isPortAvailable(config.port())) {
        throw std::runtime_error("Failed to start GRPC server at " + config.grpcBindAddress() + ":" + std::to_string(config.port()));
    }
    for (uint i = 0; i < grpcServersCount; ++i) {
        std::unique_ptr<Server> server = builder.BuildAndStart();
        if (server == nullptr) {
            throw std::runtime_error("Failed to start GRPC server at " + std::to_string(config.port()));
        }
        servers.push_back(std::move(server));
    }
    SPDLOG_INFO("Server started on port {}", config.port());

    return servers;
}

std::unique_ptr<ovms::http_server> startRESTServer() {
    const int REST_TIMEOUT = 5000;

    auto& config = ovms::Config::instance();
    if (config.restPort() != 0) {
        const std::string server_address = config.restBindAddress() + ":" + std::to_string(config.restPort());

        int workers = config.restWorkers() ? config.restWorkers() : 10;
        SPDLOG_INFO("Will start {} REST workers", workers);

        std::unique_ptr<ovms::http_server> restServer = ovms::createAndStartHttpServer(config.restBindAddress(), config.restPort(), workers, REST_TIMEOUT);
        if (restServer != nullptr) {
            SPDLOG_INFO("Started REST server at {}", server_address);
        } else {
            throw std::runtime_error("Failed to start REST server at " + server_address);
        }

        return restServer;
    }

    return nullptr;
}

int server_main(int argc, char** argv) {
    installSignalHandlers();
    try {
        auto& config = ovms::Config::instance().parse(argc, argv);
        configure_logger(config.logLevel(), config.logPath());

        PredictionServiceImpl predict_service;
        ModelServiceImpl model_service;

        auto grpc = startGRPCServer(predict_service, model_service);
        auto rest = startRESTServer();

        while (!shutdown_request) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (shutdown_request == 2) {
            SPDLOG_ERROR("Illegal operation. OVMS started on unsupported device");
        }
        SPDLOG_INFO("Shutting down");
        for (const auto& g : grpc) {
            g->Shutdown();
        }

        if (rest != nullptr) {
            rest->Terminate();
        }

        ModelManager::getInstance().join();
    } catch (std::exception& e) {
        SPDLOG_ERROR("Exception catch: {} - will now terminate.", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        SPDLOG_ERROR("Unknown exception catch - will now terminate.");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
