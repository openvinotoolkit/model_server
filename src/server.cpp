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
#include <sstream>
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
#include "stringutils.hpp"

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

struct GrpcChannelArgument {
    std::string key;
    std::string value;
};

// Parses a comma separated list of gRPC channel arguments into list of
// ChannelArgument.
std::vector<GrpcChannelArgument> parseGrpcChannelArgs(
        const std::string channel_arguments_str) {
    const std::vector<std::string> channel_arguments =
            tokenize(channel_arguments_str, ',');
    std::vector<GrpcChannelArgument> result;
    for (const std::string channel_argument : channel_arguments) {
        const std::vector<std::string> key_val =
                tokenize(channel_argument, '=');
        result.push_back({key_val[0], key_val[1]});
    }
    return result;
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
        } else if (log_level =="ERROR") {
            serving_logger->set_level(spdlog::level::err);
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
        std::stringstream shape_t;
        std::copy(config.shape().begin(), config.shape().end(), std::ostream_iterator<size_t>(shape_t, " "));
        spdlog::debug("shape: {}", shape_t.str());
        spdlog::debug("model_version_policy: {}", config.modelVersionPolicy());
        spdlog::debug("nireq: {}", config.nireq());
        spdlog::debug("target_device: {}", config.targetDevice());
        spdlog::debug("plugin_config: {}", config.pluginConfig());
    } else {
        spdlog::debug("config_path: {}", config.configPath());
    }
    spdlog::debug("gRPC port: {}", config.port());
    spdlog::debug("REST port: {}", config.restPort());
    spdlog::debug("gRPC servers: {}", config.grpcWorkers());
    spdlog::debug("gRPC channel arguments: {}", config.grpcChannelArguments());
    spdlog::debug("log level: {}", config.logLevel());
    spdlog::debug("log path: {}", config.logPath());
}

int server_main(int argc, char** argv) {
    const int GIGABYTE = 1024 * 1024 * 1024;
    auto& config = ovms::Config::instance().parse(argc, argv);
    configure_logger(config.logLevel(), config.logPath());
    logConfig(config);
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
    const std::vector<GrpcChannelArgument> channel_arguments =
            parseGrpcChannelArgs(config.grpcChannelArguments());
    for (const GrpcChannelArgument& channel_argument : channel_arguments) {
        // gRPC accept arguments of two types, int and string. We will attempt to
        // parse each arg as int and pass it on as such if successful. Otherwise we
        // will pass it as a string. gRPC will log arguments that were not accepted.
        spdlog::debug("channel {}: {}", channel_argument.key, channel_argument.value);
        try {
            int i = std::stoi(channel_argument.value);
            builder.AddChannelArgument(channel_argument.key, i);
        }
        catch (std::invalid_argument const &e) {
            builder.AddChannelArgument(channel_argument.key, channel_argument.value);
        }
        catch (std::out_of_range const &e) {
            spdlog::error("Out of range parameter {} : {}", channel_argument.key, channel_argument.value);
        }
    }

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
