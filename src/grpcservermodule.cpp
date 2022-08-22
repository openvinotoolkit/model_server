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
#include "grpcservermodule.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
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
#include "kfs_grpc_inference_service.hpp"
#include "logging.hpp"
#include "model_service.hpp"
#include "modelmanager.hpp"
#include "prediction_service.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "stringutils.hpp"
#include "version.hpp"

using grpc::ServerBuilder;

namespace ovms {
}  // namespace ovms
using namespace ovms;
static const int GIGABYTE = 1024 * 1024 * 1024;

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

uint getGRPCServersCount(const ovms::Config& config) {
    const char* environmentVariableBuffer = std::getenv("GRPC_SERVERS");
    if (environmentVariableBuffer) {
        auto result = stou32(environmentVariableBuffer);
        if (result && result.value() > 0) {
            return result.value();
        }
    }

    return std::max<uint>(1, config.grpcWorkers());
}

GRPCServerModule::~GRPCServerModule() {
    if (state != ModuleState::SHUTDOWN)
        this->shutdown();
}

GRPCServerModule::GRPCServerModule(Server& server) :
    server(server),
    tfsPredictService(this->server),
    tfsModelService(this->server),
    kfsGrpcInferenceService(this->server) {}
int GRPCServerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", GRPC_SERVER_MODULE_NAME);
    std::vector<GrpcChannelArgument> channel_arguments;
    auto status = parseGrpcChannelArgs(config.grpcChannelArguments(), channel_arguments);
    if (!status.ok()) {
        SPDLOG_ERROR("grpc channel arguments passed in wrong format: {}", config.grpcChannelArguments());
        return EXIT_FAILURE;
    }

    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(GIGABYTE);
    builder.SetMaxSendMessageSize(GIGABYTE);
    builder.AddListeningPort(config.grpcBindAddress() + ":" + std::to_string(config.port()), grpc::InsecureServerCredentials());
    builder.RegisterService(&tfsPredictService);
    builder.RegisterService(&tfsModelService);
    builder.RegisterService(&kfsGrpcInferenceService);
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
    uint grpcServersCount = getGRPCServersCount(config);
    servers.reserve(grpcServersCount);
    SPDLOG_DEBUG("Starting gRPC servers: {}", grpcServersCount);

    if (!isPortAvailable(config.port())) {
        SPDLOG_ERROR("Failed to start gRPC server at " + config.grpcBindAddress() + ":" + std::to_string(config.port()));
        return EXIT_FAILURE;
    }
    for (uint i = 0; i < grpcServersCount; ++i) {
        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
        if (server == nullptr) {
            SPDLOG_ERROR("Failed to start gRPC server at " + config.grpcBindAddress() + ":" + std::to_string(config.port()));
            return EXIT_FAILURE;
        }
        servers.push_back(std::move(server));
    }
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", GRPC_SERVER_MODULE_NAME);
    // FIXME this should be reenabled when functional tests are switched to wait
    // for servablemanager module start log
    // #KFS_CLEANUP
    // SPDLOG_INFO("Server started on port {}", config.port());
    return EXIT_SUCCESS;
}

void GRPCServerModule::shutdown() {
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", GRPC_SERVER_MODULE_NAME);
    for (const auto& server : servers) {
        server->Shutdown();
        SPDLOG_INFO("Shutdown gRPC server");
    }
    servers.clear();
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", GRPC_SERVER_MODULE_NAME);
}

const GetModelMetadataImpl& GRPCServerModule::getTFSModelMetadataImpl() const {
    return this->tfsPredictService.getTFSModelMetadataImpl();
}
KFSInferenceServiceImpl& GRPCServerModule::getKFSGrpcImpl() const {
    return this->kfsGrpcInferenceService;
}
