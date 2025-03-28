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
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <signal.h>
#include <stdlib.h>
#ifdef __linux__
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#elif _WIN32
#include <winsock2.h>
#endif

#include "config.hpp"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "logging.hpp"
#include "model_service.hpp"
#include "modelmanager.hpp"
#include "prediction_service.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "stringutils.hpp"
#include "systeminfo.hpp"
#include "version.hpp"

using grpc::ServerBuilder;

namespace ovms {
static const int GIGABYTE = 1024 * 1024 * 1024;
// Default server shutdown deadline set to 5 seconds,
// so it happens before docker container graceful stop.
static const int SERVER_SHUTDOWN_DEADLINE_SECONDS = 5;

#ifdef __linux__
bool GRPCServerModule::isPortAvailable(uint64_t port) {
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
#else  //  not __linux__

struct WSAStartupCleanupGuard {
    ~WSAStartupCleanupGuard() {
        WSACleanup();
    }
};
struct SocketOpenCloseGuard {
    SOCKET socket;
    SocketOpenCloseGuard(SOCKET socket) :
        socket(socket) {}
    ~SocketOpenCloseGuard() {
        closesocket(socket);
    }
};
bool GRPCServerModule::isPortAvailable(uint64_t port) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        SPDLOG_ERROR("WSAStartup error.");
        return false;
    }
    WSAStartupCleanupGuard wsaGuard;
    // Create a socket
    this->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (this->sock == INVALID_SOCKET) {
        SPDLOG_ERROR("INVALID_SOCKET error.");
        return false;
    }

    // Bind to port
    sockaddr_in addr;
    addr.sin_family = AF_INET;
#pragma warning(disable : 4996)
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(port);
    SocketOpenCloseGuard socketGuard(this->sock);
    if (bind(this->sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        SPDLOG_ERROR("Bind port {} error: {}", port, WSAGetLastError());
        return false;
    }

    // Set SO_EXCLUSIVEADDRUSE  options

    bool bOptVal = true;
    int bOptLen = sizeof(bool);
    int iResult = 0;

    iResult = setsockopt(this->sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (char*)&bOptVal, bOptLen);
    if (iResult == SOCKET_ERROR) {
        SPDLOG_ERROR("setsockopt for SO_EXCLUSIVEADDRUSE failed with error: {}\n", WSAGetLastError());
        return false;
    }
    return true;
}
#endif  //  not __linux__

static Status setDefaultGrpcChannelArgs(std::map<std::string, std::string>& result) {
    uint16_t cores = getCoreCount();
    result["grpc.max_concurrent_streams"] = std::to_string(cores);  // concurrent streams from a single client set to #cores by default
    return StatusCode::OK;
}

// Parses a comma separated list of gRPC channel arguments into list of
// ChannelArgument.
static Status parseGrpcChannelArgs(const std::string& channel_arguments_str, std::map<std::string, std::string>& result) {
    const std::vector<std::string> channel_arguments = tokenize(channel_arguments_str, ',');

    for (const std::string& channel_argument : channel_arguments) {
        std::vector<std::string> key_val = tokenize(channel_argument, '=');
        if (key_val.size() != 2) {
            return Status(StatusCode::GRPC_CHANNEL_ARG_WRONG_FORMAT, channel_arguments_str);
        }
        erase_spaces(key_val[0]);
        erase_spaces(key_val[1]);
        result[key_val[0]] = key_val[1];
    }

    return StatusCode::OK;
}

static uint32_t getGRPCServersCount(const ovms::Config& config) {
    const char* environmentVariableBuffer = std::getenv("GRPC_SERVERS");
    if (environmentVariableBuffer) {
        auto result = stou32(environmentVariableBuffer);
        if (result && result.value() > 0) {
            return result.value();
        }
    }

    return std::max<uint32_t>(1, config.grpcWorkers());
}

GRPCServerModule::~GRPCServerModule() {
    this->shutdown();
}

GRPCServerModule::GRPCServerModule(Server& server) :
    server(server),
    tfsPredictService(this->server),
    tfsModelService(this->server),
    kfsGrpcInferenceService(this->server) {}

Status GRPCServerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", GRPC_SERVER_MODULE_NAME);
    if (config.port() == 0) {
        // due to HTTP reusing gRPC we still need to have gRPC module initialized.
        state = ModuleState::INITIALIZED;
        SPDLOG_INFO("{} started", GRPC_SERVER_MODULE_NAME);
        SPDLOG_INFO("Port was not set. GRPC server will not be started.");
        return StatusCode::OK;
    }

    std::map<std::string, std::string> channel_arguments;
    auto status = setDefaultGrpcChannelArgs(channel_arguments);
    if (!status.ok()) {
        SPDLOG_ERROR(status.string());
        return status;
    }
    status = parseGrpcChannelArgs(config.grpcChannelArguments(), channel_arguments);
    if (!status.ok()) {
        SPDLOG_ERROR(status.string());
        return status;
    }

    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(GIGABYTE);
    builder.SetMaxSendMessageSize(GIGABYTE);
    builder.AddListeningPort(config.grpcBindAddress() + ":" + std::to_string(config.port()), grpc::InsecureServerCredentials());
    builder.RegisterService(&tfsPredictService);
    builder.RegisterService(&tfsModelService);
    builder.RegisterService(&kfsGrpcInferenceService);
    for (auto& [name, value] : channel_arguments) {
        // gRPC accept arguments of two types, int and string. We will attempt to
        // parse each arg as int and pass it on as such if successful. Otherwise we
        // will pass it as a string. gRPC will log arguments that were not accepted.
        SPDLOG_DEBUG("setting grpc channel argument {}: {}", name, value);
        try {
            int i = std::stoi(value);
            builder.AddChannelArgument(name, i);
        } catch (std::invalid_argument const&) {
            builder.AddChannelArgument(name, value);
        } catch (std::out_of_range const&) {
            SPDLOG_WARN("Out of range parameter {} : {}", name, value);
        }
    }
    ::grpc::ResourceQuota resource_quota;
    if (config.grpcMaxThreads() != 0) {
        resource_quota.SetMaxThreads(config.grpcMaxThreads());
        SPDLOG_DEBUG("setting grpc MaxThreads ResourceQuota {}", config.grpcMaxThreads());
    }
    if (config.grpcMemoryQuota() != 0) {
        resource_quota.Resize(config.grpcMemoryQuota());
        SPDLOG_DEBUG("setting grpc Memory ResourceQuota {}", config.grpcMemoryQuota());
    }
    if ((config.grpcMemoryQuota() != 0) || (config.grpcMaxThreads() != 0)) {
        builder.SetResourceQuota(resource_quota);
    }
    uint32_t grpcServersCount = getGRPCServersCount(config);
    servers.reserve(grpcServersCount);
    SPDLOG_DEBUG("Starting gRPC servers: {}", grpcServersCount);

    if (!isPortAvailable(config.port())) {
        std::stringstream ss;
        ss << "at " << config.grpcBindAddress() << ":" << std::to_string(config.port()) << " - port is busy";
        status = Status(StatusCode::FAILED_TO_START_GRPC_SERVER, ss.str());
        SPDLOG_ERROR(status.string());
        return status;
    }
    for (uint32_t i = 0; i < grpcServersCount; ++i) {
        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
        if (server == nullptr) {
            std::stringstream ss;
            ss << "at " << config.grpcBindAddress() << ":" << std::to_string(config.port());
            status = Status(StatusCode::FAILED_TO_START_GRPC_SERVER, ss.str());
            SPDLOG_ERROR(status.string());
            return status;
        }
        servers.push_back(std::move(server));
    }
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", GRPC_SERVER_MODULE_NAME);
    SPDLOG_INFO("Started gRPC server on port {}", config.port());
    return StatusCode::OK;
}

void GRPCServerModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", GRPC_SERVER_MODULE_NAME);
    std::chrono::time_point serverDeadline = std::chrono::system_clock::now() +
                                             std::chrono::seconds(SERVER_SHUTDOWN_DEADLINE_SECONDS);
    for (const auto& server : servers) {
        server->Shutdown(serverDeadline);
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
}  // namespace ovms
