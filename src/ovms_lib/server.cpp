//****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include "server.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
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
#include "grpcservermodule.hpp"
#include "http_server.hpp"
#include "kfs_grpc_inference_service.hpp"
#include "logging.hpp"
#include "model_service.hpp"
#include "modelmanager.hpp"
#include "prediction_service.hpp"
#include "profiler.hpp"
#include "servablemanagermodule.hpp"
#include "stringutils.hpp"
#include "version.hpp"

using grpc::ServerBuilder;

namespace ovms {
const std::string PROFILER_MODULE_NAME = "ProfilerModule";
const std::string GRPC_SERVER_MODULE_NAME = "GRPCServerModule";
const std::string HTTP_SERVER_MODULE_NAME = "HTTPServerModule";
const std::string SERVABLE_MANAGER_MODULE_NAME = "ServableManagerModule";
}  // namespace ovms
using namespace ovms;

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

Server& Server::instance() {
    static Server global;
    return global;
}

void logConfig(const Config& config) {
    std::string project_name(PROJECT_NAME);
    std::string project_version(PROJECT_VERSION);
    SPDLOG_INFO(project_name + " " + project_version);
    SPDLOG_INFO("OpenVINO backend {}", OPENVINO_NAME);
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
        SPDLOG_DEBUG("idle_sequence_cleanup: {}", config.idleSequenceCleanup());
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
    SPDLOG_DEBUG("sequence cleaner poll wait minutes: {}", config.sequenceCleanerPollWaitMinutes());
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

void installSignalHandlers(ovms::Server& server) {
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

static const int GIGABYTE = 1024 * 1024 * 1024;

ModuleState Module::getState() const {
    return state;
}

// TODO should replace all messages like
// start REST Server with start HTTP Server
// start Server with start gRPC server
// this should be synchronized with validation tests changes

class HTTPServerModule : public Module {
    std::unique_ptr<ovms::http_server> server;
    Server& ovmsServer;

public:
    HTTPServerModule(ovms::Server& ovmsServer) :
        ovmsServer(ovmsServer) {}
    int start(const ovms::Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        const std::string server_address = config.restBindAddress() + ":" + std::to_string(config.restPort());
        int workers = config.restWorkers() ? config.restWorkers() : 10;

        SPDLOG_INFO("Will start {} REST workers", workers);
        server = ovms::createAndStartHttpServer(config.restBindAddress(), config.restPort(), workers, this->ovmsServer);
        if (server != nullptr) {
            SPDLOG_INFO("Started REST server at {}", server_address);
        } else {
            SPDLOG_ERROR("Failed to start REST server at " + server_address);
            return EXIT_FAILURE;
        }
        state = ModuleState::INITIALIZED;
        return EXIT_SUCCESS;
    }
    void shutdown() override {
        if (server == nullptr)
            return;
        state = ModuleState::STARTED_SHUTDOWN;
        server->Terminate();
        server->WaitForTermination();
        SPDLOG_INFO("Shutdown HTTP server");
        state = ModuleState::SHUTDOWN;
    }
};

bool Server::isReady() const {
    std::shared_lock lock(modulesMtx);
    auto it = modules.find(SERVABLE_MANAGER_MODULE_NAME);
    if (it == modules.end())
        return false;
    if (ModuleState::INITIALIZED != it->second->getState())
        return false;
    return true;
}

bool Server::isLive() const {
    // TODO we might want at some time start REST only/ or respond with true only if both servers started if both are requested to start. This is to be resolved especially if we implement REST API for Kserver & potentially switch to check for starting specific module
    std::shared_lock lock(modulesMtx);
    auto it = modules.find(GRPC_SERVER_MODULE_NAME);
    if (it == modules.end())
        return false;
    if (ModuleState::INITIALIZED != it->second->getState())
        return false;
    return true;
}

ModuleState Server::getModuleState(const std::string& name) const {
    std::shared_lock lock(modulesMtx);
    auto it = modules.find(name);
    if (it == modules.end())
        return ModuleState::NOT_INITIALIZED;
    return it->second->getState();
}

const Module* Server::getModule(const std::string& name) const {
    std::shared_lock lock(modulesMtx);
    auto it = modules.find(name);
    if (it == modules.end())
        return nullptr;
    return it->second.get();
}

#ifdef MTR_ENABLED
class ProfilerModule : public Module {
    std::unique_ptr<Profiler> profiler;

public:
    ProfilerModule() = default;
    int start(const Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        auto profiler = std::make_unique<Profiler>(config.tracePath());
        if (!profiler.isInitialized()) {
            SPDLOG_ERROR("Cannot open file for profiler, --trace_path: {}", config.tracePath());
            return EXIT_FAILURE;
        }
        state = ModuleState::INITIALIZED;
        return EXIT_SUCCESS;
    }
    void shutdown() override {
#ifdef MTR_ENABLED
        state = ModuleState::STARTED_SHUTDOWN;
        profiler.reset();
        state = ModuleState::SHUTDOWN;
#endif
    }
};
#endif

void Server::setShutdownRequest(int i) {
    shutdown_request = i;
}

Server::~Server() = default;

std::unique_ptr<Module> Server::createModule(const std::string& name) {
#ifdef MTR_ENABLED
    if (name == PROFILER_MODULE_NAME)
        return std::make_unique<ProfilerModule>();
#endif
    if (name == GRPC_SERVER_MODULE_NAME)
        return std::make_unique<GRPCServerModule>(*this);
    if (name == HTTP_SERVER_MODULE_NAME)
        return std::make_unique<HTTPServerModule>(*this);
    if (name == SERVABLE_MANAGER_MODULE_NAME)
        return std::make_unique<ServableManagerModule>();
    return nullptr;
}

int Server::startModules(ovms::Config& config) {
    auto retCode = EXIT_SUCCESS;
    bool inserted = false;
    auto it = modules.end();
#if MTR_ENABLED
    {
        auto module = this->createModule(PROFILER_MODULE_NAME);
        std::unique_lock lock(modulesMtx);
        std::tie(it, inserted) = this->modules.emplace(PROFILER_MODULE_NAME, std::move(module));
    }
    retCode = modules.at(PROFILER_MODULE_NAME)->start(config);
    if (retCode)
        return retCode;
#endif
    {
        auto module = this->createModule(GRPC_SERVER_MODULE_NAME);
        std::unique_lock lock(modulesMtx);
        std::tie(it, inserted) = this->modules.emplace(GRPC_SERVER_MODULE_NAME, std::move(module));
    }

    if (!inserted)
        return EXIT_FAILURE;
    // if we ever decide not to start GRPC module then we need to implement HTTP responses without using grpc implementations
    retCode = it->second->start(config);
    if (retCode)
        return retCode;
    if (config.restPort() != 0) {
        {
            auto module = this->createModule(HTTP_SERVER_MODULE_NAME);
            std::unique_lock lock(modulesMtx);
            std::tie(it, inserted) = this->modules.emplace(HTTP_SERVER_MODULE_NAME, std::move(module));
        }
        retCode = it->second->start(config);
        if (retCode)
            return retCode;
    }
    {
        auto module = this->createModule(SERVABLE_MANAGER_MODULE_NAME);
        std::unique_lock lock(modulesMtx);
        std::tie(it, inserted) = this->modules.emplace(SERVABLE_MANAGER_MODULE_NAME, std::move(module));
    }
    retCode = it->second->start(config);
    return retCode;
}

void Server::shutdownModules(ovms::Config& config) {
    modules.at(GRPC_SERVER_MODULE_NAME)->shutdown();
    if (config.restPort() != 0)
        modules.at(HTTP_SERVER_MODULE_NAME)->shutdown();
    modules.at(SERVABLE_MANAGER_MODULE_NAME)->shutdown();
#ifdef MTR_ENABLED
    modules.at(PROFILER_MODULE_NAME)->shutdown();
#endif
    // FIXME we need to be able to quickly start grpc or start it without port
    // this is because the OS can have a delay between freeing up port before it can be requested and used again
    modules.clear();
}

int Server::start(int argc, char** argv) {
    ovms::Server& server = ovms::Server::instance();
    installSignalHandlers(server);
    try {
        auto& config = ovms::Config::instance().parse(argc, argv);
        configure_logger(config.logLevel(), config.logPath());
        logConfig(config);
        auto retCode = this->startModules(config);
        if (retCode)
            return retCode;

        while (!shutdown_request) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (shutdown_request == 2) {
            SPDLOG_ERROR("Illegal operation. OVMS started on unsupported device");
        }
        SPDLOG_INFO("Shutting down");
        this->shutdownModules(config);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Exception catch: {} - will now terminate.", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        SPDLOG_ERROR("Unknown exception catch - will now terminate.");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
