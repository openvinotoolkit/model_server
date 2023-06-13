//****************************************************************************
// Copyright 2020-2022 Intel Corporation
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
#include <memory>
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
#include <sysexits.h>
#include <unistd.h>

#include "capi_frontend/server_settings.hpp"
#include "cli_parser.hpp"
#include "config.hpp"
#include "grpcservermodule.hpp"
#include "http_server.hpp"
#include "httpservermodule.hpp"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "logging.hpp"
#include "metric_module.hpp"
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
const std::string METRICS_MODULE_NAME = "MetricsModule";

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

Server& Server::instance() {
    static Server global;
    return global;
}

static void logConfig(const Config& config) {
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
        SPDLOG_DEBUG("metrics_enabled: {}", config.metricsEnabled());
        SPDLOG_DEBUG("metrics_list: {}", config.metricsList());
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

static void onInterrupt(int status) {
    shutdown_request = 1;
}

static void onTerminate(int status) {
    shutdown_request = 1;
}

static void onIllegal(int status) {
    shutdown_request = 2;
}

static void installSignalHandlers() {
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

ModuleState Module::getState() const {
    return state;
}

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
    // we might want at some time start REST only/ or respond with true only if both servers started if both are requested to start
    // This is to be resolved especially if we implement REST API for Kserver & potentially switch to check for starting specific module
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
    Status start(const ovms::Config& config) override { return StatusCode::OK; }

    int start(const Config& config) override {
        state = ModuleState::STARTED_INITIALIZE;
        SPDLOG_INFO("{} starting", PROFILER_MODULE_NAME);
        this->profiler = std::make_unique<Profiler>(config.tracePath());
        if (!this->profiler) {
            return EXIT_FAILURE;
        }
        if (!this->profiler->isInitialized()) {
            SPDLOG_ERROR("Cannot open file for profiler, --trace_path: {}", config.tracePath());
            return EXIT_FAILURE;
        }
        state = ModuleState::INITIALIZED;
        SPDLOG_INFO("{} started", PROFILER_MODULE_NAME);
        return EXIT_SUCCESS;
    }
    void shutdown() override {
#ifdef MTR_ENABLED
        if (state == ModuleState::SHUTDOWN)
            return;
        state = ModuleState::STARTED_SHUTDOWN;
        SPDLOG_INFO("{} shutting down", PROFILER_MODULE_NAME);
        profiler.reset();
        state = ModuleState::SHUTDOWN;
        SPDLOG_INFO("{} shutdown", PROFILER_MODULE_NAME);
#endif
    }
    ~ProfilerModule() {
        this->shutdown();
    }
};
#endif

void Server::setShutdownRequest(int i) {
    shutdown_request = i;
}

Server::~Server() {
    this->shutdownModules();
}

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
        return std::make_unique<ServableManagerModule>(*this);
    if (name == METRICS_MODULE_NAME)
        return std::make_unique<MetricModule>();
    return nullptr;
}

#define INSERT_MODULE(MODULE_NAME, IT_NAME)                                                  \
    {                                                                                        \
        auto module = this->createModule(MODULE_NAME);                                       \
        std::unique_lock lock(modulesMtx);                                                   \
        std::tie(IT_NAME, inserted) = this->modules.emplace(MODULE_NAME, std::move(module)); \
    }                                                                                        \
    if (!inserted)                                                                           \
    return Status(StatusCode::MODULE_ALREADY_INSERTED, MODULE_NAME)

#define START_MODULE(IT_NAME)                \
    status = IT_NAME->second->start(config); \
    if (!status.ok())                        \
    return status

Status Server::startModules(ovms::Config& config) {
    // The order of starting modules is slightly different from inserting modules
    // due to dependency of modules on each other during runtime
    // To avoid unnecessary runtime calls in eg. prediction we have different order
    // of modules creation than start
    // HTTP depends on GRPC, SERVABLE, METRICS
    // GRPC depends on SERVABLE
    // SERVABLE depends on metrics
    // while we want to start the server as quickly as possible to respond with liveness probe
    // thats why we delay starting the servable until the very end while we need to create it before
    // GRPC & REST
    Status status;
    bool inserted = false;
    auto it = modules.end();
#if MTR_ENABLED
    INSERT_MODULE(PROFILER_MODULE_NAME, it);
    START_MODULE(it);
#endif
    // It is required to have the metrics module, it is used by ServableManagerModule.
    INSERT_MODULE(METRICS_MODULE_NAME, it);
    START_MODULE(it);

    // we need servable module during GRPC/HTTP requests so create it here
    // but start it later to quickly respond with liveness probe
    auto itServable = modules.end();
    INSERT_MODULE(SERVABLE_MANAGER_MODULE_NAME, itServable);
    auto itGrpc = modules.end();
    INSERT_MODULE(GRPC_SERVER_MODULE_NAME, itGrpc);
    START_MODULE(itGrpc);
    // if we ever decide not to start GRPC module then we need to implement HTTP responses without using grpc implementations
    auto itHttp = modules.end();
    if (config.restPort() != 0) {
        INSERT_MODULE(HTTP_SERVER_MODULE_NAME, itHttp);
        START_MODULE(itHttp);
    }
    START_MODULE(itServable);
    return status;
}

void Server::ensureModuleShutdown(const std::string& name) {
    auto it = modules.find(name);
    if (it != modules.end())
        it->second->shutdown();
}

class ModulesShutdownGuard {
    Server& server;

public:
    ModulesShutdownGuard(Server& server) :
        server(server) {}
    ~ModulesShutdownGuard() {
        this->server.shutdownModules();
    }
};

void Server::shutdownModules() {
    // we want very precise order of modules shutdown
    // first we should stop incoming new requests
    ensureModuleShutdown(GRPC_SERVER_MODULE_NAME);
    ensureModuleShutdown(HTTP_SERVER_MODULE_NAME);
    ensureModuleShutdown(SERVABLE_MANAGER_MODULE_NAME);
    ensureModuleShutdown(PROFILER_MODULE_NAME);
    // we need to be able to quickly start grpc or start it without port
    // this is because the OS can have a delay between freeing up port before it can be requested and used again
    modules.clear();
}

static int statusToExitCode(const Status& status) {
    if (status.ok()) {
        return EX_OK;
    } else if (status == StatusCode::OPTIONS_USAGE_ERROR) {
        return EX_USAGE;
    }
    return EXIT_FAILURE;
}

// OVMS Start
int Server::start(int argc, char** argv) {
    installSignalHandlers();
    CLIParser parser;
    ServerSettingsImpl serverSettings;
    ModelsSettingsImpl modelsSettings;
    parser.parse(argc, argv);
    parser.prepare(&serverSettings, &modelsSettings);
    Status ret = start(&serverSettings, &modelsSettings);
    ModulesShutdownGuard shutdownGuard(*this);
    if (!ret.ok()) {
        // Handle OVMS main() return code
        return statusToExitCode(ret);
    }
    while (!shutdown_request) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (shutdown_request == 2) {
        SPDLOG_ERROR("Illegal operation. OVMS started on unsupported device");
    }
    SPDLOG_INFO("Shutting down");
    return EXIT_SUCCESS;
}

// C-API Start
Status Server::start(ServerSettingsImpl* serverSettings, ModelsSettingsImpl* modelsSettings) {
    try {
        if (this->isLive()) {
            SPDLOG_ERROR("Cannot start OVMS - server is already live");
            return StatusCode::SERVER_ALREADY_STARTED;
        }
        auto& config = ovms::Config::instance();
        if (!config.parse(serverSettings, modelsSettings))
            return StatusCode::OPTIONS_USAGE_ERROR;
        configure_logger(config.logLevel(), config.logPath());
        logConfig(config);
        return this->startModules(config);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Exception catch: {} - will now terminate.", e.what());
        return Status(StatusCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        SPDLOG_ERROR("Unknown exception catch - will now terminate.");
        return StatusCode::INTERNAL_ERROR;
    }
}
}  // namespace ovms
