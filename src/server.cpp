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
#include <signal.h>
#include <stdlib.h>

#ifdef __linux__
#include <netinet/in.h>
#include <sys/socket.h>
#include <sysexits.h>
#elif _WIN32
#include <csignal>

#include <ntstatus.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#include <windows.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif

#include "capi_frontend/capimodule.hpp"
#include "capi_frontend/server_settings.hpp"
#include "cli_parser.hpp"
#include "config.hpp"
#include "config_export_module/config_export_module.hpp"
#include "grpcservermodule.hpp"
#include "http_server.hpp"
#include "httpservermodule.hpp"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "logging.hpp"
#include "metric_module.hpp"
#include "model_service.hpp"
#include "modelmanager.hpp"
#include "ovms_exit_codes.hpp"
#include "prediction_service.hpp"
#include "profiler.hpp"
#include "profilermodule.hpp"
#include "pull_module/hf_pull_model_module.hpp"
#include "servablemanagermodule.hpp"
#include "servables_config_manager_module/servablesconfigmanagermodule.hpp"
#include "stringutils.hpp"
#include "version.hpp"

#if (PYTHON_DISABLE == 0)
#include "python/pythoninterpretermodule.hpp"
#endif

using grpc::ServerBuilder;

namespace ovms {
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
    if (config.getServerSettings().hfSettings.pullHfModelMode) {
        SPDLOG_DEBUG("source_model: {}", config.getServerSettings().hfSettings.sourceModel);
        SPDLOG_DEBUG("model_repository_path: {}", config.getServerSettings().hfSettings.downloadPath);
        return;
    }
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
    SPDLOG_DEBUG("file system poll wait milliseconds: {}", config.filesystemPollWaitMilliseconds());
    SPDLOG_DEBUG("sequence cleaner poll wait minutes: {}", config.sequenceCleanerPollWaitMinutes());
    SPDLOG_DEBUG("list_models: {}", config.getServerSettings().listServables);
    SPDLOG_DEBUG("model_repository_path: {}", config.getServerSettings().hfSettings.downloadPath);
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

#ifdef __linux__

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
#elif _WIN32

static BOOL WINAPI onConsoleEvent(DWORD event) {
    switch (event) {
    case CTRL_C_EVENT:
        onInterrupt(SIGINT);
        return TRUE;
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        onTerminate(SIGTERM);
        return TRUE;
    default:
        return FALSE;
    }
}

static void installSignalHandlers() {
    SetConsoleCtrlHandler(onConsoleEvent, TRUE);
    signal(SIGINT, onInterrupt);
    signal(SIGTERM, onTerminate);
    signal(SIGILL, onIllegal);
}

#endif

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

bool Server::isLive(const std::string& moduleName) const {
    std::shared_lock lock(modulesMtx);
    if (modules.size() == 0) {
        return false;
    }
    auto it = modules.find(moduleName);
    if (it == modules.end()) {
        return false;
    }
    if (ModuleState::INITIALIZED != it->second->getState()) {
        return false;
    }
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
#if (PYTHON_DISABLE == 0)
    if (name == PYTHON_INTERPRETER_MODULE_NAME)
        return std::make_unique<PythonInterpreterModule>();
#endif
    if (name == METRICS_MODULE_NAME)
        return std::make_unique<MetricModule>();
    if (name == CAPI_MODULE_NAME)
        return std::make_unique<CAPIModule>(*this);
    if (name == HF_MODEL_PULL_MODULE_NAME)
        return std::make_unique<HfPullModelModule>();
    if (name == SERVABLES_CONFIG_MANAGER_MODULE_NAME)
        return std::make_unique<ServablesConfigManagerModule>();
    if (name == CONFIG_EXPORT_MODULE_NAME)
        return std::make_unique<ConfigExportModule>();
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
        return status;

#define GET_MODULE(MODULE_NAME, IT_NAME)                                                              \
    {                                                                                                 \
        std::shared_lock lock(modulesMtx);                                                            \
        IT_NAME = modules.find(MODULE_NAME);                                                          \
        if (IT_NAME == modules.end()) {                                                               \
            return Status(StatusCode::INTERNAL_ERROR, std::string("Could not find: ") + MODULE_NAME); \
        }                                                                                             \
    }

Status Server::startModules(ovms::Config& config) {
    // The order of starting modules is slightly different from inserting modules
    // due to dependency of modules on each other during runtime
    // To avoid unnecessary runtime calls in eg. prediction we have different order
    // of modules creation than start
    // CAPI module is required for MP to work
    // CAPI should start after SERVABLE is added
    // HTTP depends on GRPC, SERVABLE, METRICS
    // GRPC depends on SERVABLE
    // SERVABLE depends on metrics, python
    // while we want to start the server as quickly as possible to respond with liveness probe
    // that's why we delay starting the servable until the very end while we need to create it before
    // GRPC & REST
    Status status;
    bool inserted = false;
    auto it = modules.end();

    if (config.getServerSettings().listServables) {
        INSERT_MODULE(SERVABLES_CONFIG_MANAGER_MODULE_NAME, it);
        START_MODULE(it);
        return status;
    }
    if (config.getServerSettings().hfSettings.pullHfModelMode || config.getServerSettings().hfSettings.pullHfAndStartModelMode) {
        INSERT_MODULE(HF_MODEL_PULL_MODULE_NAME, it);
        START_MODULE(it);
        if (!status.ok()) {
            return status;
        }
        auto hfModule = dynamic_cast<const HfPullModelModule*>(it->second.get());
        status = hfModule->clone();
        // Return from modules only in --pull mode, otherwise start the rest of modules
        if (config.getServerSettings().hfSettings.pullHfModelMode)
            return status;
    }
    if (config.getServerSettings().exportConfigType != UNKNOWN_MODEL) {
        INSERT_MODULE(CONFIG_EXPORT_MODULE_NAME, it);
        START_MODULE(it);
    }

#if (PYTHON_DISABLE == 0)
    if (config.getServerSettings().withPython) {
        INSERT_MODULE(PYTHON_INTERPRETER_MODULE_NAME, it);
        START_MODULE(it);
    }
#endif
#if MTR_ENABLED
    INSERT_MODULE(PROFILER_MODULE_NAME, it);
    START_MODULE(it);
#endif
    // It is required to have the metrics module, it is used by ServableManagerModule.
    INSERT_MODULE(METRICS_MODULE_NAME, it);
    START_MODULE(it);

    // we need servable module during GRPC/HTTP requests so create it here
    // but start it later to quickly respond with liveness probe
    INSERT_MODULE(SERVABLE_MANAGER_MODULE_NAME, it);
    INSERT_MODULE(CAPI_MODULE_NAME, it);
    START_MODULE(it);
    INSERT_MODULE(GRPC_SERVER_MODULE_NAME, it);
    START_MODULE(it);
    // if we ever decide not to start GRPC module then we need to implement HTTP responses without using grpc implementations
    if (config.restPort() != 0) {
        INSERT_MODULE(HTTP_SERVER_MODULE_NAME, it);
        START_MODULE(it);
    }
    GET_MODULE(SERVABLE_MANAGER_MODULE_NAME, it);
    START_MODULE(it);
#if (PYTHON_DISABLE == 0)
    if (config.getServerSettings().withPython) {
        GET_MODULE(PYTHON_INTERPRETER_MODULE_NAME, it);
        auto pythonModule = dynamic_cast<const PythonInterpreterModule*>(it->second.get());
        pythonModule->releaseGILFromThisThread();
    }
#endif
    return status;
}

void Server::ensureModuleShutdown(const std::string& name) {
    std::shared_lock lock(modulesMtx);
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
    ensureModuleShutdown(HF_MODEL_PULL_MODULE_NAME);
    ensureModuleShutdown(GRPC_SERVER_MODULE_NAME);
    ensureModuleShutdown(HTTP_SERVER_MODULE_NAME);
    ensureModuleShutdown(SERVABLE_MANAGER_MODULE_NAME);
    ensureModuleShutdown(PROFILER_MODULE_NAME);
#if (PYTHON_DISABLE == 0)
    if (ovms::Config::instance().getServerSettings().withPython) {
        ensureModuleShutdown(PYTHON_INTERPRETER_MODULE_NAME);
    }
#endif
    // we need to be able to quickly start grpc or start it without port
    // this is because the OS can have a delay between freeing up port before it can be requested and used again
    modules.clear();
}

static int statusToExitCode(const Status& status) {
    if (status.ok()) {
        return OVMS_EX_OK;
    } else if (status == StatusCode::OPTIONS_USAGE_ERROR) {
        return OVMS_EX_USAGE;
    }
    return OVMS_EX_FAILURE;
}

// OVMS Start
int Server::start(int argc, char** argv) {
    installSignalHandlers();
    int result = OVMS_EX_OK;

    try {
        CLIParser parser;
        ServerSettingsImpl serverSettings;
        ModelsSettingsImpl modelsSettings;
        parser.parse(argc, argv);
        parser.prepare(&serverSettings, &modelsSettings);

        Status ret = start(&serverSettings, &modelsSettings);
        ModulesShutdownGuard shutdownGuard(*this);
        if (!ret.ok()) {
            return statusToExitCode(ret);
        }
        while (!shutdown_request &&
               !serverSettings.hfSettings.pullHfModelMode &&
               !serverSettings.listServables &&
               serverSettings.exportConfigType == UNKNOWN_MODEL) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (shutdown_request == 2) {
            SPDLOG_ERROR("Illegal operation. OVMS started on unsupported device");
        }
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception; {}", e.what());
        result = OVMS_EX_FAILURE;
        return result;
    }

    return EXIT_SUCCESS;
}

// C-API Start
Status Server::start(ServerSettingsImpl* serverSettings, ModelsSettingsImpl* modelsSettings) {
    try {
        std::unique_lock lock{this->startMtx, std::defer_lock};
        auto locked = lock.try_lock();
        if (!locked) {
            SPDLOG_ERROR("Cannot start OVMS - server is already starting");
            return StatusCode::SERVER_ALREADY_STARTING;
        }
        std::unique_lock lockModules(modulesMtx);
        if (!modules.empty()) {
            SPDLOG_ERROR("Cannot start OVMS - server is already live");
            return StatusCode::SERVER_ALREADY_STARTED;
        }
        lockModules.unlock();
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
