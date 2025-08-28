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
#include "httpservermodule.hpp"

#include <sstream>
#include <string>
#include <utility>

#include <curl/curl.h>

#include "config.hpp"
#include "http_server.hpp"
#include "logging.hpp"
#include "server.hpp"
#include "status.hpp"

namespace ovms {
HTTPServerModule::HTTPServerModule(ovms::Server& ovmsServer) :
    ovmsServer(ovmsServer) {}
Status HTTPServerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", HTTP_SERVER_MODULE_NAME);
    const std::string server_address = config.restBindAddress() + ":" + std::to_string(config.restPort());
    int workers = config.restWorkers() ? config.restWorkers() : 10;

    SPDLOG_INFO("Will start {} REST workers", workers);
    SPDLOG_INFO("Hello World");
#if (USE_DROGON == 0)
    netHttpServer = ovms::createAndStartNetHttpServer(config.restBindAddress(), config.restPort(), workers, this->ovmsServer);
    if (netHttpServer == nullptr) {
        std::stringstream ss;
        ss << "at " << server_address;
        auto status = Status(StatusCode::FAILED_TO_START_REST_SERVER, ss.str());
        SPDLOG_ERROR(status.string());
        return status;
    }
#else
    drogonServer = ovms::createAndStartDrogonHttpServer(config.restBindAddress(), config.restPort(), workers, this->ovmsServer);
    if (drogonServer == nullptr) {
        std::stringstream ss;
        ss << "at " << server_address;
        auto status = Status(StatusCode::FAILED_TO_START_REST_SERVER, ss.str());
        SPDLOG_ERROR(status.string());
        return status;
    }
#endif
    curl_global_init(CURL_GLOBAL_ALL);
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", HTTP_SERVER_MODULE_NAME);
    SPDLOG_INFO("Started REST server at {}", server_address);
    return StatusCode::OK;
}
void HTTPServerModule::shutdown() {
#if (USE_DROGON == 0)
    if (netHttpServer == nullptr)
        return;
#else
    if (drogonServer == nullptr)
        return;
#endif
    SPDLOG_INFO("{} shutting down", HTTP_SERVER_MODULE_NAME);
    state = ModuleState::STARTED_SHUTDOWN;
#if (USE_DROGON == 0)
    netHttpServer->Terminate();
    netHttpServer->WaitForTermination();
    netHttpServer.reset();
#else
    drogonServer->terminate();
    drogonServer.reset();
#endif
    curl_global_cleanup();
    SPDLOG_INFO("Shutdown HTTP server");
    state = ModuleState::SHUTDOWN;
}

HTTPServerModule::~HTTPServerModule() {
    if (state != ModuleState::SHUTDOWN)
        this->shutdown();
}
}  // namespace ovms
