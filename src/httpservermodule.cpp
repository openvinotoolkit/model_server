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
    // server = ovms::createAndStartHttpServer(config.restBindAddress(), config.restPort(), 2/*workers*/, this->ovmsServer);
    // if (server == nullptr) {
    //     std::stringstream ss;
    //     ss << "at " << server_address;
    //     auto status = Status(StatusCode::FAILED_TO_START_REST_SERVER, ss.str());
    //     SPDLOG_ERROR(status.string());
    //     return status;
    // }
    drogonServer = ovms::createAndStartDrogonHttpServer(config.restBindAddress(), config.restPort(), workers, this->ovmsServer);
    // if (server == nullptr) {
    //     std::stringstream ss;
    //     ss << "at " << server_address;
    //     auto status = Status(StatusCode::FAILED_TO_START_REST_SERVER, ss.str());
    //     SPDLOG_ERROR(status.string());
    //     return status;
    // }
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", HTTP_SERVER_MODULE_NAME);
    SPDLOG_INFO("Started REST server at {}", server_address);
    return StatusCode::OK;
}
void HTTPServerModule::shutdown() {
    //if (server == nullptr)
    //    return;
    SPDLOG_INFO("{} shutting down", HTTP_SERVER_MODULE_NAME);
    state = ModuleState::STARTED_SHUTDOWN;
    //server->Terminate();
    //server->WaitForTermination();
    drogonServer->terminate();
    server.reset();
    SPDLOG_INFO("Shutdown HTTP server");
    state = ModuleState::SHUTDOWN;
}

HTTPServerModule::~HTTPServerModule() {
    if (state != ModuleState::SHUTDOWN)
        this->shutdown();
}
}  // namespace ovms
