//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "drogon_http_server.hpp"

#include <drogon/drogon.h>

#include "logging.hpp"
#include "mediapipe/framework/port/threadpool.h"

namespace ovms {

DrogonHttpServer::DrogonHttpServer(size_t num_workers, int port, const std::string& address) :
    pool_(std::make_unique<mediapipe::ThreadPool>("DrogonThreadPool", num_workers)),
    port_(port),
    address_(address) {
    SPDLOG_DEBUG("Starting ThreadPool with {} workers", num_workers);
    pool_->StartWorkers();  // this is for listener and also for streaming outputs
    SPDLOG_DEBUG("ThreadPool started");
}

void DrogonHttpServer::startAcceptingRequests() {
    SPDLOG_DEBUG("///////////////////////////// DrogonHttpServer::startAcceptingRequests()");

    drogon::app().disableSigtermHandling();

    drogon::app().setDefaultHandler([this](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        this->pool_->Schedule([this, req, callback = std::move(callback)]() mutable {
            this->dispatcher_(req, std::move(callback));
        });

        // //this->dispatcher_(req, std::move(callback));  // callback
    });

    pool_->Schedule(
        [this] {
            SPDLOG_DEBUG("Running Drogon app");
            drogon::app()
                //.setThreadNum(this->pool_->num_threads())  // too many threads?
                .setThreadNum(3)  // threads only for accepting requests, the workload is on separate thread pool anyway
                .setIdleConnectionTimeout(0)
                .addListener(this->address_, this->port_)
                .run();
        });
}

void DrogonHttpServer::terminate() {
    SPDLOG_DEBUG("///////////////////////////// DrogonHttpServer::terminate() start");
    drogon::app().quit();
    pool_.reset();  // waits for all worker threads to finish
    SPDLOG_DEBUG("///////////////////////////// DrogonHttpServer::terminate() stop");
}

void DrogonHttpServer::registerRequestDispatcher(
    std::function<void(
        const drogon::HttpRequestPtr&,
        std::function<void(const drogon::HttpResponsePtr&)>&&)>
        dispatcher) {
    SPDLOG_DEBUG("///////////////////////////// DrogonHttpServer::registerRequestDispatcher()");
    dispatcher_ = std::move(dispatcher);
}

mediapipe::ThreadPool& DrogonHttpServer::getPool() {
    return *pool_;
}

}  // namespace ovms
