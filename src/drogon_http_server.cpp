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

#include <mutex>
#include <condition_variable>
#include <drogon/drogon.h>
#include <chrono>

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

Status DrogonHttpServer::startAcceptingRequests() {
    SPDLOG_DEBUG("///////////////////////////// DrogonHttpServer::startAcceptingRequests()");

    drogon::app().disableSigtermHandling();

    drogon::app().setDefaultHandler([this](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        this->pool_->Schedule([this, req, callback = std::move(callback)]() mutable {
            this->dispatcher_(req, std::move(callback));
        });

        // //this->dispatcher_(req, std::move(callback));  // callback
    });

    if (drogon::app().isRunning()) {
        SPDLOG_DEBUG("Drogon is already running");
        throw 42;
    }

    pool_->Schedule(
        [a = this->address_, p = this->port_] {
            static int x = 0;
            x++;
            if (x > 1) {
                // TODO
                SPDLOG_DEBUG("Drogon was already started, cannot start it again");
                return;
            }
            SPDLOG_DEBUG("Running Drogon app for the {} time {} {} {}", x, drogon::app().isRunning(), a, p);
            drogon::app()
                //.setThreadNum(this->pool_->num_threads())  // too many threads?
                .setThreadNum(3)  // threads only for accepting requests, the workload is on separate thread pool anyway
                .setIdleConnectionTimeout(0)
                .addListener(a, p)
                .run();
        });
    // wait until drogon becomes ready drogon::app().isRunning()
    SPDLOG_DEBUG("Waiting until drogon becomes ready...");
    int i = 0;
    while (!drogon::app().isRunning()) {
        i++;
        SPDLOG_DEBUG("Waiting until drogon becomes ready...");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // TODO
        if (i > 10) {
            SPDLOG_DEBUG("Waiting for drogon timed out");
            return StatusCode::INTERNAL_ERROR;
        }
    }
    SPDLOG_DEBUG("Drogon is ready");
    return StatusCode::OK;
}

void DrogonHttpServer::terminate() {
    if (!drogon::app().isRunning()) {
        SPDLOG_DEBUG("Drogon is not running");
        throw 42;
    }
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
