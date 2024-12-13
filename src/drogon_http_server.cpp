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

#include <chrono>
#include <limits>
#include <thread>
#include <utility>

#include <drogon/drogon.h>

#include "logging.hpp"
#include "mediapipe/framework/port/threadpool.h"

namespace ovms {

static const int THREAD_COUNT_FOR_DROGON_LISTENER = 3;  // TODO: how many is best perf?

DrogonHttpServer::DrogonHttpServer(size_t num_workers, int port, const std::string& address) :
    num_workers(num_workers),
    pool(std::make_unique<mediapipe::ThreadPool>("DrogonThreadPool", num_workers)),
    port(port),
    address(address) {
    SPDLOG_DEBUG("Starting http thread pool ({} threads)", num_workers);
    pool->StartWorkers();  // this is for actual workload which is scheduled to other threads than drogon's internal listener threads
    SPDLOG_DEBUG("Thread pool started");
}

Status DrogonHttpServer::startAcceptingRequests() {
    SPDLOG_DEBUG("DrogonHttpServer::startAcceptingRequests()");

    // OVMS has its own sigterm handling
    drogon::app().disableSigtermHandling();

    drogon::app().setDefaultHandler([this](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        this->pool->Schedule([this, req, callback = std::move(callback)]() mutable {
            try {
                this->dispatcher(req, std::move(callback));
            } catch (...) {
                SPDLOG_DEBUG("Exception caught in REST request handler");
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
                callback(resp);
            }
        });
    });

    // Should never happen
    if (drogon::app().isRunning()) {
        SPDLOG_ERROR("Drogon is already running");
        throw 42;
    }

    pool->Schedule(
        [address = this->address, port = this->port] {
            static int numberOfLaunchesInApplication = 0;
            numberOfLaunchesInApplication++;
            if (numberOfLaunchesInApplication > 1) {
                SPDLOG_ERROR("Drogon was already started, cannot start it again");
                return;
            }
            SPDLOG_DEBUG("Starting to listen on port {}", port);
            try {
                drogon::app()
                    .setThreadNum(THREAD_COUNT_FOR_DROGON_LISTENER)  // threads only for accepting requests, the workload is on separate thread pool
                    .setIdleConnectionTimeout(0)
                    .setClientMaxBodySize(1024 * 1024 * 1024)  // 1GB
                    .setClientMaxMemoryBodySize(std::numeric_limits<size_t>::max())
                    // .setMaxConnectionNum(100000)  // default is 100000
                    // .setMaxConnectionNumPerIP(0)  // default is 0=unlimited
                    // .setServerHeaderField("OpenVINO Model Server")
                    .enableServerHeader(false)
                    .enableDateHeader(false)
                    .addListener(address, port)
                    .run();
            } catch (...) {
                SPDLOG_ERROR("Exception occurred during drogon::run()");
            }
            SPDLOG_DEBUG("drogon::run() exits normally");
        });

    // wait until drogon becomes ready
    size_t runningCheckIntervalMillisec = 50;
    size_t maxTotalRunningCheckTimeMillisec = 1000;
    size_t maxChecks = maxTotalRunningCheckTimeMillisec / runningCheckIntervalMillisec;
    while (!drogon::app().isRunning()) {
        SPDLOG_DEBUG("Waiting for drogon to become ready on port {}...", port);
        if (maxChecks == 0) {
            SPDLOG_DEBUG("Waiting for drogon server launch timed out");
            return StatusCode::INTERNAL_ERROR;
        }
        maxChecks--;
        std::this_thread::sleep_for(std::chrono::milliseconds(runningCheckIntervalMillisec));
    }
    SPDLOG_INFO("REST server listening on port {} with {} threads", port, num_workers);
    return StatusCode::OK;
}

void DrogonHttpServer::terminate() {
    SPDLOG_DEBUG("DrogonHttpServer::terminate()");

    // Should never happen
    if (!drogon::app().isRunning()) {
        SPDLOG_DEBUG("Drogon is not running");
        throw 42;
    }

    drogon::app().quit();
    pool.reset();  // waits for all worker threads to finish
}

void DrogonHttpServer::registerRequestDispatcher(
    std::function<void(
        const drogon::HttpRequestPtr&,
        std::function<void(const drogon::HttpResponsePtr&)>&&)>
        dispatcher) {
    this->dispatcher = std::move(dispatcher);
}

mediapipe::ThreadPool& DrogonHttpServer::getPool() {
    return *pool;
}

}  // namespace ovms
