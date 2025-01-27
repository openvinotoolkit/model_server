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

DrogonHttpServer::DrogonHttpServer(size_t numWorkersForUnary, size_t numWorkersForStreaming, int port, const std::string& address) :
    numWorkersForUnary(numWorkersForUnary),
    numWorkersForStreaming(numWorkersForStreaming),
    pool(std::make_unique<mediapipe::ThreadPool>("DrogonThreadPool", numWorkersForStreaming)),
    port(port),
    address(address) {
    SPDLOG_DEBUG("Starting http thread pool for streaming ({} threads)", numWorkersForStreaming);
    pool->StartWorkers();  // this tp is for streaming workload which cannot use drogon's internal listener threads
    SPDLOG_DEBUG("Thread pool started");
    trantor::Logger::setLogLevel(trantor::Logger::kInfo);
}

Status DrogonHttpServer::startAcceptingRequests() {
    SPDLOG_DEBUG("DrogonHttpServer::startAcceptingRequests()");

    // OVMS has its own sigterm handling
    drogon::app().disableSigtermHandling();

    drogon::app().setDefaultHandler([this](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        // No separate pool for unary requests, they are handled by drogon's listener threads
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

    // Should never happen
    if (drogon::app().isRunning()) {
        SPDLOG_ERROR("Drogon is already running");
        throw 42;
    }

    pool->Schedule(
        [this] {
            static int numberOfLaunchesInApplication = 0;
            numberOfLaunchesInApplication++;
            if (numberOfLaunchesInApplication > 1) {
                SPDLOG_ERROR("Drogon was already started, cannot start it again");
                return;
            }
            SPDLOG_DEBUG("Starting to listen on port {}", this->port);
            SPDLOG_DEBUG("Thread pool size for unary ({} drogon threads)", this->numWorkersForUnary);
            try {
                drogon::app()
                    .setThreadNum(this->numWorkersForUnary)  // threads for unary processing, streaming is done in separate pool
                    .setIdleConnectionTimeout(0)
                    .setClientMaxBodySize(1024 * 1024 * 1024)  // 1GB
                    .setClientMaxMemoryBodySize(std::numeric_limits<size_t>::max())
                    // .setMaxConnectionNum(100000)  // default is 100000
                    // .setMaxConnectionNumPerIP(0)  // default is 0=unlimited
                    // .setServerHeaderField("OpenVINO Model Server")
                    .enableServerHeader(false)
                    .enableDateHeader(false)
                    .addListener(this->address, this->port)
                    .run();
            } catch (...) {
                SPDLOG_ERROR("Exception occurred during drogon::run()");
            }
            SPDLOG_DEBUG("drogon::run() exits normally");
        });

    // wait until drogon becomes ready
    size_t runningCheckIntervalMillisec = 50;
    size_t maxTotalRunningCheckTimeMillisec = 5000;
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
    SPDLOG_INFO("REST server listening on port {} with {} unary threads and {} streaming threads",
        port,
        numWorkersForUnary,
        numWorkersForStreaming);
    return StatusCode::OK;
}

void DrogonHttpServer::terminate() {
    SPDLOG_DEBUG("DrogonHttpServer::terminate() begin");

    // // Should never happen
    // //if (!drogon::app().isRunning()) {
    // if (!drogon::app().isFullyRunning()) {
    //     SPDLOG_DEBUG("Drogon is not fully running");
    //     throw 42;
    // }

    // wait until drogon becomes fully ready
    size_t runningCheckIntervalMillisec = 50;
    size_t maxTotalRunningCheckTimeMillisec = 5000;
    size_t maxChecks = maxTotalRunningCheckTimeMillisec / runningCheckIntervalMillisec;
    while (!drogon::app().isFullyRunning()) {
        SPDLOG_DEBUG("Waiting for drogon to become fully before termination...", port);
        if (maxChecks == 0) {
            SPDLOG_DEBUG("Waiting for drogon readiness timed out");
            throw 42;
        }
        maxChecks--;
        std::this_thread::sleep_for(std::chrono::milliseconds(runningCheckIntervalMillisec));
    }

    SPDLOG_DEBUG("DrogonHttpServer::terminate() before quit");
    drogon::app().quit();
    SPDLOG_DEBUG("DrogonHttpServer::terminate() after quit before pool reset");
    pool.reset();  // waits for all worker threads to finish
    SPDLOG_DEBUG("DrogonHttpServer::terminate() end after pool reset");
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
