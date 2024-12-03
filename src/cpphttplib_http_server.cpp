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
#include "cpphttplib_http_server.hpp"

#include <mutex>
#include <condition_variable>
#include <chrono>

#include "httplib.h"

#include "logging.hpp"
#include "mediapipe/framework/port/threadpool.h"

namespace ovms {

CppHttpLibHttpServer::CppHttpLibHttpServer(size_t num_workers, int port, const std::string& address) :
    pool_(std::make_unique<mediapipe::ThreadPool>("CppHttpLibThreadPool", 3/*only for listener*//*num_workers*/)),
    port_(port),
    address_(address),
    //server_(std::make_unique<httplib::Server>()) {
    server_(std::make_unique<httplib::Server>(num_workers)) {
    SPDLOG_DEBUG("Starting ThreadPool with {} workers", 3);
    pool_->StartWorkers();  // this is for listener and also for streaming outputs
    SPDLOG_DEBUG("ThreadPool started");
    SPDLOG_DEBUG("CPPHTTPLIB Number of Workers: {}", num_workers);
}

void CppHttpLibHttpServer::startAcceptingRequests() {
    SPDLOG_DEBUG("///////////////////////////// CppHttpLibHttpServer::startAcceptingRequests()");

    // Any Get or Post
    server_->Get(R"(/.*)", [this](const httplib::Request& req, httplib::Response& res) {
        // this->pool_->Schedule([this, req, res]() mutable {
        //     // this->dispatcher_(req, std::move(callback));
        //     SPDLOG_DEBUG("Request: {}; Body: {};", req.path, req.body);
        //     res.set_content("Hello, World!", "text/plain");
        // });

        // measure time
        auto start = std::chrono::high_resolution_clock::now();

        this->dispatcher_(req, res);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        SPDLOG_DEBUG("Request took {} milliseconds", duration.count() / 1000.f);
    });

    server_->Post(R"(/.*)", [this](const httplib::Request& req, httplib::Response& res) {
        // this->pool_->Schedule([this, req, res]() mutable {
        //     // this->dispatcher_(req, std::move(callback));
        //     SPDLOG_DEBUG("Request: {}; Body: {};", req.path, req.body);
        //     res.set_content("Hello, World!", "text/plain");
        // });

        // measure time
        auto start = std::chrono::high_resolution_clock::now();

        this->dispatcher_(req, res);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        SPDLOG_DEBUG("Request took {} milliseconds", duration.count() / 1000.f);
    });

    pool_->Schedule(
        [this] {
            SPDLOG_DEBUG("Running httplib server {} {}", address_, port_);
            server_->listen(address_, port_);
            SPDLOG_DEBUG("Stopped httplib", address_, port_);
        });

    server_->wait_until_ready();
}

void CppHttpLibHttpServer::terminate() {
    SPDLOG_DEBUG("///////////////////////////// CppHttpLibHttpServer::terminate()");
    server_->stop();
}

void CppHttpLibHttpServer::registerRequestDispatcher(
    std::function<void(
        const httplib::Request& req, httplib::Response& res)>
        dispatcher) {
    SPDLOG_DEBUG("///////////////////////////// CppHttpLibHttpServer::registerRequestDispatcher()");
    dispatcher_ = std::move(dispatcher);
}

mediapipe::ThreadPool& CppHttpLibHttpServer::getPool() {
    return *pool_;
}

}  // namespace ovms
