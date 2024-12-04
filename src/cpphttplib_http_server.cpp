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

#include <chrono>

#include "httplib.h"

#include "logging.hpp"
#include "mediapipe/framework/port/threadpool.h"

namespace ovms {

class CustomHttpPool : public httplib::TaskQueue {
    mediapipe::ThreadPool& pool;
public:
  CustomHttpPool(mediapipe::ThreadPool& pool) : pool(pool) {}

  virtual bool enqueue(std::function<void()> fn) override {
    pool.Schedule(fn);
    return true;
  }

  virtual void shutdown() override {}
};

CppHttpLibHttpServer::CppHttpLibHttpServer(size_t num_workers, int port, const std::string& address) :
    num_workers(num_workers),
    pool_(std::make_unique<mediapipe::ThreadPool>("CppHttpLibThreadPool", num_workers)),
    port_(port),
    address_(address),
    server_(std::make_unique<httplib::Server>()) {
    SPDLOG_DEBUG("Starting thread pool ({} threads)", num_workers);
    pool_->StartWorkers();  // this is for listener and also for streaming outputs
    server_->new_task_queue = [this] {
        return new CustomHttpPool(*this->pool_);
    };
    SPDLOG_DEBUG("Thread pool started");
}

bool CppHttpLibHttpServer::startAcceptingRequests() {
    SPDLOG_DEBUG("CppHttpLibHttpServer::startAcceptingRequests()");

    // Any Get or Post
    server_->Get(R"(/.*)", [this](const httplib::Request& req, httplib::Response& res) {
        // measure time
        auto start = std::chrono::high_resolution_clock::now();

        this->dispatcher_(req, res);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        SPDLOG_DEBUG("CppHttpLibHttpServer request handling took {} milliseconds", duration.count() / 1000.f);
    });

    server_->Post(R"(/.*)", [this](const httplib::Request& req, httplib::Response& res) {
        // measure time
        auto start = std::chrono::high_resolution_clock::now();

        this->dispatcher_(req, res);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        SPDLOG_DEBUG("CppHttpLibHttpServer request handling took {} milliseconds", duration.count() / 1000.f);
    });

    pool_->Schedule(
        [this] {
            SPDLOG_DEBUG("Starting to listen on port {}", port_);
            server_->listen(address_, port_);
            SPDLOG_DEBUG("Stopped listening");
        });

    server_->wait_until_ready();
    if (!server_->is_running()) {
        SPDLOG_ERROR("Failed to start cpp-httplib server on port {}", port_);
        return false;
    }

    SPDLOG_DEBUG("Server launched on port {}", port_);
    SPDLOG_INFO("REST server listening on port {} with {} threads", port_, num_workers);
    return true;
}

void CppHttpLibHttpServer::terminate() {
    SPDLOG_DEBUG("CppHttpLibHttpServer::terminate()");
    server_->stop();
    pool_.reset();  // waits for all worker threads to finish
}

void CppHttpLibHttpServer::registerRequestDispatcher(
    std::function<void(
        const httplib::Request& req, httplib::Response& res)>
        dispatcher) {
    dispatcher_ = std::move(dispatcher);
}

mediapipe::ThreadPool& CppHttpLibHttpServer::getPool() {
    return *pool_;
}

}  // namespace ovms
