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

#include <atomic>
#include <chrono>
#include <limits>
#include <thread>
#include <utility>
#pragma warning(push)
#pragma warning(disable : 6326)
#include <drogon/drogon.h>
#pragma warning(pop)

#include "config.hpp"
#include "logging.hpp"
#include "mediapipe/framework/port/threadpool.h"
#include "timer.hpp"
#include "stringutils.hpp"

namespace ovms {

// Process-lifetime pointer to the currently active DrogonHttpServer instance.
// Access is guarded by std::atomic to allow safe reads from Drogon's event threads
// while the OVMS server is being started or stopped.
static std::atomic<DrogonHttpServer*> g_currentServer{nullptr};

// True once drogon::app().run() has been scheduled for the first time.
// Drogon's global app() singleton cannot be restarted after quit(), so it is
// kept alive for the full duration of the process.
static std::atomic<bool> g_drogonLaunched{false};

// Owns the Drogon event loop thread so it can be joined at process exit.
// Allocated once with `new` to avoid std::thread's static-destructor terminate().
static std::thread* g_drogonThread = nullptr;

DrogonHttpServer::DrogonHttpServer(size_t numWorkersForUnary, size_t numWorkersForStreaming, int port, const std::string& address) :
    numWorkersForUnary(numWorkersForUnary),
    numWorkersForStreaming(numWorkersForStreaming),
    pool(std::make_unique<mediapipe::ThreadPool>("DrogonThreadPool", numWorkersForStreaming)),
    port(port),
    address(address) {
    SPDLOG_DEBUG("Starting http thread pool for streaming ({} threads)", numWorkersForStreaming);
    pool->StartWorkers();  // this tp is for streaming workload which cannot use drogon's internal listener threads
    SPDLOG_DEBUG("Thread pool started");

    const char* envVarValue = std::getenv("DROGON_LOG_LEVEL");
    if (envVarValue != nullptr) {
        auto logLevelOpt = stoi32(envVarValue);
        if (!logLevelOpt.has_value() || logLevelOpt.value() < 0 || logLevelOpt.value() >= trantor::Logger::kNumberOfLogLevels) {
            SPDLOG_WARN("Invalid DROGON_LOG_LEVEL value, using default log level INFO", envVarValue);
            trantor::Logger::setLogLevel(trantor::Logger::kInfo);
        } else {
            int logLevel = logLevelOpt.value();
            SPDLOG_DEBUG("Setting drogon log level to {}", logLevel);
            if (logLevel == trantor::Logger::kTrace) {
                SPDLOG_DEBUG("Note: Setting log level to trace, but trace logs are disabled at compile time anyway");
            }
            trantor::Logger::setLogLevel(static_cast<trantor::Logger::LogLevel>(logLevel));
        }
    } else {
        SPDLOG_DEBUG("DROGON_LOG_LEVEL env var not set, using default log level INFO");
        trantor::Logger::setLogLevel(trantor::Logger::kInfo);
    }
}

namespace {
enum : unsigned int {
    WAIT_RUN,
    TIMER_END
};
}  // namespace

void DrogonHttpServer::dispatch(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& drogonResponseInitializeCallback) {
    try {
        this->dispatcher(req, drogonResponseInitializeCallback);
    } catch (...) {
        SPDLOG_DEBUG("Exception caught in REST request handler");
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
        drogonResponseInitializeCallback(resp);
    }
}

Status DrogonHttpServer::startAcceptingRequests() {
    SPDLOG_DEBUG("DrogonHttpServer::startAcceptingRequests()");

    // Publish this instance as the active server so the static Drogon handler
    // can route requests to it.
    g_currentServer.store(this, std::memory_order_release);

    if (g_drogonLaunched.exchange(true)) {
        // Drogon was already started by a previous OVMS server instance in this
        // process.  Because drogon::app() is a process-lifetime singleton that
        // cannot be restarted after quit(), we keep it running and simply reuse
        // the existing listener.  The handler already reads from g_currentServer,
        // so no further Drogon reconfiguration is needed.
        if (drogon::app().isRunning()) {
            SPDLOG_INFO("Reusing running Drogon instance on port {}", port);
            SPDLOG_INFO("REST server reusing port {} with {} unary threads and {} streaming threads",
                port,
                numWorkersForUnary,
                numWorkersForStreaming);
            return StatusCode::OK;
        }
        // Drogon was previously stopped (quit was called externally) — cannot recover.
        SPDLOG_ERROR("Drogon was previously stopped and cannot be restarted in the same process");
        return StatusCode::INTERNAL_ERROR;
    }

    // First launch: configure the process-lifetime Drogon singleton.

    // OVMS has its own sigterm handling.
    drogon::app().disableSigtermHandling();

    // Register a handler that always routes through the atomic pointer so that
    // the handler remains valid across OVMS server restarts without requiring
    // a second call to setDefaultHandler (which is not safe while Drogon runs).
    drogon::app().setDefaultHandler([](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& drogonResponseInitializeCallback) {
        auto* srv = g_currentServer.load(std::memory_order_acquire);
        if (!srv) {
            // OVMS is between restarts or shutting down — return 503.
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::HttpStatusCode::k503ServiceUnavailable);
            drogonResponseInitializeCallback(resp);
            return;
        }
        bool isStreamingEndpoint = req->path().find("/completions") != std::string::npos ||
                                   req->path().find("/responses") != std::string::npos ||
                                   req->path().find("/audio/transcriptions") != std::string::npos;

        // Schedule streaming requests to the per-instance thread pool so that
        // Drogon's disconnection callback works correctly.
        if (isStreamingEndpoint) {
            srv->pool->Schedule([srv, req, drogonResponseInitializeCallback = std::move(drogonResponseInitializeCallback)]() mutable {
                SPDLOG_DEBUG("Request URI {} dispatched to streaming thread pool", req->path());
                srv->dispatch(req, std::move(drogonResponseInitializeCallback));
            });
        } else {
            // Unary requests are handled directly on Drogon's listener threads.
            SPDLOG_DEBUG("Request URI working in drogon thread pool", req->path());
            srv->dispatch(req, std::move(drogonResponseInitializeCallback));
        }
    });

    // Run the Drogon event loop in a dedicated thread so that it outlives any
    // individual DrogonHttpServer instance and does not block the streaming pool.
    // The thread is stored globally so the atexit handler can join it.
    auto numUnary = this->numWorkersForUnary;
    auto portCopy = this->port;
    auto addressCopy = this->address;

    g_drogonThread = new std::thread([numUnary, portCopy, addressCopy] {
        SPDLOG_DEBUG("Starting to listen on port {}", portCopy);
        SPDLOG_DEBUG("Thread pool size for unary ({} drogon threads)", numUnary);
        try {
            drogon::app()
                .setThreadNum(numUnary)  // threads for unary processing; streaming uses a separate pool
                .setIdleConnectionTimeout(0)
                .setClientMaxBodySize(1024 * 1024 * 1024)  // 1GB
                .setClientMaxMemoryBodySize(std::numeric_limits<size_t>::max())
                // .setMaxConnectionNum(100000)  // default is 100000
                // .setMaxConnectionNumPerIP(0)  // default is 0=unlimited
                // .setServerHeaderField("OpenVINO Model Server")
                .enableServerHeader(false)
                .enableDateHeader(false)
                .registerPreSendingAdvice([](const drogon::HttpRequestPtr& req, const drogon::HttpResponsePtr& resp) {
                    static const bool allowCredentials = ovms::Config::instance().allowCredentials();
                    if (allowCredentials) {
                        resp->addHeader("Access-Control-Allow-Credentials", "true");
                    }
                    const auto& allowedOrigins = ovms::Config::instance().allowedOrigins();
                    if (allowedOrigins.size()) {
                        resp->addHeader("Access-Control-Allow-Origin", allowedOrigins);
                    }
                    const auto& allowedMethods = ovms::Config::instance().allowedMethods();
                    if (allowedMethods.size()) {
                        resp->addHeader("Access-Control-Allow-Methods", allowedMethods);
                    }
                    const auto& allowedHeaders = ovms::Config::instance().allowedHeaders();
                    if (allowedHeaders.size()) {
                        resp->addHeader("Access-Control-Allow-Headers", allowedHeaders);
                    }
                });

            auto ips = ovms::tokenize(addressCopy, ',');
            for (const auto& ip : ips) {
                SPDLOG_INFO("Binding REST server to address: {}:{}", ip, portCopy);
                drogon::app().addListener(ip, portCopy);
            }
            drogon::app().run();
        } catch (...) {
            SPDLOG_ERROR("Exception occurred during drogon::run()");
        }
        SPDLOG_DEBUG("drogon::run() exits normally");
    });

    // Register a process-exit handler that calls app().quit() and joins the
    // Drogon thread.  This ensures Drogon frees its internal heap state before
    // ASAN's leak check runs, preventing false leak reports.
    std::atexit([] {
        if (drogon::app().isRunning()) {
            drogon::app().quit();
        }
        if (g_drogonThread && g_drogonThread->joinable()) {
            g_drogonThread->join();
        }
        delete g_drogonThread;
        g_drogonThread = nullptr;
    });

    // Wait until Drogon is accepting connections.
    size_t runningCheckIntervalMillisec = 50;
    size_t maxTotalRunningCheckTimeMillisec = 5000;
    size_t maxChecks = maxTotalRunningCheckTimeMillisec / runningCheckIntervalMillisec;
    Timer<TIMER_END> timer;
    timer.start(WAIT_RUN);
    while (!drogon::app().isRunning()) {
        SPDLOG_DEBUG("Waiting for drogon to become ready on port {}...", port);
        if (maxChecks == 0) {
            SPDLOG_DEBUG("Waiting for drogon server launch timed out");
            return StatusCode::INTERNAL_ERROR;
        }
        maxChecks--;
        std::this_thread::sleep_for(std::chrono::milliseconds(runningCheckIntervalMillisec));
    }
    timer.stop(WAIT_RUN);
    SPDLOG_DEBUG("Drogon run procedure took: {} ms", timer.elapsed<std::chrono::microseconds>(WAIT_RUN) / 1000);
    SPDLOG_INFO("REST server listening on port {} with {} unary threads and {} streaming threads",
        port,
        numWorkersForUnary,
        numWorkersForStreaming);
    return StatusCode::OK;
}

void DrogonHttpServer::terminate() {
    // Clear the active-server pointer so that requests arriving while OVMS is
    // shutting down receive a 503 response instead of being routed to a server
    // that is mid-teardown.
    g_currentServer.store(nullptr, std::memory_order_release);

    // Do NOT call drogon::app().quit() here.  drogon::app() is a process-lifetime
    // singleton: once quit() is called its event loop cannot be restarted, which
    // would break any subsequent OVMS server instance in the same process (e.g.
    // the C-API fuzzer restarts the server on every corpus entry).  The Drogon
    // event loop is intentionally kept alive until the process exits.

    pool.reset();  // waits for all in-flight streaming worker threads to finish
}

void DrogonHttpServer::registerRequestDispatcher(
    std::function<void(
        const drogon::HttpRequestPtr&,
        std::function<void(const drogon::HttpResponsePtr&)>)>
        dispatcher) {
    this->dispatcher = std::move(dispatcher);
}

mediapipe::ThreadPool& DrogonHttpServer::getPool() {
    return *pool;
}

}  // namespace ovms
