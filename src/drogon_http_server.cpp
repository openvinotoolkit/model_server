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

// True after the one-time Drogon configuration (handler, options, listeners).
// Never reset — these settings persist in Drogon's internal state across restarts.
static std::atomic<bool> g_drogonConfigured{false};

// True while drogon::app().run() is active.  Reset by terminate() after joining
// the event-loop thread so that the next startAcceptingRequests() can restart.
static std::atomic<bool> g_drogonLaunched{false};

// Owns the Drogon event-loop thread between startAcceptingRequests() and terminate().
// Allocated with new to avoid std::thread's static-destructor terminate().
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

    const bool alreadyLaunched = g_drogonLaunched.exchange(true);
    if (alreadyLaunched) {
        // app().run() is still active from a previous start that was not yet terminated.
        if (drogon::app().isRunning()) {
            SPDLOG_INFO("Reusing running Drogon instance on port {}", port);
            SPDLOG_INFO("REST server reusing port {} with {} unary threads and {} streaming threads",
                port,
                numWorkersForUnary,
                numWorkersForStreaming);
            return StatusCode::OK;
        }
        SPDLOG_ERROR("Drogon launched flag set but app is not running — internal state inconsistency");
        return StatusCode::INTERNAL_ERROR;
    }

    // This is either the very first start or a restart after terminate().
    // Perform one-time Drogon configuration only on the very first start;
    // on restarts the configuration (handler, options, listener ports) is already
    // stored in Drogon's internal state and must not be duplicated.
    const bool needsConfig = !g_drogonConfigured.exchange(true);
    if (needsConfig) {
        // OVMS has its own sigterm handling.
        drogon::app().disableSigtermHandling();

        // Register a handler that routes through the atomic pointer so that
        // it remains valid across OVMS server restarts without requiring a
        // second call to setDefaultHandler (which is not safe while Drogon runs).
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

        // One-time server-wide options.
        auto numUnary = this->numWorkersForUnary;
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

        // Register listeners once; they persist in Drogon's internal state across
        // restarts so addListener() must not be called again on subsequent starts.
        auto ips = ovms::tokenize(this->address, ',');
        for (const auto& ip : ips) {
            SPDLOG_INFO("Binding REST server to address: {}:{}", ip, this->port);
            drogon::app().addListener(ip, this->port);
        }
    } else {
        SPDLOG_DEBUG("Restarting Drogon on port {} (reusing existing configuration)", port);
    }

    // Start (or restart) the Drogon event loop in a dedicated thread.
    // On restart, Drogon reuses the listener configuration set during the first
    // launch; calling addListener() again is not needed and would cause duplicates.
    g_drogonThread = new std::thread([portCopy = this->port] {
        SPDLOG_DEBUG("Starting to listen on port {}", portCopy);
        try {
            drogon::app().run();
        } catch (...) {
            SPDLOG_ERROR("Exception occurred during drogon::run()");
        }
        SPDLOG_DEBUG("drogon::run() exits normally");
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
    // Clear the active-server pointer so that requests arriving during OVMS
    // shutdown receive a 503 instead of being routed to a torn-down instance.
    g_currentServer.store(nullptr, std::memory_order_release);

    // Stop Drogon's event loop and free its internal heap state.  This prevents
    // LSAN from reporting Drogon's live allocations as leaks during the
    // inter-iteration leak check that libFuzzer performs between corpus entries.
    // Drogon CAN be restarted by calling drogon::app().run() again; the one-time
    // configuration (listeners, handler, options) is preserved in Drogon's
    // internal state and does not need to be re-applied on the next start.
    if (drogon::app().isRunning()) {
        drogon::app().quit();
    }
    if (g_drogonThread && g_drogonThread->joinable()) {
        g_drogonThread->join();
        delete g_drogonThread;
        g_drogonThread = nullptr;
    }
    // Allow startAcceptingRequests() to restart app().run() on the next OVMS
    // server instance in this process.
    g_drogonLaunched.store(false, std::memory_order_release);

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
