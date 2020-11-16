//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "http_server.hpp"

#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#include "tensorflow_serving/util/threadpool_executor.h"
#pragma GCC diagnostic pop

#include "http_rest_api_handler.hpp"
#include "status.hpp"

namespace ovms {

namespace net_http = tensorflow::serving::net_http;

class RequestExecutor final : public net_http::EventExecutor {
public:
    explicit RequestExecutor(int num_threads) :
        executor_(tensorflow::Env::Default(), "httprestserver", num_threads) {}

    void Schedule(std::function<void()> fn) override { executor_.Schedule(fn); }

private:
    tensorflow::serving::ThreadPoolExecutor executor_;
};

class RestApiRequestDispatcher {
public:
    RestApiRequestDispatcher(int timeout_in_ms) :
        regex_(HttpRestApiHandler::kPathRegexExp) {
        handler_ = std::make_unique<HttpRestApiHandler>(timeout_in_ms);
    }

    net_http::RequestHandler dispatch(net_http::ServerRequestInterface* req) {
        return [this](net_http::ServerRequestInterface* req) {
            this->processRequest(req);
        };
    }

private:
    void processRequest(net_http::ServerRequestInterface* req) {
        SPDLOG_DEBUG("REST request {}", req->uri_path());
        std::string body;
        int64_t num_bytes = 0;
        auto request_chunk = req->ReadRequestBytes(&num_bytes);
        while (request_chunk != nullptr) {
            body.append(std::string_view(request_chunk.get(), num_bytes));
            request_chunk = req->ReadRequestBytes(&num_bytes);
        }

        std::vector<std::pair<std::string, std::string>> headers;
        std::string output;
        SPDLOG_DEBUG("Processing HTTP request: {} {} body: {} bytes",
            req->http_method(),
            req->uri_path(),
            body.size());
        const auto status = handler_->processRequest(req->http_method(), req->uri_path(), body, &headers, &output);
        if (!status.ok() && output.empty()) {
            output.append("{\"error\": \"" + status.string() + "\"}");
        }
        const auto http_status = status.http();
        for (const auto& kv : headers) {
            req->OverwriteResponseHeader(kv.first, kv.second);
        }
        req->WriteResponseString(output);
        if (http_status != net_http::HTTPStatusCode::OK) {
            SPDLOG_DEBUG("Processing HTTP/REST request failed: {} {}. Reason: {}",
                req->http_method(),
                req->uri_path(),
                status.string());
        }
        req->ReplyWithStatus(http_status);
    }

    const std::regex regex_;
    std::unique_ptr<HttpRestApiHandler> handler_;
};

std::unique_ptr<http_server> createAndStartHttpServer(const std::string& address, int port, int num_threads, int timeout_in_ms) {
    auto options = std::make_unique<net_http::ServerOptions>();
    options->AddPort(static_cast<uint32_t>(port));
    options->SetAddress(address);
    options->SetExecutor(std::make_unique<RequestExecutor>(num_threads));

    auto server = net_http::CreateEvHTTPServer(std::move(options));
    if (server == nullptr) {
        SPDLOG_ERROR("Failed to create http server");
        return nullptr;
    }

    std::shared_ptr<RestApiRequestDispatcher> dispatcher =
        std::make_shared<RestApiRequestDispatcher>(timeout_in_ms);

    net_http::RequestHandlerOptions handler_options;
    server->RegisterRequestDispatcher(
        [dispatcher](net_http::ServerRequestInterface* req) {
            return dispatcher->dispatch(req);
        },
        handler_options);

    if (server->StartAcceptingRequests()) {
        SPDLOG_INFO("REST server listening on port {} with {} threads", port, num_threads);
        return server;
    }

    return nullptr;
}
}  // namespace ovms
