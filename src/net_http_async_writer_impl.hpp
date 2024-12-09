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
#pragma once

#include <functional>
#include <string>

#include "http_async_writer_interface.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow_serving/util/net_http/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#include "tensorflow_serving/util/threadpool_executor.h"
#pragma GCC diagnostic pop

#include "http_status_code.hpp"

namespace ovms {

class NetHttpAsyncWriterImpl : public HttpAsyncWriter {
    tensorflow::serving::net_http::ServerRequestInterface* req;

public:
    NetHttpAsyncWriterImpl(
        tensorflow::serving::net_http::ServerRequestInterface* req) :
        req(req) {}

    // Used by V3 handler
    void OverwriteResponseHeader(const std::string& key, const std::string& value) override;
    void PartialReplyWithStatus(std::string message, HTTPStatusCode status) override;
    void PartialReplyBegin(std::function<void()> callback) override;
    void PartialReplyEnd() override;

    // Used by graph executor impl
    void PartialReply(std::string message) override;

    // Used by calculator via HttpClientConnection
    bool IsDisconnected() const override;
    void RegisterDisconnectionCallback(std::function<void()> callback) override;
};

}  // namespace ovms
