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
#include "drogon_http_async_writer_impl.hpp"

#include <functional>
#include <string>
#include <utility>

#include "logging.hpp"

namespace ovms {

// Used by V3 handler
void DrogonHttpAsyncWriterImpl::OverwriteResponseHeader(const std::string& key, const std::string& value) {
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::OverwriteResponseHeader {} {}", key, value);
    this->additionalHeaders[key] = value;
}
void DrogonHttpAsyncWriterImpl::PartialReplyWithStatus(std::string message, HTTPStatusCode status) {
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyWithStatus {} {}", message, int(status));
    if (this->isDisconnected) {
        return;
    }
    if (!this->stream->send(message))
        this->isDisconnected = true;
}
void DrogonHttpAsyncWriterImpl::PartialReplyBegin(std::function<void()> cb) {
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyBegin");
    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, cb = std::move(cb)](drogon::ResponseStreamPtr stream) {
            this->stream = std::move(stream);
            this->pool.Schedule([cb = std::move(cb)] {
                SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyBegin::Schedule begin");
                try {
                    cb();  // run actual workload (mediapipe executor inferStream) which uses PartialReply
                } catch (...) {
                    SPDLOG_ERROR("Exception caught in REST request streaming handler");
                }
                SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyBegin::Schedule end");
            });
        });

    // Convert headers to drogon format
    for (const auto& [key, value] : this->additionalHeaders) {
        if (key == "Content-Type") {
            resp->setContentTypeString(value);
            continue;
        }
        resp->addHeader(key, value);
    }
    this->callback(resp);
}
void DrogonHttpAsyncWriterImpl::PartialReplyEnd() {
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyEnd");
    this->stream->close();
}
// Used by graph executor impl
void DrogonHttpAsyncWriterImpl::PartialReply(std::string message) {
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReply {}", message);
    if (this->isDisconnected) {
        return;
    }
    if (!this->stream->send(message))
        this->isDisconnected = true;
}
// Used by calculator via HttpClientConnection
bool DrogonHttpAsyncWriterImpl::IsDisconnected() const {
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::IsDisconnected");
    return this->isDisconnected;
}

void DrogonHttpAsyncWriterImpl::RegisterDisconnectionCallback(std::function<void()> callback) {
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::RegisterDisconnectionCallback");
    // TODO: Implement once https://github.com/drogonframework/drogon/pull/2204 is merged
}

}  // namespace ovms
