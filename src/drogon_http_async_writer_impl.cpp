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
    this->additionalHeaders[key] = value;
}
void DrogonHttpAsyncWriterImpl::PartialReplyWithStatus(std::string message, HTTPStatusCode status) {
    if (!began) {
        //std::string d = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\ncache-control: no-cache\r\nconnection: keep-alive\r\n\r\n";
        //this->stream->send(d);

        began = true;
    }
    if (this->isDisconnected) {
        return;
    }
    if (!this->stream->send(message))
        this->isDisconnected = true;
}
void DrogonHttpAsyncWriterImpl::PartialReplyBegin(std::function<void()> actualWorkloadCallback) {
    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, actualWorkloadCallback = std::move(actualWorkloadCallback)](drogon::ResponseStreamPtr stream) {
            SPDLOG_INFO("asyncStreamCallback begin...");
            this->stream = std::move(stream);
            this->pool.Schedule([actualWorkloadCallback = std::move(actualWorkloadCallback)] {
                SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyBegin::Schedule begin");
                try {
                    SPDLOG_INFO("actualWorkloadCallback begin...");
                    actualWorkloadCallback();  // run actual workload (mediapipe executor inferStream) which uses PartialReply
                    SPDLOG_INFO("actualWorkloadCallback end...");
                } catch (...) {
                    SPDLOG_ERROR("Exception caught in REST request streaming handler");
                }
                SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyBegin::Schedule end");
            });
            SPDLOG_INFO("asyncStreamCallback end...");
        });

    // Convert headers to drogon format
    for (const auto& [key, value] : this->additionalHeaders) {
        if (key == "Content-Type") {
            resp->setContentTypeString(value);
            continue;
        }
        resp->addHeader(key, value);
    }
    this->drogonResponseInitializeCallback(resp);
}
void DrogonHttpAsyncWriterImpl::PartialReplyEnd() {
    this->stream->close();
}
// Used by graph executor impl
void DrogonHttpAsyncWriterImpl::PartialReply(std::string message) {
    return PartialReplyWithStatus(std::move(message), HTTPStatusCode::OK);
    // if (this->IsDisconnected()) {
    //     return;
    // }
    // if (!this->stream->send(message))
    //     this->isDisconnected = true;
}
// Used by calculator via HttpClientConnection
bool DrogonHttpAsyncWriterImpl::IsDisconnected() const {
    return this->isDisconnected || !requestPtr->connected();
}

void DrogonHttpAsyncWriterImpl::RegisterDisconnectionCallback(std::function<void()> onDisconnectedCallback) {
    const auto& weakConnPtr = requestPtr->getConnectionPtr();
    if (auto connPtr = weakConnPtr.lock()) {
        connPtr->setCloseCallback([onDisconnectedCallback = std::move(onDisconnectedCallback)](const trantor::TcpConnectionPtr& conn) {
            onDisconnectedCallback();
        });
    }
}

}  // namespace ovms
