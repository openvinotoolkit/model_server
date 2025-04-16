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
void DrogonHttpAsyncWriterImpl::sendHeaderIfFirstResponse(HTTPStatusCode status) {
    if (firstResponse) {
        firstResponse = false;
        this->responsePtr->setCustomStatusCode(int(status));
        this->stream->sendHeader(this->responsePtr->renderHeaderToString());
    }
}
void DrogonHttpAsyncWriterImpl::PartialReplyWithStatus(std::string message, HTTPStatusCode status) {
    if (this->isDisconnected) {
        return;
    }
    this->sendHeaderIfFirstResponse(status);
    if (!this->stream->send(message))
        this->isDisconnected = true;
}
void DrogonHttpAsyncWriterImpl::PartialReplyBegin(std::function<void()> actualWorkloadCallback) {
    this->responsePtr = drogon::HttpResponse::newAsyncStreamResponse(
        [this, actualWorkloadCallback = std::move(actualWorkloadCallback)](drogon::ResponseStreamPtr stream) {
            this->stream = std::move(stream);
            this->pool.Schedule([actualWorkloadCallback = std::move(actualWorkloadCallback)] {
                SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyBegin::Schedule begin");
                try {
                    actualWorkloadCallback();  // run actual workload (mediapipe executor inferStream) which uses PartialReply
                } catch (...) {
                    SPDLOG_ERROR("Exception caught in REST request streaming handler");
                }
                SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyBegin::Schedule end");
            });
        });

    // Convert headers to drogon format
    for (const auto& [key, value] : this->additionalHeaders) {
        if (key == "Content-Type") {
            this->responsePtr->setContentTypeString(value);
            continue;
        }
        this->responsePtr->addHeader(key, value);
    }

    // Originally this also sent http response header (with status code)
    // We have drogon patch that delays it till first streaming response
    this->drogonResponseInitializeCallback(this->responsePtr);
}
void DrogonHttpAsyncWriterImpl::PartialReplyEnd() {
    this->stream->close();
}
// Used by graph executor impl
void DrogonHttpAsyncWriterImpl::PartialReply(std::string message) {
    return PartialReplyWithStatus(std::move(message), HTTPStatusCode::OK);
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
