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
    // Use weak_ptr to break circular reference: 
    // Connection -> responsePtr -> lambda -> this -> requestPtr -> Connection
    auto sharedThis = this->shared_from_this();
    auto weakThis = std::weak_ptr<DrogonHttpAsyncWriterImpl>(sharedThis);
    
    this->responsePtr = drogon::HttpResponse::newAsyncStreamResponse(
        [weakThis, actualWorkloadCallback = std::move(actualWorkloadCallback), &pool = this->pool](drogon::ResponseStreamPtr stream) {
            auto strongThis = weakThis.lock();
            if (!strongThis) {
                SPDLOG_ERROR("Writer destroyed before stream callback");
                return;
            }
            strongThis->stream = std::move(stream);
            pool.Schedule([actualWorkloadCallback = std::move(actualWorkloadCallback)] {
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
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::PartialReplyEnd() called");
    this->stream->close();
    
    // Clear the close callback to allow connection cleanup
    // This is critical: even with weak_ptr in ResponseStream, the closeCallback_ stored
    // in TcpConnection prevents it from being released for connection reuse
    const auto& weakConnPtr = requestPtr->getConnectionPtr();
    if (auto connPtr = weakConnPtr.lock()) {
        SPDLOG_DEBUG("Clearing close callback on connection");
        connPtr->setCloseCallback([](const trantor::TcpConnectionPtr&) {});
    }
    
    // Stop the polling thread and clean up
    if (this->pollingThread && this->pollingThread->joinable()) {
        SPDLOG_DEBUG("Stopping disconnection polling thread");
        this->stopPolling.store(true);
        this->pollingThread->join();
        this->pollingThread.reset();
        this->disconnectionCallback = nullptr;
    } else {
        SPDLOG_DEBUG("No polling thread to stop");
    }
    
    // CRITICAL: Release request and response to break circular references
    // Writer may still be held by mediapipe/calculators, but this releases the connection
    SPDLOG_DEBUG("Releasing requestPtr and responsePtr to free connection");
    this->requestPtr.reset();
    this->responsePtr.reset();
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
    SPDLOG_DEBUG("DrogonHttpAsyncWriterImpl::RegisterDisconnectionCallback() called");
    
    const auto& weakConnPtr = requestPtr->getConnectionPtr();
    if (auto connPtr = weakConnPtr.lock()) {
        // Get the original callback set by TcpServer (does connSet_.erase())
        auto originalCallback = connPtr->getCloseCallback();
        
        // Chain callbacks: call original first (for cleanup), then ours (for early stop optimization)
        connPtr->setCloseCallback([originalCallback, onDisconnectedCallback = std::move(onDisconnectedCallback)](const trantor::TcpConnectionPtr& conn) {
            // Call original callback to maintain Drogon's connection management
            if (originalCallback) {
                originalCallback(conn);
            }
            // Then call our callback for early generation stop
            if (onDisconnectedCallback) {
                onDisconnectedCallback();
            }
        });
    }
}

}  // namespace ovms
