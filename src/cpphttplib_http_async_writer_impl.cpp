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
#include "cpphttplib_http_async_writer_impl.hpp"

#include <functional>
#include <string>

#include "logging.hpp"

namespace ovms {

// Used by V3 handler
void CppHttpLibHttpAsyncWriterImpl::OverwriteResponseHeader(const std::string& key, const std::string& value) {
    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::OverwriteResponseHeader {} {}", key, value);
    // TODO
}
void CppHttpLibHttpAsyncWriterImpl::PartialReplyWithStatus(std::string message, HTTPStatusCode status) {
    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReplyWithStatus {} {}", message, int(status));
    // TODO
}
void CppHttpLibHttpAsyncWriterImpl::PartialReplyBegin(std::function<void()> cb) {
    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReplyBegin start");

    auto chunked_content_provider = [this, cb = std::move(cb)] (size_t, httplib::DataSink & sink) {
        SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::chunked_content_provider");

        // Save the sink for later use by PartialReply
        this->sink = &sink;

        this->pool.Schedule([this, cb = std::move(cb)] {
            SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReplyBegin::Schedule before");

            cb();

            // Notify main thread to continue (disconnect)
            std::unique_lock<std::mutex> lock(this->mtx);
            this->cv.notify_all();

            SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReplyBegin::Schedule after");
        });

        // Block, wait for PartialReplyEnd
        std::unique_lock<std::mutex> lock(this->mtx);
        this->cv.wait(lock);

        return false;
    };

    auto on_complete = [this] (bool) {
        SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::on_complete");
    };

    this->resp.set_chunked_content_provider("text/event-stream", chunked_content_provider , on_complete);

    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReplyBegin end");
}
void CppHttpLibHttpAsyncWriterImpl::PartialReplyEnd() {
    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReplyEnd begin");

    this->sink->done();

    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReplyEnd end");
}
// Used by graph executor impl
void CppHttpLibHttpAsyncWriterImpl::PartialReply(std::string message) {
    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::PartialReply {}", message);
    
    this->sink->write(message.data(), message.size());
}
// Used by calculator via HttpClientConnection
bool CppHttpLibHttpAsyncWriterImpl::IsDisconnected() const {
    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::IsDisconnected");
    return false;
}

void CppHttpLibHttpAsyncWriterImpl::RegisterDisconnectionCallback(std::function<void()> callback) {
    SPDLOG_DEBUG("CppHttpLibHttpAsyncWriterImpl::RegisterDisconnectionCallback");
}

}  // namespace ovms
