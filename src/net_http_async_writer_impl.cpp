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
#include "net_http_async_writer_impl.hpp"

#include <functional>
#include <string>
#include <utility>

#include "http_status_code.hpp"

namespace ovms {

// Used by V3 handler
void NetHttpAsyncWriterImpl::OverwriteResponseHeader(const std::string& key, const std::string& value) {
    this->req->OverwriteResponseHeader(key, value);
}
void NetHttpAsyncWriterImpl::PartialReplyWithStatus(std::string message, HTTPStatusCode status) {
    this->req->PartialReplyWithStatus(message, tensorflow::serving::net_http::HTTPStatusCode(int(status)));
}
void NetHttpAsyncWriterImpl::PartialReplyBegin(std::function<void()> cb) {
    cb();  // net_http can simply run the callback sequentially
}
void NetHttpAsyncWriterImpl::PartialReplyEnd() {
    this->req->PartialReplyEnd();
}
// Used by graph executor impl
void NetHttpAsyncWriterImpl::PartialReply(std::string message) {
    this->req->PartialReply(std::move(message));
}
// Used by calculator via HttpClientConnection
bool NetHttpAsyncWriterImpl::IsDisconnected() const {
    return this->req->IsDisconnected();
}

void NetHttpAsyncWriterImpl::RegisterDisconnectionCallback(std::function<void()> callback) {
    this->req->RegisterDisconnectionCallback(std::move(callback));
}

}  // namespace ovms
