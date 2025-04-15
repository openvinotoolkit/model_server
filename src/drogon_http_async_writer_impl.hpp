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
#include <unordered_map>
#include <utility>
#pragma warning(push)
#pragma warning(disable : 6326)
#include <drogon/drogon.h>
#pragma warning(pop)

#include "http_async_writer_interface.hpp"
#include "mediapipe/framework/port/threadpool.h"

namespace ovms {

class DrogonHttpAsyncWriterImpl : public HttpAsyncWriter {
    std::function<void(const drogon::HttpResponsePtr&)> drogonResponseInitializeCallback;
    mediapipe::ThreadPool& pool;
    drogon::ResponseStreamPtr stream;
    bool isDisconnected = false;
    std::unordered_map<std::string, std::string> additionalHeaders;
    const drogon::HttpRequestPtr requestPtr{nullptr};
    drogon::HttpResponsePtr responsePtr{nullptr};

public:
    DrogonHttpAsyncWriterImpl(
        std::function<void(const drogon::HttpResponsePtr&)> callback,
        mediapipe::ThreadPool& pool,
        const drogon::HttpRequestPtr& requestPtr) :
        drogonResponseInitializeCallback(std::move(callback)),
        pool(pool),
        requestPtr(requestPtr) {}

    // Used by V3 handler
    void OverwriteResponseHeader(const std::string& key, const std::string& value) override;
    void PartialReplyWithStatus(std::string message, HTTPStatusCode status) override;
    void PartialReplyBegin(std::function<void()> actualWorkloadCallback) override;
    void PartialReplyEnd() override;

    // Used by graph executor impl
    void PartialReply(std::string message) override;

    // Used by calculator via HttpClientConnection
    bool IsDisconnected() const override;
    void RegisterDisconnectionCallback(std::function<void()> onDisconnectedCallback) override;

private:
    void sendHeaderIfFirstResponse(HTTPStatusCode status);
    bool firstResponse{true};
};

}  // namespace ovms
