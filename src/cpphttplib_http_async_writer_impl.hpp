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

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "http_async_writer_interface.hpp"
#include "httplib.h"  // NOLINT
#include "mediapipe/framework/port/threadpool.h"

namespace ovms {

class CppHttpLibHttpAsyncWriterImpl : public HttpAsyncWriter {
    httplib::Response& resp;
    mediapipe::ThreadPool& pool;

    httplib::DataSink* sink{nullptr};

    std::condition_variable cv;
    std::mutex mtx;

public:
    CppHttpLibHttpAsyncWriterImpl(
        httplib::Response& resp, mediapipe::ThreadPool& pool) :
        resp(resp),
        pool(pool) {}

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
