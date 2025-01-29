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
#include <memory>
#include <string>
#pragma warning(push)
#pragma warning(disable : 6326)
#include <drogon/drogon.h>
#pragma warning(pop)

#include "http_async_writer_interface.hpp"
#include "mediapipe/framework/port/threadpool.h"
#include "status.hpp"

namespace ovms {

class DrogonHttpServer {
    size_t numWorkersForUnary;
    size_t numWorkersForStreaming;
    std::unique_ptr<mediapipe::ThreadPool> pool;
    int port;
    std::string address;
    std::function<void(
        const drogon::HttpRequestPtr&,
        std::function<void(const drogon::HttpResponsePtr&)>&&)>
        dispatcher;

public:
    DrogonHttpServer(
        size_t numWorkersForUnary,
        size_t numWorkersForStreaming,
        int port,
        const std::string& address);

    Status startAcceptingRequests();
    void terminate();

    mediapipe::ThreadPool& getPool();

    void registerRequestDispatcher(
        std::function<void(
            const drogon::HttpRequestPtr&,
            std::function<void(const drogon::HttpResponsePtr&)>&&)>
            dispatcher);
};

}  // namespace ovms
