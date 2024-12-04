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

#include "http_async_writer_interface.hpp"
#include "mediapipe/framework/port/threadpool.h"

namespace httplib {
    class Server;
    class Request;
    class Response;
}

namespace ovms {

class CppHttpLibHttpServer {
    size_t num_workers{0};
    std::unique_ptr<mediapipe::ThreadPool> pool_;
    int port_;
    std::string address_;
    std::unique_ptr<httplib::Server> server_{nullptr};
    std::function<void(
        const httplib::Request& req, httplib::Response& res)>
        dispatcher_;

public:
    CppHttpLibHttpServer(size_t num_workers, int port, const std::string& address);

    bool startAcceptingRequests();
    void terminate();

    mediapipe::ThreadPool& getPool();

    void registerRequestDispatcher(
        std::function<void(
            const httplib::Request& req, httplib::Response& res)>
            dispatcher);
};

}  // namespace ovms
