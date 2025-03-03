//****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <memory>
#include <utility>

#if (USE_DROGON == 0)
#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011 4457 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow_serving/util/net_http/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#include "tensorflow_serving/util/threadpool_executor.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#else
#include "drogon_http_server.hpp"
#endif
#include "http_server.hpp"
#include "module.hpp"

namespace ovms {
class Config;
class Server;
class HTTPServerModule : public Module {
#if (USE_DROGON == 0)
    std::unique_ptr<tensorflow::serving::net_http::HTTPServerInterface> netHttpServer;
#else
    std::unique_ptr<DrogonHttpServer> drogonServer;
#endif
    Server& ovmsServer;

public:
    HTTPServerModule(Server& ovmsServer);
    ~HTTPServerModule();
    Status start(const ovms::Config& config) override;

    void shutdown() override;
};
}  // namespace ovms
