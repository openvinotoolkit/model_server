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

#include "drogon_http_server.hpp"
#include "cpphttplib_http_server.hpp"
#include "http_server.hpp"
#include "module.hpp"

namespace ovms {
class Config;
class Server;
class HTTPServerModule : public Module {
    //std::unique_ptr<ovms::http_server> server;
    std::unique_ptr<CppHttpLibHttpServer> cppHttpLibServer;
    //std::unique_ptr<DrogonHttpServer> drogonServer;
    Server& ovmsServer;

public:
    HTTPServerModule(Server& ovmsServer);
    ~HTTPServerModule();
    Status start(const ovms::Config& config) override;

    void shutdown() override;
};
}  // namespace ovms
