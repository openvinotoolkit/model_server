//****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <memory>
#include <utility>

#include "http_server.hpp"

namespace ovms {
class Server;
class Config;
// TODO should replace all messages like
// start REST Server with start HTTP Server
// start Server with start gRPC server
// this should be synchronized with validation tests changes
class HTTPServerModule : public Module {
    std::unique_ptr<ovms::http_server> server;
    Server& ovmsServer;

public:
    HTTPServerModule(Server& ovmsServer);
    ~HTTPServerModule();
    int start(const Config& config) override;
    void shutdown() override;
};
}  //namespace ovms
