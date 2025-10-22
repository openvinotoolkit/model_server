//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <string>
#include "config.hpp"
#include "drogon_http_server.hpp"

namespace ovms {
class Server;

std::unique_ptr<DrogonHttpServer> createAndStartDrogonHttpServer(const std::string& address, int port, int num_threads, ovms::Server& ovmsServer, const ovms::Config& config, int timeout_in_ms = -1);

}  // namespace ovms
