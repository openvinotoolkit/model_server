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

#if (USE_DROGON == 0)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow_serving/util/net_http/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#include "tensorflow_serving/util/threadpool_executor.h"
#pragma GCC diagnostic pop
#else
#include "drogon_http_server.hpp"
#endif

namespace ovms {
class Server;

#if (USE_DROGON == 0)
std::unique_ptr<tensorflow::serving::net_http::HTTPServerInterface> createAndStartNetHttpServer(const std::string& address, int port, int num_threads, ovms::Server& ovmsServer, int timeout_in_ms = -1);
#else
std::unique_ptr<DrogonHttpServer> createAndStartDrogonHttpServer(const std::string& address, int port, int num_threads, ovms::Server& ovmsServer, int timeout_in_ms = -1);
#endif

}  // namespace ovms
