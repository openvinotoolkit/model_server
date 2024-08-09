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

#include <utility>

#include "../client_connection.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#pragma GCC diagnostic pop

namespace ovms {

class HttpClientConnection : public ClientConnection {
    tensorflow::serving::net_http::ServerRequestInterface* serverReaderWriter;

public:
    HttpClientConnection(tensorflow::serving::net_http::ServerRequestInterface* serverReaderWriter) :
        serverReaderWriter(serverReaderWriter) {}

    bool isDisconnected() const override {
        return this->serverReaderWriter->IsDisconnected();
    }

    void registerDisconnectionCallback(std::function<void()> fn) override {
        serverReaderWriter->RegisterDisconnectionCallback(std::move(fn));
    }
};

}  // namespace ovms
