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

#include <memory>
#include <utility>

#include "../client_connection.hpp"

#include "../http_async_writer_interface.hpp"

namespace ovms {

class HttpClientConnection : public ClientConnection {
    std::shared_ptr<HttpAsyncWriter> serverReaderWriter;

public:
    HttpClientConnection(std::shared_ptr<HttpAsyncWriter> serverReaderWriter) :
        serverReaderWriter(serverReaderWriter) {}

    bool isDisconnected() const override {
        return this->serverReaderWriter->IsDisconnected();
    }

    void registerDisconnectionCallback(std::function<void()> fn) override {
        serverReaderWriter->RegisterDisconnectionCallback(std::move(fn));
    }
};

}  // namespace ovms
