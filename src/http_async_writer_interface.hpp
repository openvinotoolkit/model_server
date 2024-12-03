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
#include <string>

namespace ovms {

enum class HTTPStatus : int {
    OK = 200,
    INVALID = 403,
};

class DrogonHttpAsyncWriter {
public:
    // Used by V3 handler
    virtual void OverwriteResponseHeader(const std::string& key, const std::string& value) = 0;
    virtual void PartialReplyWithStatus(std::string message, HTTPStatus status) = 0;
    virtual void PartialReplyBegin(std::function<void()> callback) = 0;
    virtual void PartialReplyEnd() = 0;

    // Used by graph executor impl
    virtual void PartialReply(std::string message) = 0;

    // Used by calculator via HttpClientConnection
    virtual bool IsDisconnected() const = 0;
    virtual void RegisterDisconnectionCallback(std::function<void()> callback) = 0;
};

}  // namespace ovms
