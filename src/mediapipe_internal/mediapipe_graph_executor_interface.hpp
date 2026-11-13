//*****************************************************************************
// Copyright 2026 Intel Corporation
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
//****************************************************************************
#pragma once

#include <memory>
#include <string>

#include "src/execution_context.hpp"
#include "src/status.hpp"
#include "live_graph_session.hpp"

namespace grpc {
template <typename W, typename R>
class ServerReaderWriterInterface;
}

namespace inference {
class ModelInferRequest;
class ModelInferResponse;
class ModelStreamInferResponse;
}  // namespace inference

namespace ovms {

class HttpAsyncWriter;
struct HttpPayload;

class MediapipeGraphExecutorInterface {
public:
    virtual ~MediapipeGraphExecutorInterface() = default;

    /// Reports whether a named live execution profile was prepared during graph loading.
    virtual bool supportsSession(const std::string& profile) const = 0;
    /// Starts a managed live execution using normal graph resources. The returned
    /// session owns its lifetime independently of this executor.
    virtual Status startSession(const std::string& profile, LiveGraphSession::Observer observer,
        std::unique_ptr<LiveGraphSession>& session) = 0;

    virtual Status infer(const inference::ModelInferRequest* request,
        inference::ModelInferResponse* response,
        const ExecutionContext& executionContext) = 0;
    virtual Status inferStream(const inference::ModelInferRequest& firstRequest,
        grpc::ServerReaderWriterInterface<inference::ModelStreamInferResponse, inference::ModelInferRequest>& serverReaderWriter,
        const ExecutionContext& executionContext) = 0;
    virtual Status infer(const HttpPayload* request,
        std::string* response,
        const ExecutionContext& executionContext) = 0;
    virtual Status inferStream(const HttpPayload& firstRequest,
        HttpAsyncWriter& serverReaderWriter,
        const ExecutionContext& executionContext) = 0;
};

}  // namespace ovms
