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

#include <string>

#include "execution_context.hpp"
#include "status.hpp"

namespace grpc_impl {
template <typename W, typename R>
class ServerReaderWriterInterface;
}

namespace inference {
class ModelInferRequest;
class ModelInferResponse;
class ModelStreamInferResponse;
}

namespace ovms {

class HttpAsyncWriter;
class HttpPayload;

class MediapipeGraphExecutorInterface {
public:
    virtual ~MediapipeGraphExecutorInterface() = default;

    virtual Status infer(const inference::ModelInferRequest* request,
        inference::ModelInferResponse* response,
        const ExecutionContext& executionContext) = 0;
    virtual Status inferStream(const inference::ModelInferRequest& firstRequest,
        grpc_impl::ServerReaderWriterInterface<inference::ModelStreamInferResponse, inference::ModelInferRequest>& serverReaderWriter,
        const ExecutionContext& executionContext) = 0;
    virtual Status infer(const HttpPayload* request,
        std::string* response,
        const ExecutionContext& executionContext) = 0;
    virtual Status inferStream(const HttpPayload& firstRequest,
        HttpAsyncWriter& serverReaderWriter,
        const ExecutionContext& executionContext) = 0;
};

}  // namespace ovms