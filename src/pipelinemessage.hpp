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
#include <utility>

#include "executinstreamidguard.hpp"
#include "modelinstanceunloadguard.hpp"
#include "status.hpp"

namespace ovms {
class PipelineMessage {
public:
    PipelineMessage(Node& node,
        ovms::StatusCode statusCode,
        std::unique_ptr<ovms::ExecutingStreamIdGuard> executinStreamIdGuard,
        std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard) :
        node(node),
        statusCode(statusCode),
        executingStreamIdGuard(std::move(executingStreamIdGuard)),
        modelInstanceUnloadGuard(std::move(modelInstanceUnloadGuard)) {}
    PipelineMessage(PipelineMessage&) = delete;
    PipelineMessage(PipelineMessage&& rhs) :
        node(rhs.node),
        statusCode(std::move(rhs.statusCode)),
        executingStreamIdGuard(std::move(rhs.executingStreamIdGuard)),
        modelInstanceUnloadGuard(std::move(rhs.modelInstanceUnloadGuard)) {}
    PipelineMessage operator=(PipelineMessage&) = delete;

private:
    Node& node;
    ovms::StatusCode statusCode;
    std::unique_ptr<ovms::ExecutingStreamIdGuard> executingStreamIdGuard;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
};
}  //  namespace ovms
