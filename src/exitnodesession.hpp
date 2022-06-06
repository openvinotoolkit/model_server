//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <openvino/openvino.hpp>

#include "gatherexitnodeinputhandler.hpp"
#include "nodeinputhandler.hpp"
#include "nodesession.hpp"
#include "nodesessionmetadata.hpp"
#include "status.hpp"
#include "tensormap.hpp"

namespace ovms {

class Node;
class TensorInfo;

template <typename ResponseType>
class ExitNodeSession : public NodeSession {
public:
    ExitNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ResponseType* response) :
        NodeSession(metadata, nodeName, inputsCount, collapsingDetails) {
        if (collapsingDetails.collapsedSessionNames.size() != 0) {
            this->inputHandler = std::make_unique<GatherExitNodeInputHandler<ResponseType>>(inputsCount, collapsingDetails, response);
        }
    }
    virtual ~ExitNodeSession() = default;

    const TensorMap& getInputTensors() const { return this->inputHandler->getInputs(); }
    void release() override {}
};
}  // namespace ovms
