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

#include "nodesession.hpp"
#include "pipelineeventqueue.hpp"
#include "tensormap.hpp"

class CustomNodeTensor;
class CustomNodeParam;

namespace ovms {

class Node;
class NodeLibrary;
class Status;

class CustomNodeSession : public NodeSession {
    TensorMap resultTensors;

public:
    CustomNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails);
    CustomNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails);
    virtual ~CustomNodeSession();

    Status execute(
        PipelineEventQueue& notifyEndQueue,
        Node& node,
        const NodeLibrary& library,
        std::unique_ptr<struct CustomNodeParam[]>& parameters,
        int parametersCount,
        void* customNodeLibraryInternalManager);

    Status fetchResult(const std::string& name, ov::Tensor& resultTensor);

    void clearInputs();
    void release() override;

private:
    static void releaseTensorResources(const struct CustomNodeTensor* tensor, const NodeLibrary& library, void* customNodeLibraryInternalManager);
    Status createTensor(const struct CustomNodeTensor* tensor, ov::Tensor& resultTensor, const NodeLibrary& library, void* customNodeLibraryInternalManager);
};
}  // namespace ovms
