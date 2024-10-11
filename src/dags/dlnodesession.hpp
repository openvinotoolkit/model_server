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
#include <utility>

#include <openvino/openvino.hpp>

#include "../modelversion.hpp"
#include "nodesession.hpp"
#include "pipelineeventqueue.hpp"

namespace ovms {

class ModelManager;
class ModelInstance;
class Node;
class NodeStreamIdGuard;
class ModelInstanceUnloadGuard;
class Status;
class TensorInfo;

class DLNodeSession : public NodeSession {
    std::shared_ptr<ModelInstance> model;
    std::unique_ptr<NodeStreamIdGuard> nodeStreamIdGuard;
    std::unique_ptr<ModelInstanceUnloadGuard> modelUnloadGuard;

    ModelManager& modelManager;
    const std::string& modelName;
    const model_version_t modelVersion;

public:
    DLNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ModelManager& manager, const std::string& modelName, model_version_t modelVersion);
    DLNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ModelManager& manager, const std::string& modelName, model_version_t modelVersion);
    virtual ~DLNodeSession();

    ov::InferRequest& getInferRequest(const uint32_t microseconds);
    ModelInstance& getModelInstance();

private:
    Status requestExecuteRequiredResources();

public:
    Status prepareInputsAndModelForInference();
    Status validate(const ov::Tensor& tensor, const TensorInfo& info);
    Status execute(PipelineEventQueue& notifyEndQueue, uint32_t waitForStreamIdTimeoutMicroseconds, Node& node);
    Status executeInference(PipelineEventQueue& notifyEndQueue, ov::InferRequest&, Node& node);
    Status setInputsForInference(ov::InferRequest& inferRequest);
    Status getRealInputName(const std::string& alias, std::string* result) const;
    void release() override;

    void clearInputs();

    const std::string& getModelName() { return modelName; }
    bool tryDisarm(uint32_t microseconds) override;
};
}  // namespace ovms
