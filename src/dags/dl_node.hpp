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
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "../executingstreamidguard.hpp"
#include "../model_version_policy.hpp"  // for model_version_t typename
#include "../modelversion.hpp"
#include "node.hpp"

namespace ovms {

class ModelInstance;
class ModelInstanceUnloadGuard;
class NodeStreamIdGuard;
class ModelManager;

class DLNode : public Node {
protected:
    std::string modelName;
    std::optional<model_version_t> modelVersion;
    ModelManager& modelManager;
    const std::unordered_map<std::string, std::string> nodeOutputNameAlias;

    std::shared_ptr<ModelInstance> model;
    std::unique_ptr<NodeStreamIdGuard> nodeStreamIdGuard;
    std::unique_ptr<ModelInstanceUnloadGuard> modelUnloadGuard;

public:
    DLNode(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion,
        ModelManager& modelManager,
        std::unordered_map<std::string, std::string> nodeOutputNameAlias = {},
        std::optional<int32_t> demultiplyCount = std::nullopt, std::set<std::string> gatherFromNode = {});

    Status execute(session_key_t sessionKey, PipelineEventQueue& notifyEndQueue) override;

    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) override;

private:
    Status fetchResults(TensorWithSourceMap& outputs, ov::InferRequest& inferRequest, ModelInstance& model, session_key_t sessionKey);

public:
    void release(session_key_t sessionId) override;

private:
    Status getRealOutputName(ModelInstance& model, const std::string& alias, std::string* result) const;

    Status executeInference(PipelineEventQueue& notifyEndQueue, ov::InferRequest& infer_request);
    bool tryDisarm(const session_key_t& sessionKey, const uint32_t microseconds = 1) override;

protected:
    std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) override;
};

}  // namespace ovms
