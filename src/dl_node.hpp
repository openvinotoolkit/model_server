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
#include <string>
#include <unordered_map>

#include "executingstreamidguard.hpp"
#include "model_version_policy.hpp"  // for model_version_t typename
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "node.hpp"
#include "nodestreamidguard.hpp"

namespace ovms {

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
        std::unordered_map<std::string, std::string> nodeOutputNameAlias = {}) :
        Node(nodeName),
        modelName(modelName),
        modelVersion(modelVersion),
        modelManager(modelManager),
        nodeOutputNameAlias(nodeOutputNameAlias) {
    }

    Status execute(session_key_t sessionKey, PipelineEventQueue& notifyEndQueue) override;

    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) override;

private:
    Status fetchResults(BlobMap& outputs, InferenceEngine::InferRequest& inferRequest, ModelInstance& model, session_key_t sessionKey);

public:
    Status validate(const InferenceEngine::Blob::Ptr& blob, const TensorInfo& info);

    void release(session_key_t sessionId) override;

private:
    Status getRealInputName(ModelInstance& model, const std::string& alias, std::string* result) const {
        if (model.getInputsInfo().count(alias) == 0) {
            return StatusCode::INVALID_MISSING_INPUT;
        }
        *result = model.getInputsInfo().at(alias)->getName();
        return StatusCode::OK;
    }

    Status getRealOutputName(ModelInstance& model, const std::string& alias, std::string* result) const {
        const auto& modelOutputName = nodeOutputNameAlias.count(alias) == 1 ? nodeOutputNameAlias.at(alias) : alias;
        if (model.getOutputsInfo().count(modelOutputName) == 0) {
            return StatusCode::INVALID_MISSING_OUTPUT;
        }
        *result = model.getOutputsInfo().at(modelOutputName)->getName();
        return StatusCode::OK;
    }

    Status executeInference(PipelineEventQueue& notifyEndQueue, InferenceEngine::InferRequest& infer_request);
    bool tryDisarm(const session_key_t& sessionKey, const uint microseconds = 1) override;

protected:
    std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, session_id_t shardsCount) override;
};

}  // namespace ovms
