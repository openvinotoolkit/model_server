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

#include "executinstreamidguard.hpp"
#include "model_version_policy.hpp"  // for model_version_t typename
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "node.hpp"

namespace ovms {

class DLNode : public Node {
    std::string modelName;
    std::optional<model_version_t> modelVersion;
    ModelManager& modelManager;

    std::shared_ptr<ModelInstance> model;
    std::unique_ptr<ExecutingStreamIdGuard> streamIdGuard;
    std::unique_ptr<ModelInstanceUnloadGuard> modelUnloadGuard;

public:
    DLNode(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion, ModelManager& modelManager = ModelManager::getInstance()) :
        Node(nodeName),
        modelName(modelName),
        modelVersion(modelVersion),
        modelManager(modelManager) {
    }

    Status execute(ThreadSafeQueue<std::reference_wrapper<Node>>& notifyEndQueue) override;

    Status fetchResults(BlobMap& outputs) override;

    Status validate(const InferenceEngine::Blob::Ptr& blob, const TensorInfo& info);
};

}  // namespace ovms
