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

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "aliases.hpp"
#include "modelversion.hpp"
#include "nodeinfo.hpp"
#include "pipelinedefinitionstatus.hpp"
#include "pipelinedefinitionunloadguard.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

class ModelManager;
class Pipeline;

using tensor_map_t = std::map<std::string, std::shared_ptr<TensorInfo>>;

class PipelineDefinition {
    struct ValidationResultNotifier {
        ValidationResultNotifier(PipelineDefinitionStatus& status, std::condition_variable& loadedNotify) :
            status(status),
            loadedNotify(loadedNotify) {}
        ~ValidationResultNotifier() {
            if (passed) {
                status.handle(ValidationPassedEvent());
                loadedNotify.notify_all();
            } else {
                status.handle(ValidationFailedEvent());
            }
        }
        bool passed = false;

    private:
        PipelineDefinitionStatus& status;
        std::condition_variable& loadedNotify;
    };

    const std::string pipelineName;
    std::vector<NodeInfo> nodeInfos;
    pipeline_connections_t connections;

    std::atomic<uint64_t> requestsHandlesCounter = 0;
    std::shared_mutex loadMtx;

    std::condition_variable loadedNotify;

    // Pipelines are not versioned and any available definition has constant version equal 1.
    static constexpr model_version_t VERSION = 1;

protected:
    PipelineDefinitionStatus status;

private:
    std::set<std::pair<const std::string, model_version_t>> subscriptions;

    Status validateNode(ModelManager& manager, const NodeInfo& node, const bool isMultiBatchAllowed);

    const NodeInfo& findNodeByName(const std::string& name) const;
    shape_t getNodeGatherShape(const NodeInfo& info) const;

public:
    static constexpr uint64_t WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS = 10000;
    PipelineDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections) :
        pipelineName(pipelineName),
        nodeInfos(nodeInfos),
        connections(connections),
        status(this->pipelineName) {}

    Status create(std::unique_ptr<Pipeline>& pipeline,
        const tensorflow::serving::PredictRequest* request,
        tensorflow::serving::PredictResponse* response,
        ModelManager& manager);
    Status reload(ModelManager& manager, const std::vector<NodeInfo>&& nodeInfos, const pipeline_connections_t&& connections);
    void retire(ModelManager& manager);
    Status validate(ModelManager& manager);
    Status validateNodes(ModelManager& manager);
    Status validateForCycles();
    const std::string& getName() const { return pipelineName; }
    const PipelineDefinitionStateCode getStateCode() const { return status.getStateCode(); }
    const model_version_t getVersion() const { return VERSION; }

    void notifyUsedModelChanged(const std::string& ownerDetails) {
        this->status.handle(UsedModelChangedEvent(ownerDetails));
    }

    const PipelineDefinitionStatus& getStatus() const {
        return this->status;
    }

    const std::vector<NodeInfo>& getNodeInfos() {
        return this->nodeInfos;
    }

    void makeSubscriptions(ModelManager& manager);
    void resetSubscriptions(ModelManager& manager);

    virtual Status getInputsInfo(tensor_map_t& inputsInfo, const ModelManager& manager) const;
    virtual Status getOutputsInfo(tensor_map_t& outputsInfo, const ModelManager& manager) const;

    static Status getCustomNodeMetadata(const NodeInfo& customNodeInfo, tensor_map_t& inputsInfo, metadata_fn callback, const std::string& pipelineName);

    Status populateOutputsInfoWithDLModelOutputs(
        const NodeInfo& dependencyNodeInfo,
        const ModelManager& manager,
        tensor_map_t& outputsInfo,
        const Aliases& aliases,
        const shape_t& gatherShape) const;

    Status populateOutputsInfoWithCustomNodeOutputs(
        const NodeInfo& dependencyNodeInfo,
        const ModelManager& manager,
        tensor_map_t& outputsInfo,
        const Aliases& aliases,
        const shape_t& gatherShape) const;

    void increaseRequestsHandlesCount() {
        ++requestsHandlesCounter;
    }

    void decreaseRequestsHandlesCount() {
        --requestsHandlesCounter;
    }

    Status waitForLoaded(std::unique_ptr<PipelineDefinitionUnloadGuard>& unloadGuard, const uint waitForLoadedTimeoutMicroseconds = WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS);
};
}  // namespace ovms
