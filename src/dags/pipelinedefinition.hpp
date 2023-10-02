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
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../modelversion.hpp"
#include "../tensorinfo.hpp"
#include "aliases.hpp"
#include "nodeinfo.hpp"
#include "pipelinedefinitionstatus.hpp"

namespace ovms {
class CNLIMWrapper;
class MetricConfig;
class MetricRegistry;
class ModelManager;
class ServableMetricReporter;
class NodeValidator;
class Pipeline;
class PipelineDefinitionUnloadGuard;
class Status;

class PipelineDefinition {
    friend NodeValidator;
    friend PipelineDefinitionUnloadGuard;
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
    std::map<std::string, std::shared_ptr<CNLIMWrapper>> nodeResources = {};
    pipeline_connections_t connections;

protected:
    tensor_map_t inputsInfo;
    tensor_map_t outputsInfo;

private:
    mutable std::shared_mutex metadataMtx;
    std::atomic<uint64_t> requestsHandlesCounter = 0;
    std::condition_variable loadedNotify;

    // Pipelines are not versioned and any available definition has constant version equal 1.
    static constexpr model_version_t VERSION = 1;

    std::unique_ptr<ServableMetricReporter> reporter;

protected:
    PipelineDefinitionStatus status;

private:
    std::set<std::pair<const std::string, model_version_t>> subscriptions;

    Status validateNode(ModelManager& manager, const NodeInfo& node, const bool isMultiBatchAllowed);

    const NodeInfo& findNodeByName(const std::string& name) const;
    Shape getNodeGatherShape(const NodeInfo& info) const;

public:
    static constexpr uint64_t WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS = 500000;
    PipelineDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr);
    template <typename RequestType, typename ResponseType>
    Status create(std::unique_ptr<Pipeline>& pipeline,
        const RequestType* request,
        ResponseType* response,
        ModelManager& manager);

private:
    template <typename RequestType, typename ResponseType>
    Status createPrivate(std::unique_ptr<Pipeline>& pipeline,
        const RequestType* request,
        ResponseType* response,
        ModelManager& manager);

public:
    Status reload(ModelManager& manager, const std::vector<NodeInfo>&& nodeInfos, const pipeline_connections_t&& connections);
    void retire(ModelManager& manager);
    Status validate(ModelManager& manager);
    Status validateNodes(ModelManager& manager);
    Status validateForCycles();
    Status validateDemultiplexerGatherNodesOrder();
    Status initializeNodeResources(ModelManager& manager);
    std::vector<NodeInfo> calculateNodeInfosDiff(const std::vector<NodeInfo>& nodeInfos);
    void deinitializeNodeResources(const std::vector<NodeInfo>& nodeInfosDiff);

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

    ServableMetricReporter& getMetricReporter() const { return *this->reporter; }

protected:
    Status updateInputsInfo(const ModelManager& manager);
    Status updateOutputsInfo(const ModelManager& manager);

public:
    const tensor_map_t getInputsInfo() const;
    const tensor_map_t getOutputsInfo() const;

private:
    static Status getCustomNodeMetadata(const NodeInfo& customNodeInfo, tensor_map_t& inputsInfo, metadata_fn callback, const std::string& pipelineName, void* customNodeLibraryInternalManager);

    Status populateOutputsInfoWithDLModelOutputs(
        const NodeInfo& dependencyNodeInfo,
        const ModelManager& manager,
        tensor_map_t& outputsInfo,
        const Aliases& aliases,
        const Shape& gatherShape) const;

    Status populateOutputsInfoWithCustomNodeOutputs(
        const NodeInfo& dependencyNodeInfo,
        const ModelManager& manager,
        tensor_map_t& outputsInfo,
        const Aliases& aliases,
        const Shape& gatherShape) const;

    void increaseRequestsHandlesCount() {
        ++requestsHandlesCounter;
    }

    void decreaseRequestsHandlesCount() {
        --requestsHandlesCounter;
    }

public:
    static const std::string SCHEDULER_CLASS_NAME;
    Status waitForLoaded(std::unique_ptr<PipelineDefinitionUnloadGuard>& unloadGuard, const uint waitForLoadedTimeoutMicroseconds = WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS);
};
}  // namespace ovms
