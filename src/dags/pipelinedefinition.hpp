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

#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../model_metric_reporter.hpp"
#include "../modelversion.hpp"
#include "../notifyreceiver.hpp"
#include "../single_version_servable_definition.hpp"
#include "../tensorinfo.hpp"
#include "aliases.hpp"
#include "nodeinfo.hpp"
#include "pipelinedefinitionstatus.hpp"

namespace ovms {
struct CNLIMWrapper;
class DagResourceManager;
class MetricConfig;
class MetricRegistry;
class ModelInstanceProvider;
class NodeValidator;
class ServableNameChecker;
class Pipeline;
class Status;

class PipelineDefinition : public SingleVersionServableDefinition, public NotifyReceiver {
    friend NodeValidator;
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

    std::vector<NodeInfo> nodeInfos;
    std::map<std::string, std::shared_ptr<CNLIMWrapper>> nodeResources = {};
    pipeline_connections_t connections;

protected:
    tensor_map_t inputsInfo;
    tensor_map_t outputsInfo;

private:
    mutable std::shared_mutex metadataMtx;

    std::unique_ptr<ServableMetricReporter> reporter;

protected:
    PipelineDefinitionStatus status;

private:
    std::set<std::pair<const std::string, model_version_t>> subscriptions;

    Status validateNode(ModelInstanceProvider& modelInstanceProvider, const NodeInfo& node, const bool isMultiBatchAllowed);

    const NodeInfo& findNodeByName(const std::string& name) const;
    Shape getNodeGatherShape(const NodeInfo& info) const;

public:
    PipelineDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr);
    template <typename RequestType, typename ResponseType>
    Status create(std::unique_ptr<Pipeline>& pipeline,
        const RequestType* request,
        ResponseType* response,
        ModelInstanceProvider& modelInstanceProvider);

public:
    Status reload(ModelInstanceProvider& modelInstanceProvider, ServableNameChecker& nameChecker, DagResourceManager& resourceMgr, const std::vector<NodeInfo>&& nodeInfos, const pipeline_connections_t&& connections);
    void retire(ModelInstanceProvider& modelInstanceProvider);
    Status validate(ModelInstanceProvider& modelInstanceProvider, ServableNameChecker& nameChecker, DagResourceManager& resourceMgr);
    Status validateNodes(ModelInstanceProvider& modelInstanceProvider);
    Status validateForCycles();
    Status validateDemultiplexerGatherNodesOrder();
    Status initializeNodeResources(DagResourceManager& resourceMgr);
    std::vector<NodeInfo> calculateNodeInfosDiff(const std::vector<NodeInfo>& nodeInfos);
    void deinitializeNodeResources(const std::vector<NodeInfo>& nodeInfosDiff);

    const std::string& getName() const override { return SingleVersionServableDefinition::getName(); }
    const PipelineDefinitionStateCode getStateCode() const { return status.getStateCode(); }
    bool isAvailable() const override { return status.isAvailable(); }

    void receiveNotification(const std::string& ownerDetails) override {
        this->status.handle(UsedModelChangedEvent(ownerDetails));
    }

    const PipelineDefinitionStatus& getStatus() const override {
        return this->status;
    }

    const std::vector<NodeInfo>& getNodeInfos() {
        return this->nodeInfos;
    }

    void makeSubscriptions(ModelInstanceProvider& modelInstanceProvider);
    void resetSubscriptions(ModelInstanceProvider& modelInstanceProvider);

    ServableMetricReporter& getMetricReporter() const override { return *this->reporter; }

protected:
    Status updateInputsInfo(const ModelInstanceProvider& modelInstanceProvider);
    Status updateOutputsInfo(const ModelInstanceProvider& modelInstanceProvider);

public:
    const tensor_map_t getInputsInfo() const override;
    const tensor_map_t getOutputsInfo() const override;

private:
    static Status getCustomNodeMetadata(const NodeInfo& customNodeInfo, tensor_map_t& inputsInfo, metadata_fn callback, const std::string& pipelineName, void* customNodeLibraryInternalManager);

    Status populateOutputsInfoWithDLModelOutputs(
        const NodeInfo& dependencyNodeInfo,
        const ModelInstanceProvider& modelInstanceProvider,
        tensor_map_t& outputsInfo,
        const Aliases& aliases,
        const Shape& gatherShape) const;

    Status populateOutputsInfoWithCustomNodeOutputs(
        const NodeInfo& dependencyNodeInfo,
        tensor_map_t& outputsInfo,
        const Aliases& aliases,
        const Shape& gatherShape) const;

    StatusCode notLoadedYetCode() const override;
    StatusCode notLoadedAnymoreCode() const override;

public:
    static const std::string SCHEDULER_CLASS_NAME;
};
}  // namespace ovms
