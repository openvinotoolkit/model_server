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

#include "nodeinfo.hpp"

namespace ovms {

class DagResourceManager;
class MetricConfig;
class MetricRegistry;
class ModelInstanceProvider;
struct NodeInfo;
class Pipeline;
class PipelineDefinition;
class ServableNameChecker;
class Status;

class PipelineFactory {
    std::map<std::string, std::unique_ptr<PipelineDefinition>> definitions;
    mutable std::shared_mutex definitionsMtx;

public:
    Status createDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections,
        ModelInstanceProvider& modelInstanceProvider,
        ServableNameChecker& nameChecker,
        DagResourceManager& resourceMgr,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr);

    bool definitionExists(const std::string& name) const;

private:
    template <typename RequestType, typename ResponseType>
    Status createInternal(std::unique_ptr<Pipeline>& pipeline,
        const std::string& name,
        const RequestType* request,
        ResponseType* response,
        ModelInstanceProvider& modelInstanceProvider) const;

public:
    template <typename RequestType, typename ResponseType>
    Status create(std::unique_ptr<Pipeline>& pipeline,
        const std::string& name,
        const RequestType* request,
        ResponseType* response,
        ModelInstanceProvider& modelInstanceProvider) const;

    PipelineDefinition* findDefinitionByName(const std::string& name) const;
    Status reloadDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>&& nodeInfos,
        const pipeline_connections_t&& connections,
        ModelInstanceProvider& modelInstanceProvider,
        ServableNameChecker& nameChecker,
        DagResourceManager& resourceMgr);

    void retireOtherThan(std::set<std::string>&& pipelinesInConfigFile, ModelInstanceProvider& modelInstanceProvider);
    Status revalidatePipelines(ModelInstanceProvider& modelInstanceProvider, ServableNameChecker& nameChecker, DagResourceManager& resourceMgr);
    const std::vector<std::string> getPipelinesNames() const;
    const std::vector<std::string> getNamesOfAvailablePipelines() const;
};

}  // namespace ovms
