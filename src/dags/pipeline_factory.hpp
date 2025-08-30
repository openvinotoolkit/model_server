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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "nodeinfo.hpp"

namespace ovms {

class ModelManager;
struct NodeInfo;
class Pipeline;
class PipelineDefinition;
class Status;

class PipelineFactory {
    std::map<std::string, std::unique_ptr<PipelineDefinition>> definitions;
    mutable std::shared_mutex definitionsMtx;

public:
    Status createDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections,
        ModelManager& manager);

    bool definitionExists(const std::string& name) const;

private:
    template <typename RequestType, typename ResponseType>
    Status createInternal(std::unique_ptr<Pipeline>& pipeline,
        const std::string& name,
        const RequestType* request,
        ResponseType* response,
        ModelManager& manager) const;

public:
    template <typename RequestType, typename ResponseType>
    Status create(std::unique_ptr<Pipeline>& pipeline,
        const std::string& name,
        const RequestType* request,
        ResponseType* response,
        ModelManager& manager) const;

    PipelineDefinition* findDefinitionByName(const std::string& name) const;
    Status reloadDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>&& nodeInfos,
        const pipeline_connections_t&& connections,
        ModelManager& manager);

    void retireOtherThan(std::set<std::string>&& pipelinesInConfigFile, ModelManager& manager);
    Status revalidatePipelines(ModelManager&);
    const std::vector<std::string> getPipelinesNames() const;

    std::shared_mutex& getDefinitionsMtx() const {
        return definitionsMtx;
    }
};

}  // namespace ovms
