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
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma warning(push, 0)
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma warning(pop)

#include "pipeline.hpp"
#include "status.hpp"

namespace ovms {

class ModelManager;

using pipeline_connections_t = std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>>;

enum class NodeKind {
    ENTRY,
    DL,
    EXIT
};

const std::string DL_NODE_CONFIG_TYPE = "DL model";

Status toNodeKind(const std::string& str, NodeKind& nodeKind);

struct NodeInfo {
    NodeKind kind;
    std::string nodeName;
    std::string modelName;
    std::optional<model_version_t> modelVersion;
    std::unordered_map<std::string, std::string> outputNameAliases;

    NodeInfo(NodeKind kind,
        const std::string& nodeName,
        const std::string& modelName = "",
        std::optional<model_version_t> modelVersion = std::nullopt,
        std::unordered_map<std::string, std::string> outputNameAliases = {}) :
        kind(kind),
        nodeName(nodeName),
        modelName(modelName),
        modelVersion(modelVersion),
        outputNameAliases(outputNameAliases) {}
};

class PipelineDefinition {
    std::string pipelineName;
    std::vector<NodeInfo> nodeInfos;
    pipeline_connections_t connections;

private:
    Status validateNode(ModelManager& manager, NodeInfo& node);

public:
    PipelineDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections) :
        pipelineName(pipelineName),
        nodeInfos(nodeInfos),
        connections(connections) {}

    Status create(std::unique_ptr<Pipeline>& pipeline,
        const tensorflow::serving::PredictRequest* request,
        tensorflow::serving::PredictResponse* response,
        ModelManager& manager) const;

    Status validateNodes(ModelManager& manager);
    Status validateForCycles();
};

class PipelineFactory {
    std::map<std::string, std::unique_ptr<PipelineDefinition>> definitions;
    mutable std::shared_mutex definitionsMtx;

public:
    Status createDefinition(const std::string& pipelineName,
        const std::vector<NodeInfo>& nodeInfos,
        const pipeline_connections_t& connections,
        ModelManager& manager);

    bool definitionExists(const std::string& name) const {
        std::shared_lock lock(definitionsMtx);
        return definitions.count(name) == 1;
    }

    Status create(std::unique_ptr<Pipeline>& pipeline,
        const std::string& name,
        const tensorflow::serving::PredictRequest* request,
        tensorflow::serving::PredictResponse* response,
        ModelManager& manager) const;
};

}  // namespace ovms
