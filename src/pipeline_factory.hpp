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
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "pipeline.hpp"
#include "modelmanager.hpp"

namespace ovms {

enum class NodeKind {
    ENTRY,
    DL,
    EXIT
};

struct NodeInfo {
    NodeKind kind;
    std::string nodeName;
    std::string modelName;
    std::optional<model_version_t> modelVersion;

    NodeInfo(NodeKind kind,
             const std::string& nodeName,
             const std::string& modelName = "",
             std::optional<model_version_t> modelVersion = std::nullopt) :
        kind(kind),
        nodeName(nodeName),
        modelName(modelName),
        modelVersion(modelVersion) {}
};

class PipelineDefinition {
    std::string pipelineName;
    std::vector<NodeInfo> nodeInfos;
    std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>> connections;

public:
    PipelineDefinition(const std::vector<NodeInfo>& nodeInfos,
                       const std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>>& connections) :
        nodeInfos(nodeInfos),
        connections(connections) {}

    Status create(std::unique_ptr<Pipeline>& pipeline,
                  const tensorflow::serving::PredictRequest* request,
                  tensorflow::serving::PredictResponse* response,
                  ModelManager& manager) const;

    // TODO: validate method for one entry, one exit, acyclic, connected, no dead ends
    // https://jira.devtools.intel.com/browse/CVS-34360
};

class PipelineFactory {
    std::map<std::string, std::unique_ptr<PipelineDefinition>> definitions;

public:
    Status createDefinition(const std::string& pipelineName,
                            const std::vector<NodeInfo>& nodeInfos,
                            const std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>>& connections);

    bool definitionExists(const std::string& name) const {
        return definitions.count(name) == 1;
    }

    Status create(std::unique_ptr<Pipeline>& pipeline,
                  const std::string& name,
                  tensorflow::serving::PredictRequest* request,
                  tensorflow::serving::PredictResponse* response,
                  ModelManager& manager = ModelManager::getInstance()) const;
};

}  // namespace ovms
