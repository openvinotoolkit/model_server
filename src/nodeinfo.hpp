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

#include "model_version_policy.hpp"
#include "node.hpp"  // TODO remove
#include "pipelinedefinitionstatus.hpp"
#include "pipelinedefinitionunloadguard.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

class ModelManager;
class Pipeline;

using pipeline_connections_t = std::unordered_map<std::string, std::unordered_map<std::string, InputPairs>>;
using InputPairs = std::vector<std::pair<std::string, std::string>>;
using tensor_map_t = std::map<std::string, std::shared_ptr<TensorInfo>>;

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
}  // namespace ovms
