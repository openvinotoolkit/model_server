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
#include "node_library.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

class ModelManager;
class Pipeline;

using pipeline_connections_t = std::unordered_map<std::string, std::unordered_map<std::string, Aliases>>;
using tensor_map_t = std::map<std::string, std::shared_ptr<TensorInfo>>;
using parameters_t = std::unordered_map<std::string, std::string>;

enum class NodeKind {
    ENTRY,
    DL,
    CUSTOM,
    EXIT
};

const std::string DL_NODE_CONFIG_TYPE = "DL model";
const std::string CUSTOM_NODE_CONFIG_TYPE = "custom";

Status toNodeKind(const std::string& str, NodeKind& nodeKind);

struct DLNodeInfo {
    std::string modelName;
    std::optional<model_version_t> modelVersion;
};

struct CustomNodeInfo {
    NodeLibrary library;
    parameters_t parameters;
};

struct NodeInfo {
    NodeKind kind;
    std::string nodeName;
    std::string modelName;
    std::optional<model_version_t> modelVersion;
    std::unordered_map<std::string, std::string> outputNameAliases;
    std::optional<size_t> demultiplyCount;
    std::set<std::string> gatherFromNode;
    NodeLibrary library;
    parameters_t parameters;

    NodeInfo(NodeKind kind,
        const std::string& nodeName,
        const std::string& modelName = "",
        std::optional<model_version_t> modelVersion = std::nullopt,
        std::unordered_map<std::string, std::string> outputNameAliases = {},
        std::optional<size_t> demultiplyCount = std::nullopt,
        const std::set<std::string>& gatherFromNode = {},
        const NodeLibrary& library = {},
        const parameters_t& parameters = {}) :
        kind(kind),
        nodeName(nodeName),
        modelName(modelName),
        modelVersion(modelVersion),
        outputNameAliases(outputNameAliases),
        demultiplyCount(demultiplyCount),
        gatherFromNode(gatherFromNode),
        library(library),
        parameters(parameters) {}
};
}  // namespace ovms
