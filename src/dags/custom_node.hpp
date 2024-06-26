//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include <set>
#include <string>
#include <unordered_map>

#include "node.hpp"
#include "node_library.hpp"
#include "nodeinfo.hpp"
#include "pipelineeventqueue.hpp"

namespace ovms {

class NodeLibrary;
class Status;
class CNLIMWrapper;

class CustomNode : public Node {
    NodeLibrary library;
    parameters_t parameters;
    std::unordered_map<std::string, std::string> nodeOutputNameAlias;

    std::unique_ptr<struct CustomNodeParam[]> libraryParameters = nullptr;

    std::shared_ptr<CNLIMWrapper> customNodeLibraryInternalManager;

public:
    CustomNode(
        const std::string& nodeName,
        const NodeLibrary& library,
        const parameters_t& parameters,
        const std::unordered_map<std::string, std::string>& nodeOutputNameAlias = {},
        std::optional<int32_t> demultiplyCount = std::nullopt,
        std::set<std::string> gatherFromNode = {},
        std::shared_ptr<CNLIMWrapper> customNodeLibraryInternalManager = nullptr);

    Status execute(session_key_t sessionKey, PipelineEventQueue& notifyEndQueue) override;

    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) override;
    Status fetchResults(TensorWithSourceMap& outputs, session_key_t sessionKey);

    const std::string& getRealOutputName(const std::string& alias) const {
        auto it = nodeOutputNameAlias.find(alias);
        return it != nodeOutputNameAlias.end() ? it->second : alias;
    }

    std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) override;
};

}  // namespace ovms
