//*****************************************************************************
// Copyright 2020,2021 Intel Corporation
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
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "../precision.hpp"
#include "../shape.hpp"
#include "aliases.hpp"
#include "nodesessionresult.hpp"
#include "pipelineeventqueue.hpp"
#include "tensormap.hpp"

namespace ovms {

using TensorNames = std::vector<std::string>;
using session_key_t = std::string;

class NodeSession;
class NodeSessionMetadata;
class Status;

class Node {
protected:
    std::string nodeName;

    std::vector<std::reference_wrapper<Node>> previous;
    std::vector<std::reference_wrapper<Node>> next;

    // Tensors ready and waiting for execution
    std::unordered_map<session_key_t, std::unique_ptr<NodeSession>> nodeSessions;

    // Input/Output name mapping and list of required inputs from previous nodes
    std::unordered_map<std::string, Aliases> tensorNamesMapping;

    const std::optional<int32_t> demultiplexCount;
    const std::optional<std::set<std::string>> gatherFrom;

public:
    Node(const std::string& nodeName, std::optional<int32_t> demultiplyCount = std::nullopt, std::set<std::string> gatherFromNode = {});

    virtual ~Node();

    const std::string& getName() const { return this->nodeName; }

    virtual Status execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) = 0;
    Status fetchResults(session_key_t sessionId, SessionResults& nodeSessionOutputs);

protected:
    virtual Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) = 0;
    Status demultiplyOutputs(SessionResults& nodeSessionOutputs);
    virtual Status createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName);

public:
    Status setInputs(const Node& dependency, TensorWithSourceMap& inputs, NodeSessionMetadata& metadata);
    Status setInputs(const Node& dependency, SessionResults& inputs);

    virtual void addDependency(Node& node, const Aliases& tensorNamesMapping) {
        this->previous.emplace_back(node);
        this->tensorNamesMapping[node.getName()] = tensorNamesMapping;
    }

    virtual void addDependant(Node& node) { this->next.emplace_back(node); }

    const Aliases& getMappingByDependency(const Node& dependency) {
        return tensorNamesMapping.at(dependency.getName());
    }

    std::vector<session_key_t> getReadySessions() const;
    const std::vector<std::reference_wrapper<Node>>& getNextNodes() {
        return next;
    }
    virtual void release(session_key_t sessionId) {}
    virtual bool tryDisarm(const session_key_t& sessionKey, const uint32_t microseconds = 1) { return true; }

    static void printNodeConnections(const std::string& nodeName, const std::string& sourceNode, const Aliases& pairs);

    NodeSession* getNodeSession(const NodeSessionMetadata& metadata);

protected:
    NodeSession& getNodeSession(const session_key_t& sessionKey) const;
    virtual std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails);
};

}  // namespace ovms
