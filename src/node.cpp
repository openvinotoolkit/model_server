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
#include "node.hpp"

#include <algorithm>
#include <sstream>

#include "logging.hpp"
#include "nodesession.hpp"
#include "status.hpp"

namespace ovms {

Node::Node(const std::string& nodeName) :
    nodeName(nodeName) {
}
Status Node::fetchResults(session_key_t sessionId, SessionResults& nodeSessionOutputs) {
    auto it = nodeSessions.find(sessionId);
    auto& nodeSession = it->second;
    if (it == nodeSessions.end()) {
        // TODO error handle
        return StatusCode::UNKNOWN_ERROR;
    }
    auto status = fetchResults(*nodeSession, nodeSessionOutputs);
    // TODO outputhandler->postprocessResults();
    return status;
}

void Node::printNodeConnections(const std::string& nodeName, const std::string& sourceNode, const InputPairs& pairs) {
    std::stringstream ss;
    ss << "Links from:" << sourceNode << " to:" << nodeName << ":\n";
    for (auto& pair : pairs) {
        ss << "\t" << nodeName << "[" << pair.second << "]=" << sourceNode << "[" << pair.first << "]\n";
    }
    SPDLOG_DEBUG(ss.str());
}

Status Node::setInputs(const Node& dependency, SessionResults& sessionResults) {
    SPDLOG_ERROR("setInputs for metadata3, sessionResults:{}", sessionResults.size());
    for (auto& [sessionKey, metadataInputsPair] : sessionResults) {
        auto& [metadata, inputs] = metadataInputsPair;
        SPDLOG_ERROR("setInputs for metadata2");
        auto status = this->setInputs(dependency, inputs, metadata);  // TODO retcode handle
    }
    return StatusCode::OK;  // TODO
}

Status Node::setInputs(const Node& dependency, BlobMap& inputs, NodeSessionMetadata& metadata) {
    SPDLOG_ERROR("setInputs for metadata");
    // mapping for dependency - keeps mapping between dependency output name and this node input name
    const auto& mapping_for_dependency = this->getMappingByDependency(dependency);
    NodeSession& nodeSession = getNodeSession(metadata);
    // assign all input blobs from inputs that are required by this node for future inference
    for (const auto& pair : mapping_for_dependency) {
        const auto& dependency_output_name = pair.first;
        const auto& current_node_input_name = pair.second;

        // possibly incorrectly constructed pipeline - required input missing from previous node
        auto it = inputs.find(dependency_output_name);
        if (it == inputs.end()) {
            SPDLOG_LOGGER_WARN(dag_executor_logger, "Node::setInputs: error setting required input for (Node name {}) from (Node name {}): dependency is missing output name {}",
                getName(),
                dependency.getName(),
                dependency_output_name);
            return StatusCode::INVALID_MISSING_INPUT;
        }
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node::setInputs: setting required input for (Node name {}) from (Node name {}), input name: {}, dependency output name: {}",
            getName(),
            dependency.getName(),
            current_node_input_name,
            dependency_output_name);
        this->inputBlobs[current_node_input_name] = it->second;
        nodeSession.setInput(current_node_input_name, it->second);
    }
    SPDLOG_ERROR("Notify");
    nodeSession.notifyFinishedDependency();  // TODO gatherer need to change this mechanism
    finishedDependenciesCount++;
    return StatusCode::OK;
}

Status Node::setInputs(const Node& dependency, BlobMap& inputs) {
    SPDLOG_ERROR("setInputs for metadata4");
    // mapping for dependency - keeps mapping between dependency output name and this node input name
    const auto& mapping_for_dependency = this->getMappingByDependency(dependency);

    // assign all input blobs from inputs that are required by this node for future inference
    for (const auto& pair : mapping_for_dependency) {
        const auto& dependency_output_name = pair.first;
        const auto& current_node_input_name = pair.second;

        // possibly incorrectly constructed pipeline - required input missing from previous node
        auto it = inputs.find(dependency_output_name);
        if (it == inputs.end()) {
            SPDLOG_WARN("Node::setInputs: error setting required input for (Node name {}) from (Node name {}): dependency is missing output name {}",
                getName(),
                dependency.getName(),
                dependency_output_name);
            return StatusCode::INVALID_MISSING_INPUT;
        }
        SPDLOG_DEBUG("Node::setInputs: setting required input for (Node name {}) from (Node name {}), input name: {}, dependency output name: {}",
            getName(),
            dependency.getName(),
            current_node_input_name,
            dependency_output_name);
        this->inputBlobs[current_node_input_name] = it->second;
        NodeSessionMetadata metadata;
        getNodeSession(metadata).setInput(current_node_input_name, it->second);
    }

    finishedDependenciesCount++;
    return StatusCode::OK;
}

NodeSession& Node::getNodeSession(const session_key_t& sessionKey) const {
    auto it = nodeSessions.find(sessionKey);
    if (it == nodeSessions.end()) {
        throw std::logic_error("SOME");  // TODO some other kind of error
    }
    return *(*it).second;
}

NodeSession& Node::getNodeSession(const NodeSessionMetadata& metadata) {
    auto it = nodeSessions.find(metadata.getSessionKey());
    if (it != nodeSessions.end()) {
        return *(*it).second;
    }
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will create new session: {} for node: {}",
        metadata.getSessionKey(), getName());
    auto emplacePair = nodeSessions.emplace(metadata.getSessionKey(), createNodeSession(metadata));
    return *(*(emplacePair.first)).second;
}

std::unique_ptr<NodeSession> Node::createNodeSession(const NodeSessionMetadata& metadata) {
    return std::make_unique<NodeSession>(metadata, getName(), previous.size());
}

std::vector<session_key_t> Node::getReadySessions() const {
    std::vector<session_key_t> readySessions;
    for (auto& [sessionKey, nodeSession] : nodeSessions) {
        SPDLOG_ERROR("Checking readiness of node: {} session: {}", getName(), nodeSession->getSessionKey());  // TODO report readiness
        if (nodeSession->isReady()) {
            readySessions.emplace_back(sessionKey);
        }
    }
    return std::move(readySessions);
}

}  // namespace ovms
