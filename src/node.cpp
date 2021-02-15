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
#include <set>
#include <sstream>
#include <vector>

#include "logging.hpp"
#include "nodesession.hpp"
#include "ov_utils.hpp"
#include "status.hpp"

namespace ovms {

Node::Node(const std::string& nodeName, uint32_t demultiplyCount, std::set<std::string> gatherFromNode) :
    nodeName(nodeName),
    demultiplexCount(demultiplyCount ? std::optional<uint32_t>(demultiplyCount) : std::nullopt),
    gatherFrom(!gatherFromNode.empty() ? std::optional<std::set<std::string>>(gatherFromNode) : std::nullopt) {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will create node: {} with demultiply: {}, gatherFrom: {}.",
        getName(),
        demultiplyCount,
        std::accumulate(gatherFromNode.begin(), gatherFromNode.end(), std::string(), [](const std::string& lhs, const std::string& rhs) { return lhs + ", " + rhs; }));
}

Status Node::fetchResults(session_key_t sessionId, SessionResults& nodeSessionOutputs) {
    auto it = nodeSessions.find(sessionId);
    auto& nodeSession = it->second;
    if (it == nodeSessions.end()) {
        return StatusCode::UNKNOWN_ERROR;
    }
    auto status = fetchResults(*nodeSession, nodeSessionOutputs);
    if (status.ok() && demultiplexCount) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will demultiply node: {} outputs to: {} shards", getName(), demultiplexCount.value());
        status = demultiplyOutputs(nodeSessionOutputs);
    }
    nodeSessions.erase(sessionId);
    return status;
}

void Node::printNodeConnections(const std::string& nodeName, const std::string& sourceNode, const Aliases& pairs) {
    std::stringstream ss;
    ss << "Links from:" << sourceNode << " to:" << nodeName << ":\n";
    for (auto& pair : pairs) {
        ss << "\t" << nodeName << "[" << pair.second << "]=" << sourceNode << "[" << pair.first << "]\n";
    }
    SPDLOG_DEBUG(ss.str());
}

Status Node::setInputs(const Node& dependency, SessionResults& sessionResults) {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "node: {} set inputs from node: {}", getName(), dependency.getName());
    for (auto& [sessionKey, metadataInputsPair] : sessionResults) {
        auto& [metadata, inputs] = metadataInputsPair;
        auto status = this->setInputs(dependency, inputs, metadata);
        if (!status.ok()) {
            return status;
        }
    }
    return StatusCode::OK;
}

Status Node::setInputs(const Node& dependency, BlobMap& inputs, NodeSessionMetadata& metadata) {
    // mapping for dependency - keeps mapping between dependency output name and this node input name
    const auto& mapping_for_dependency = this->getMappingByDependency(dependency);
    NodeSession& nodeSession = getNodeSession(metadata);
    const session_id_t shardId = metadata.getShardId(gatherFrom.value_or(std::set<std::string>()));
    // assign all input blobs from inputs that are required by this node for future inference
    for (const auto& pair : mapping_for_dependency) {
        const auto& dependency_output_name = pair.first;
        const auto& current_node_input_name = pair.second;

        // possibly incorrectly constructed pipeline - required input missing from previous node
        auto it = inputs.find(dependency_output_name);
        if (it == inputs.end()) {
            SPDLOG_LOGGER_WARN(dag_executor_logger, "node: {} error setting required input from node: {} dependency is missing output name: {}",
                getName(),
                dependency.getName(),
                dependency_output_name);
            return StatusCode::INVALID_MISSING_INPUT;
        }
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "node: {} setting required input from node: {}, input name: {}, dependency output name: {}",
            getName(),
            dependency.getName(),
            current_node_input_name,
            dependency_output_name);
        auto status = nodeSession.setInput(current_node_input_name, it->second, shardId);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "node: {} failed to set input: {}, shard: {}", getName(), current_node_input_name, shardId);
            return status;
        }
    }
    return nodeSession.notifyFinishedDependency();
}

NodeSession& Node::getNodeSession(const session_key_t& sessionKey) const {
    auto it = nodeSessions.find(sessionKey);
    if (it == nodeSessions.end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to get non-existing node: {} session: {}.", getName(), sessionKey);
        throw std::logic_error("Tried to get non existing session");  // TODO some other kind of error
    }
    return *(*it).second;
}

NodeSession& Node::getNodeSession(const NodeSessionMetadata& metadata) {
    // TODO check for exit node if no session levels remains
    session_key_t sessionKey;
    if (gatherFrom) {
        sessionKey = metadata.getSessionKey(gatherFrom.value());
    } else {
        sessionKey = metadata.getSessionKey();
    }
    auto it = nodeSessions.find(sessionKey);
    if (it != nodeSessions.end()) {
        return *(*it).second;
    }
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will create new session: {} for node: {}",
        metadata.getSessionKey(), getName());
    NodeSessionMetadata newSessionMetadata;
    CollapseDetails collapsingDetails;
    if (gatherFrom) {
        std::tie(newSessionMetadata, collapsingDetails) = metadata.getCollapsedSessionMetadata(gatherFrom.value());
    } else {
        newSessionMetadata = metadata;
    }
    std::unique_ptr<NodeSession> nodeSession = createNodeSession(newSessionMetadata, collapsingDetails);
    auto emplacePair = nodeSessions.emplace(sessionKey, std::move(nodeSession));
    return *(emplacePair.first->second);
}

std::unique_ptr<NodeSession> Node::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) {
    return std::make_unique<NodeSession>(metadata, getName(), previous.size(), collapsingDetails);
}

std::vector<session_key_t> Node::getReadySessions() const {
    std::vector<session_key_t> readySessions;
    for (auto& [sessionKey, nodeSession] : nodeSessions) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Checking readiness of node: {} session: {}", getName(), nodeSession->getSessionKey());
        if (nodeSession->isReady()) {
            readySessions.emplace_back(sessionKey);
        }
    }
    return std::move(readySessions);
}

Status Node::demultiplyOutputs(SessionResults& nodeSessionOutputs) {
    auto& [metadata, blobMap] = nodeSessionOutputs.begin()->second;
    std::vector<NodeSessionMetadata> newSessionMetadatas(metadata.generateSubsessions(getName(), demultiplexCount.value()));
    for (auto& [blobName, blob] : blobMap) {
        auto tensorDesc = blob->getTensorDesc();
        auto newDims = tensorDesc.getDims();
        if (newDims.size() < 3) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Wrong number of dimensions: {} to demultiply. Must be at least 3", newDims.size());
            return StatusCode::PIPELINE_WRONG_NUMBER_OF_DIMENSIONS_TO_DEMULTIPLY;
        }
        if (newDims[1] != demultiplexCount.value()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Wrong dim[1] size: {} expected: {} to demultiply",
                newDims[1], demultiplexCount.value());
            return StatusCode::PIPELINE_WRONG_DIMENSION_SIZE_TO_DEMULTIPLY;
        }
        newDims.erase(newDims.begin() + 1);
        const InferenceEngine::TensorDesc dividedBlobDesc(
            tensorDesc.getPrecision(),
            newDims,
            InferenceEngine::Layout::ANY);
        const auto step = blob->byteSize() / demultiplexCount.value();
        for (size_t i = 0; i < newSessionMetadatas.size(); ++i) {
            InferenceEngine::Blob::Ptr dividedBlob;
            auto status = createSharedBlob(dividedBlob, dividedBlobDesc);
            if (!status.ok()) {
                return status;
            }
            if (dividedBlob->byteSize() != step) {
                SPDLOG_LOGGER_ERROR(dag_executor_logger, "Created blob have wrong byte size: {}, expected: {}",
                    dividedBlob->byteSize(), step);
                return StatusCode::UNKNOWN_ERROR;
            }
            memcpy((char*)dividedBlob->buffer(), (char*)blob->buffer() + i * step, step);
            auto it = nodeSessionOutputs.find(newSessionMetadatas[i].getSessionKey());
            if (it == nodeSessionOutputs.end()) {
                nodeSessionOutputs.emplace(newSessionMetadatas[i].getSessionKey(), SessionResult{newSessionMetadatas[i], BlobMap{{blobName, dividedBlob}}});
            } else {
                it->second.second.emplace(blobName, dividedBlob);
            }
        }
    }
    nodeSessionOutputs.erase(metadata.getSessionKey());
    return StatusCode::OK;
}

}  // namespace ovms
