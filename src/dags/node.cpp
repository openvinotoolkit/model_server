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

#include "../logging.hpp"
#include "../ov_utils.hpp"
#include "../profiler.hpp"
#include "../shape.hpp"
#include "../status.hpp"
#include "nodesession.hpp"
#include "tensormap.hpp"

const uint64_t DEMULTIPLY_LIMIT = 10'000;

namespace ovms {
Node::~Node() = default;

static std::string demultiplyCountSettingToString(std::optional<int32_t> demultiplyCount) {
    if (!demultiplyCount) {
        return "NA";
    }
    if (demultiplyCount.value() == -1) {
        return "dynamic";
    }
    return std::to_string(demultiplyCount.value());
}

Node::Node(const std::string& nodeName, std::optional<int32_t> demultiplyCount, std::set<std::string> gatherFromNode) :
    nodeName(nodeName),
    demultiplexCount(demultiplyCount),
    gatherFrom(!gatherFromNode.empty() ? std::optional<std::set<std::string>>(gatherFromNode) : std::nullopt) {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will create node: {} with demultiply: {}, gatherFrom: {}.",
        getName(),
        demultiplyCountSettingToString(demultiplexCount),
        std::accumulate(gatherFromNode.begin(), gatherFromNode.end(), std::string("NA"), [](const std::string& lhs, const std::string& rhs) {
            if (lhs == "NA") {
                return rhs;
            } else {
                return lhs + ", " + rhs;
            } }));
}

Status Node::fetchResults(session_key_t sessionId, SessionResults& nodeSessionOutputs) {
    OVMS_PROFILE_FUNCTION();
    auto it = nodeSessions.find(sessionId);

    if (it == nodeSessions.end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Could not find session: {} for node: {}", sessionId, getName());
        return StatusCode::UNKNOWN_ERROR;
    }
    auto& nodeSession = it->second;
    auto status = fetchResults(*nodeSession, nodeSessionOutputs);
    if (status.ok() && demultiplexCount) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will demultiply node: {} outputs with demultiplyCount: {}", getName(), demultiplyCountSettingToString(demultiplexCount));
        status = demultiplyOutputs(nodeSessionOutputs);
    }
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will remove node: {} session: {}", getName(), sessionId);
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
    OVMS_PROFILE_FUNCTION();
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

Status Node::setInputs(const Node& dependency, TensorWithSourceMap& inputs, NodeSessionMetadata& metadata) {
    // mapping for dependency - keeps mapping between dependency output name and this node input name
    const auto& mapping_for_dependency = this->getMappingByDependency(dependency);
    NodeSession* nodeSession = getNodeSession(metadata);
    if (!nodeSession) {
        SPDLOG_ERROR("Failed to get node session for node: {}, session key: {}", getName(), metadata.getSessionKey());
        return StatusCode::INTERNAL_ERROR;
    }
    session_id_t shardId;
    try {
        static const std::set<std::string> emptySet;
        shardId = metadata.getShardId(gatherFrom.value_or(emptySet));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to get shardId for node: {}", getName());
        return StatusCode::INTERNAL_ERROR;
    }
    // assign all input tensors from inputs that are required by this node for future inference
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
        auto status = nodeSession->setInput(current_node_input_name, it->second, shardId);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "node: {} failed to set input: {}, shard: {}", getName(), current_node_input_name, shardId);
            return status;
        }
    }
    return nodeSession->notifyFinishedDependency();
}

NodeSession& Node::getNodeSession(const session_key_t& sessionKey) const {
    auto it = nodeSessions.find(sessionKey);
    if (it == nodeSessions.end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to get non-existing node: {} session: {}.", getName(), sessionKey);
        throw std::runtime_error("Tried to get non existing session");
    }
    return *it->second;
}

NodeSession* Node::getNodeSession(const NodeSessionMetadata& metadata) {
    session_key_t sessionKey;
    if (gatherFrom) {
        try {
            sessionKey = metadata.getSessionKey(gatherFrom.value());
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to create collapsed metadata session key for node: {}, incomming session key: {}",
                getName(), metadata.getSessionKey());
            return nullptr;
        }
    } else {
        sessionKey = metadata.getSessionKey();
    }
    auto it = nodeSessions.find(sessionKey);
    if (it != nodeSessions.end()) {
        return it->second.get();
    }
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will create new session: {} for node: {}",
        sessionKey, getName());
    NodeSessionMetadata newSessionMetadata = metadata;
    CollapseDetails collapsingDetails;
    if (gatherFrom) {
        try {
            std::tie(newSessionMetadata, collapsingDetails) = metadata.getCollapsedSessionMetadata(gatherFrom.value());
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to create collapsed metadata for node: {}", getName());
            return nullptr;
        }
    }
    std::unique_ptr<NodeSession> nodeSession = createNodeSession(newSessionMetadata, collapsingDetails);
    auto emplacePair = nodeSessions.emplace(sessionKey, std::move(nodeSession));
    return emplacePair.first->second.get();
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
    return readySessions;
}

Status Node::demultiplyOutputs(SessionResults& nodeSessionOutputs) {
    OVMS_PROFILE_FUNCTION();
    if (!demultiplexCount) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node: {} called demultiplyOutputs but node does not have demultiplexCount set", getName());
        return StatusCode::INTERNAL_ERROR;
    }
    auto& [metadata, tensorMap] = nodeSessionOutputs.begin()->second;
    auto firstTensorShape = tensorMap.begin()->second.getActualTensor().get_shape();
    uint32_t resultsDemultiplyCount = firstTensorShape[0];
    if (firstTensorShape[0] > DEMULTIPLY_LIMIT) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node: {} - too large dim[0] size: {} of tensor: {}. Maximum allowed is: {}",
            getName(), firstTensorShape[0], tensorMap.begin()->first, DEMULTIPLY_LIMIT);
        return StatusCode::PIPELINE_TOO_LARGE_DIMENSION_SIZE_TO_DEMULTIPLY;
    }
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will demultiply node: {} outputs to: {} shards", getName(), resultsDemultiplyCount);
    std::vector<NodeSessionMetadata> newSessionMetadatas;
    try {
        newSessionMetadatas = std::move(metadata.generateSubsessions(getName(), resultsDemultiplyCount));
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node: {} failed to generate subsessions due to error: {}", getName(), e.what());
        return StatusCode::INTERNAL_ERROR;
    }
    for (auto& [tensorName, tensorWithSource] : tensorMap) {
        auto& tensor = tensorWithSource.getActualTensor();
        OVMS_PROFILE_SCOPE("Demultiply Tensor");
        auto newDims = tensor.get_shape();
        if (newDims.size() < 3) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Wrong number of dimensions: {} to demultiply. Must be at least 3", newDims.size());
            return StatusCode::PIPELINE_WRONG_NUMBER_OF_DIMENSIONS_TO_DEMULTIPLY;
        }
        if ((demultiplexCount.value() != -1) &&
            (newDims[0] != static_cast<size_t>(demultiplexCount.value()))) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Wrong dim[0] size: {} of tensor: {} expected: {} to demultiply",
                newDims[0], tensorName, demultiplexCount.value());
            return StatusCode::PIPELINE_WRONG_DIMENSION_SIZE_TO_DEMULTIPLY;
        }
        if (resultsDemultiplyCount == 0) {
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} has no results. Dynamic demultiplexer with demultiply == 0 is not supported yet.", this->getName());
            nodeSessionOutputs.erase(metadata.getSessionKey());
            return StatusCode::PIPELINE_DEMULTIPLEXER_NO_RESULTS;
        }

        newDims.erase(newDims.begin());
        const auto step = tensor.get_byte_size() / resultsDemultiplyCount;
        for (size_t i = 0; i < newSessionMetadatas.size(); ++i) {
            OVMS_PROFILE_SCOPE("Create Shard");
            ov::Tensor dividedTensor;
            this->createShardedTensor(dividedTensor, ovElementTypeToOvmsPrecision(tensor.get_element_type()), newDims, tensor, i, step, metadata, tensorName);
            if (dag_executor_logger->level() <= spdlog::level::debug) {
                std::stringstream ss;
                ss << "Node: " << getName() << " input demultiplied: " << tensorName
                   << "; Actual: " << shapeToString(dividedTensor.get_shape());
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "{}", ss.str());
            }
            auto sessionKey = newSessionMetadatas[i].getSessionKey();
            auto it = nodeSessionOutputs.find(sessionKey);
            if (it == nodeSessionOutputs.end()) {
                nodeSessionOutputs.emplace(sessionKey, SessionResult{newSessionMetadatas[i], TensorWithSourceMap{{tensorName, TensorWithSource{dividedTensor, tensor}}}});
            } else {
                it->second.second.emplace(tensorName, TensorWithSource{dividedTensor, tensor});
            }
        }
    }
    nodeSessionOutputs.erase(metadata.getSessionKey());
    return StatusCode::OK;
}

Status Node::createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName) {
    dividedTensor = createSharedTensor(tensor.get_element_type(), shape, (char*)(tensor.data()) + i * step);
    return StatusCode::OK;
}
}  // namespace ovms
