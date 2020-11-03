
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
#include "pipelinedefinition.hpp"

#include <set>

#include "prediction_service_utils.hpp"

namespace ovms {

Status toNodeKind(const std::string& str, NodeKind& nodeKind) {
    if (str == DL_NODE_CONFIG_TYPE) {
        nodeKind = NodeKind::DL;
        return StatusCode::OK;
    }
    SPDLOG_ERROR("Unsupported node type:{}", str);
    return StatusCode::PIPELINE_NODE_WRONG_KIND_CONFIGURATION;
}

Status PipelineDefinition::create(std::unique_ptr<Pipeline>& pipeline,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    ModelManager& manager) const {
    std::unordered_map<std::string, std::unique_ptr<Node>> nodes;

    EntryNode* entry = nullptr;
    ExitNode* exit = nullptr;
    for (const auto& info : nodeInfos) {
        SPDLOG_DEBUG("Creating pipeline:{}. Adding nodeName:{}, modelName:{}",
            getName(), info.nodeName, info.modelName);
        switch (info.kind) {
        case NodeKind::ENTRY: {
            auto node = std::make_unique<EntryNode>(request);
            entry = node.get();
            nodes.insert(std::make_pair(info.nodeName, std::move(node)));
            break;
        }
        case NodeKind::DL:
            nodes.insert(std::make_pair(info.nodeName, std::move(std::make_unique<DLNode>(info.nodeName,
                                                           info.modelName,
                                                           info.modelVersion,
                                                           manager,
                                                           info.outputNameAliases))));
            break;
        case NodeKind::EXIT: {
            auto node = std::make_unique<ExitNode>(response);
            exit = node.get();
            nodes.insert(std::make_pair(info.nodeName, std::move(node)));
            break;
        }
        default:
            throw std::invalid_argument("unknown node kind");
        }
    }
    for (const auto& kv : connections) {
        const auto& dependantNode = nodes.at(kv.first);
        for (const auto& pair : kv.second) {
            const auto& dependencyNode = nodes.at(pair.first);
            SPDLOG_DEBUG("Connecting pipeline:{}, from:{}, to:{}", getName(), dependencyNode->getName(), dependantNode->getName());
            Pipeline::connect(*dependencyNode, *dependantNode, pair.second);
        }
    }
    pipeline = std::make_unique<Pipeline>(*entry, *exit, pipelineName);
    for (auto& kv : nodes) {
        pipeline->push(std::move(kv.second));
    }
    return StatusCode::OK;
}

void PipelineDefinition::resetSubscriptions(ModelManager& manager) {
    for (auto& [modelName, modelVersion] : subscriptions) {
        if (modelVersion) {
            SPDLOG_INFO("Unsubscribing pipeline:{} from model: {}, version:{}",
                getName(), modelName, modelVersion);
            manager.findModelByName(modelName)->getModelInstanceByVersion(modelVersion)->unsubscribe(*this);
        } else {  // using default version
            SPDLOG_INFO("Unsubscribing pipeline:{} from model: {}",
                getName(), modelName);
            manager.findModelByName(modelName)->unsubscribe(*this);
        }
    }
    subscriptions.clear();
}

void PipelineDefinition::makeSubscriptions(ModelManager& manager) {
    for (auto& node : nodeInfos) {
        if (node.kind == NodeKind::DL) {
            if (subscriptions.find({node.modelName, node.modelVersion.value_or(0)}) != subscriptions.end()) {
                continue;
            }
            std::unique_ptr<ModelInstanceUnloadGuard> nodeModelInstanceUnloadGuard;
            std::shared_ptr<ModelInstance> nodeModelInstance;
            auto status = getModelInstance(manager, node.modelName, node.modelVersion.value_or(0), nodeModelInstance,
                nodeModelInstanceUnloadGuard);
            if (!status.ok()) {
                std::stringstream ss;
                ss << "Pipeline: " << getName() << "Failed to make subscription to model: " << node.modelName;
                if (node.modelVersion) {
                    ss << " version: " << node.modelVersion.value();
                }
                SPDLOG_INFO(ss.str());
                continue;
            }
            if (node.modelVersion) {
                nodeModelInstance->subscribe(*this);
            } else {
                auto model = manager.findModelByName(node.modelName);
                model->subscribe(*this);
            }
            subscriptions.insert({node.modelName, node.modelVersion.value_or(0)});
        }
    }
}

Status PipelineDefinition::validateNode(ModelManager& manager, NodeInfo& node) {
    std::unique_ptr<ModelInstanceUnloadGuard> nodeModelInstanceUnloadGuard;
    std::shared_ptr<ModelInstance> nodeModelInstance;
    tensor_map_t nodeInputs;
    SPDLOG_DEBUG("Validation of pipeline: {} node: {} type: {}", getName(), node.nodeName, node.kind);

    Status result;
    if (node.kind == NodeKind::DL) {
        result = getModelInstance(manager, node.modelName, node.modelVersion.value_or(0), nodeModelInstance,
            nodeModelInstanceUnloadGuard);
        if (!result.ok()) {
            SPDLOG_ERROR("Validation of pipeline({}) definition failed. Missing model: {} version: {}", this->pipelineName, node.modelName, node.modelVersion.value_or(0));
            return StatusCode::MODEL_NAME_MISSING;
        }

        auto& config = nodeModelInstance->getModelConfig();
        if (config.getBatchingMode() == Mode::AUTO) {
            SPDLOG_ERROR("Validation of pipeline({}) definition failed. Node name {} used model name {} with dynamic batch size which is forbidden.", this->pipelineName, node.nodeName, node.modelName);
            return StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER;
        }

        auto& shapes = config.getShapes();
        for (auto& shape : shapes) {
            if (shape.second.shapeMode == Mode::AUTO) {
                SPDLOG_ERROR("Validation of pipeline({}) definition failed. Node name {} used model name {} with dynamic shape which is forbidden.", this->pipelineName, node.nodeName, node.modelName);
                return StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER;
            }
        }

        nodeInputs = nodeModelInstance->getInputsInfo();
    }

    for (auto& connection : connections[node.nodeName]) {
        std::unique_ptr<ModelInstanceUnloadGuard> sourceNodeModelInstanceUnloadGuard;
        const std::string& sourceNodeName = connection.first;
        auto findByName = [sourceNodeName](const NodeInfo& nodeInfo) {
            return nodeInfo.nodeName == sourceNodeName;
        };
        std::vector<NodeInfo>::iterator sourceNodeInfo = std::find_if(std::begin(nodeInfos), std::end(nodeInfos), findByName);
        if (sourceNodeInfo == std::end(nodeInfos)) {
            SPDLOG_ERROR("Validation of pipeline({}) definition failed. For node: {} missing dependency node: {} ", this->pipelineName, node.nodeName, sourceNodeName);
            return StatusCode::MODEL_NAME_MISSING;
        }

        if (sourceNodeInfo->kind == NodeKind::DL) {
            std::shared_ptr<ModelInstance> sourceNodeModelInstance;
            result = getModelInstance(manager, sourceNodeInfo->modelName, 0, sourceNodeModelInstance,
                sourceNodeModelInstanceUnloadGuard);
            if (!result.ok()) {
                SPDLOG_ERROR("Validation of pipeline({}) definition failed. Missing model: {} version: {}", this->pipelineName, sourceNodeInfo->modelName, sourceNodeInfo->modelVersion.value_or(0));
                return StatusCode::MODEL_MISSING;
            }
            const tensor_map_t& sourceNodeOutputs = sourceNodeModelInstance->getOutputsInfo();

            if (connection.second.size() == 0) {
                SPDLOG_ERROR("Validation of pipeline({}) definition failed. Missing dependency mapping for node: {}", this->pipelineName, node.nodeName);
                return StatusCode::PIPELINE_DEFINITION_MISSING_DEPENDENCY_MAPPING;
            }

            for (auto alias : connection.second) {
                std::string& dependencyOutputAliasName = alias.first;
                std::string dependencyOutputName;
                if (sourceNodeInfo->outputNameAliases.count(dependencyOutputAliasName)) {
                    dependencyOutputName = sourceNodeInfo->outputNameAliases[dependencyOutputAliasName];
                } else {
                    dependencyOutputName = dependencyOutputAliasName;
                }
                auto dependencyOutput = sourceNodeOutputs.find(dependencyOutputName);
                if (dependencyOutput == sourceNodeOutputs.end()) {
                    SPDLOG_ERROR("Validation of pipeline({}) definition failed. Missing output: {} of model: {}", this->pipelineName, dependencyOutputName, sourceNodeModelInstance->getName());
                    return StatusCode::INVALID_MISSING_OUTPUT;
                }

                if (node.kind != NodeKind::DL) {
                    break;
                }
                std::string& inputName = alias.second;
                auto nodeInput = nodeInputs.find(inputName);
                if (nodeInput == nodeInputs.end()) {
                    SPDLOG_ERROR("Validation of pipeline({}) definition failed. Missing input: {} of node: {}", this->pipelineName, inputName, node.nodeName);
                    return StatusCode::INVALID_MISSING_INPUT;
                }
            }
        }
    }
    return StatusCode::OK;
}

// Because of the way how pipeline_connections is implemented, this function is using
// transpose of PipelineDefinition graph.(Transpose contains same cycles as original graph)
Status PipelineDefinition::validateForCycles() {
    std::vector<std::string> visited;
    std::vector<std::string> parentNodes;
    visited.reserve(nodeInfos.size());
    parentNodes.reserve(nodeInfos.size());

    auto pred = [](const NodeInfo& nodeInfo) {
        return nodeInfo.kind == NodeKind::EXIT;
    };

    const auto& itr = std::find_if(std::begin(nodeInfos), std::end(nodeInfos), pred);
    if (itr == nodeInfos.end()) {
        SPDLOG_ERROR("Pipeline does not contain response node.");
        return StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT;
    }
    std::string nodeName = itr->nodeName;
    visited.push_back(nodeName);

    bool anyUnvisitedLeft = true;
    while (anyUnvisitedLeft) {
        bool unvisistedFound = false;
        const auto& connectedToNode = connections[nodeName];
        for (const auto& node : connectedToNode) {
            if (nodeName == node.first) {
                SPDLOG_ERROR("Node {} is connected to itself.", nodeName);
                return StatusCode::PIPELINE_CYCLE_FOUND;
            }

            if (std::find(visited.begin(), visited.end(), node.first) == visited.end()) {
                parentNodes.push_back(nodeName);
                visited.push_back(node.first);
                nodeName = node.first;
                unvisistedFound = true;
                break;
            } else {
                if (std::find(parentNodes.begin(), parentNodes.end(), node.first) != parentNodes.end()) {
                    std::string cycleNodes;
                    for (auto& cycleNode : parentNodes) {
                        cycleNodes += cycleNode;
                        if (cycleNode != parentNodes.back()) {
                            cycleNodes += ", ";
                        }
                    }
                    SPDLOG_ERROR("Following nodes creates cycle: {}", cycleNodes);
                    return StatusCode::PIPELINE_CYCLE_FOUND;
                }
            }
        }

        if (!unvisistedFound) {
            if (parentNodes.size() == 0) {
                anyUnvisitedLeft = false;
                if (visited.size() != nodeInfos.size()) {
                    SPDLOG_ERROR("There are nodes not connected to pipeline.");
                    return StatusCode::PIPELINE_CONTAINS_UNCONNECTED_NODES;
                }
            } else {
                nodeName = parentNodes.back();
                parentNodes.pop_back();
            }
        }
    }
    return StatusCode::OK;
}

Status PipelineDefinition::validateNodes(ModelManager& manager) {
    SPDLOG_DEBUG("Validation of pipeline definition:{} nodes started.", getName());
    bool entryFound = false;
    bool exitFound = false;
    for (auto& node : nodeInfos) {
        auto findByName = [node](const NodeInfo& nodeInfo) {
            return nodeInfo.nodeName == node.nodeName;
        };
        if (std::count_if(nodeInfos.begin(), nodeInfos.end(), findByName) > 1) {
            return StatusCode::PIPELINE_NODE_NAME_DUPLICATE;
        }
        auto result = validateNode(manager, node);
        if (result != StatusCode::OK) {
            return result;
        }
        if (node.kind == NodeKind::ENTRY) {
            if (entryFound) {
                return StatusCode::PIPELINE_MULTIPLE_ENTRY_NODES;
            }
            entryFound = true;
        }
        if (node.kind == NodeKind::EXIT) {
            if (exitFound) {
                return StatusCode::PIPELINE_MULTIPLE_EXIT_NODES;
            }
            exitFound = true;
        }
    }
    if (!entryFound) {
        SPDLOG_ERROR("PipelineDefinition: {} is missing request node", pipelineName);
        return StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT;
    }
    if (!exitFound) {
        SPDLOG_ERROR("PipelineDefinition: {} is missing response node", pipelineName);
        return StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT;
    }
    return StatusCode::OK;
}
}  // namespace ovms
