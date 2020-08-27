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
#include "pipeline_factory.hpp"

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
        if (nodes.count(info.nodeName) == 1) {
            return StatusCode::PIPELINE_NODE_NAME_DUPLICATE;
        }
        SPDLOG_DEBUG("Creating pipeline:{}. Adding nodeName:{}, modelName:{}",
            info.nodeName, info.modelName);
        switch (info.kind) {
        case NodeKind::ENTRY: {
            if (entry != nullptr) {
                return StatusCode::PIPELINE_MULTIPLE_ENTRY_NODES;
            }
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
            if (exit != nullptr) {
                return StatusCode::PIPELINE_MULTIPLE_EXIT_NODES;
            }
            auto node = std::make_unique<ExitNode>(response);
            exit = node.get();
            nodes.insert(std::make_pair(info.nodeName, std::move(node)));
            break;
        }
        default:
            throw std::invalid_argument("unknown node kind");
        }
    }
    if (!entry) {
        SPDLOG_INFO("Pipeline:{} is missing entry node", pipelineName);
        return StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT;
    }
    if (!exit) {
        SPDLOG_INFO("Pipeline:{} is missing exit node", pipelineName);
        return StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT;
    }
    for (const auto& kv : connections) {
        const auto& dependantNode = nodes.at(kv.first);
        for (const auto& pair : kv.second) {
            const auto& dependencyNode = nodes.at(pair.first);
            SPDLOG_DEBUG("Connecting from:{}, to:{}", dependencyNode->getName(), dependantNode->getName());
            Pipeline::connect(*dependencyNode, *dependantNode, pair.second);
        }
    }
    pipeline = std::make_unique<Pipeline>(*entry, *exit);
    for (auto& kv : nodes) {
        pipeline->push(std::move(kv.second));
    }
    return StatusCode::OK;
}

Status PipelineDefinition::validate(ModelManager& manager) {
    SPDLOG_DEBUG("Validation of pipeline definition started.");
    for (auto& node : nodeInfos) {
        std::unique_ptr<ModelInstanceUnloadGuard> nodeInstanceUnloadGuard;
        std::shared_ptr<ModelInstance> nodeInstance;
        tensor_map_t nodeInputs;
        SPDLOG_ERROR("Validation of node{}", node.nodeName);

        Status result;
        if (node.kind == NodeKind::DL) {
            result = getModelInstance(manager, node.modelName, node.modelVersion.value_or(0), nodeInstance,
                nodeInstanceUnloadGuard);
            if (!result.ok()) {
                SPDLOG_ERROR("Validation of pipeline definition failed. Missing model:{} version:{}", node.modelName, node.modelVersion.value_or(0));
                return StatusCode::MODEL_NAME_MISSING;
            }

            auto& config = nodeInstance->getModelConfig();
            if (config.getBatchingMode() == Mode::AUTO) {
                SPDLOG_ERROR("Validation of pipeline definition failed. Node name {} used model name {} with dynamic batch size which is forbidden.", node.nodeName, node.modelName);
                return StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER;
            }

            auto& shapes = config.getShapes();
            for (auto& shape : shapes) {
                if (shape.second.shapeMode == Mode::AUTO) {
                    SPDLOG_ERROR("Validation of pipeline definition failed. Node name {} used model name {} with dynamic shape which is forbidden.", node.nodeName, node.modelName);
                    return StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER;
                }
            }

            nodeInputs = nodeInstance->getInputsInfo();
        }

        for (auto& connection : connections[node.nodeName]) {
            std::unique_ptr<ModelInstanceUnloadGuard> sourceNodeInstanceUnloadGuard;
            const std::string& sourceNodeName = connection.first;
            auto pred = [sourceNodeName](const NodeInfo& nodeInfo) {
                return nodeInfo.nodeName == sourceNodeName;
            };

            std::vector<NodeInfo>::iterator sourceNodeInfo = std::find_if(std::begin(nodeInfos), std::end(nodeInfos), pred);
            if (sourceNodeInfo == std::end(nodeInfos)) {
                SPDLOG_ERROR("Validation of pipeline definition failed. For node:{} missing dependency node:{} ", node.nodeName, sourceNodeName);
                return StatusCode::MODEL_NAME_MISSING;
            }

            if (sourceNodeInfo->kind == NodeKind::DL) {
                std::shared_ptr<ModelInstance> sourceNodeInstance;
                result = getModelInstance(manager, sourceNodeInfo->modelName, 0, sourceNodeInstance,
                    sourceNodeInstanceUnloadGuard);
                if (!result.ok()) {
                    SPDLOG_ERROR("Validation of pipeline definition failed. Missing model:{} version:{}", sourceNodeInfo->modelName, sourceNodeInfo->modelVersion.value_or(0));
                    return StatusCode::MODEL_MISSING;
                }
                const tensor_map_t& sourceNodeOutputs = sourceNodeInstance->getOutputsInfo();

                if (connection.second.size() == 0) {
                    SPDLOG_ERROR("Validation of pipeline definition failed. Missing dependency mapping for node:{}", node.nodeName);
                    return StatusCode::INVALID_MISSING_INPUT;
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
                        SPDLOG_ERROR("Validation of pipeline definition failed. Missing output:{} of model:{}", dependencyOutputName, sourceNodeInstance->getName());
                        return StatusCode::INVALID_MISSING_INPUT;
                    }

                    if (node.kind != NodeKind::DL) {
                        break;
                    }
                    std::string& inputName = alias.second;
                    auto nodeInput = nodeInputs.find(inputName);
                    if (nodeInput == nodeInputs.end()) {
                        SPDLOG_ERROR("Validation of pipeline definition failed. Missing input:{} of node:{}", inputName, node.nodeName);
                        return StatusCode::INVALID_MISSING_INPUT;
                    }
                }
            }
        }
    }

    return StatusCode::OK;
}

Status PipelineFactory::createDefinition(const std::string& pipelineName,
    const std::vector<NodeInfo>& nodeInfos,
    const pipeline_connections_t& connections,
    ModelManager& manager) {
    if (definitionExists(pipelineName)) {
        SPDLOG_WARN("Two pipelines with the same name:{} defined in config file. Ignoring the second definition", pipelineName);
        return StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST;
    }
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>(pipelineName, nodeInfos, connections);

    Status validationResult = pipelineDefinition->validate(manager);
    if (validationResult != StatusCode::OK) {
        return validationResult;
    }

    definitions[pipelineName] = std::move(pipelineDefinition);
    // TODO: Add check if pipeline graph is acyclic, connected, no dead ends
    // https://jira.devtools.intel.com/browse/CVS-34361

    return StatusCode::OK;
}

Status PipelineFactory::create(std::unique_ptr<Pipeline>& pipeline,
    const std::string& name,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    ModelManager& manager) const {
    if (!definitionExists(name)) {
        SPDLOG_INFO("Pipeline with requested name:{} does not exist", name);
        return StatusCode::PIPELINE_DEFINITION_NAME_MISSING;
    }
    return definitions.at(name)->create(pipeline, request, response, manager);
}
}  // namespace ovms
