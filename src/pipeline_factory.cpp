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

Status PipelineFactory::createDefinition(const std::string& pipelineName,
    const std::vector<NodeInfo>& nodeInfos,
    const pipeline_connections_t& connections) {
    if (definitionExists(pipelineName)) {
        return StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST;
    }
    definitions[pipelineName] = std::make_unique<PipelineDefinition>(pipelineName, nodeInfos, connections);

    // TODO: Call PipelineDefinition::validate method to check for one entry, one exit, acyclic, connected, no dead ends
    // https://jira.devtools.intel.com/browse/CVS-34360

    return StatusCode::OK;
}

Status PipelineFactory::create(std::unique_ptr<Pipeline>& pipeline,
    const std::string& name,
    tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    ModelManager& manager) const {
    if (!definitionExists(name)) {
        SPDLOG_INFO("Pipeline with requested name:{} does not exist", name);
        return StatusCode::PIPELINE_DEFINITION_NAME_MISSING;
    }
    return definitions.at(name)->create(pipeline, request, response, manager);
}
}  // namespace ovms
