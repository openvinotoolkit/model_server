//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "pipelinedefinition.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "src/logging.hpp"
#include "src/model_instance_provider.hpp"
#include "src/modelinstanceunloadguard.hpp"
#include "src/servable_definition_unload_guard.hpp"
#include "custom_node.hpp"
#include "dl_node.hpp"
#include "nodestreamidguard.hpp"
#include "pipeline.hpp"

// Including file must provide entry_node_impl.hpp and exit_node_impl.hpp
// with the appropriate frontend-specific deserialization/serialization headers
// before including this file.

namespace ovms {
template <typename RequestType, typename ResponseType>
Status PipelineDefinition::create(std::unique_ptr<Pipeline>& pipeline,
    const RequestType* request,
    ResponseType* response,
    ModelInstanceProvider& modelInstanceProvider) {
    std::unique_ptr<ServableDefinitionUnloadGuard> unloadGuard;
    Status status = waitForLoaded(unloadGuard);
    if (!status.ok()) {
        return status;
    }

    std::unordered_map<std::string, std::unique_ptr<Node>> nodes;
    EntryNode<RequestType>* entry = nullptr;
    ExitNode<ResponseType>* exit = nullptr;

    for (const auto& info : nodeInfos) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Creating pipeline: {}. Adding nodeName: {}, modelName: {}",
            getName(), info.nodeName, info.modelName);
        switch (info.kind) {
        case NodeKind::ENTRY: {
            auto node = std::make_unique<EntryNode<RequestType>>(request, getInputsInfo(), info.demultiplyCount);
            entry = node.get();
            nodes.emplace(info.nodeName, std::move(node));
            break;
        }
        case NodeKind::DL:
            nodes.emplace(info.nodeName, std::make_unique<DLNode>(
                                             info.nodeName,
                                             info.modelName,
                                             info.modelVersion,
                                             modelInstanceProvider,
                                             info.outputNameAliases,
                                             info.demultiplyCount,
                                             info.gatherFromNode));
            break;
        case NodeKind::CUSTOM:
            nodes.emplace(info.nodeName, std::make_unique<CustomNode>(
                                             info.nodeName,
                                             info.library,
                                             info.parameters,
                                             info.outputNameAliases,
                                             info.demultiplyCount,
                                             info.gatherFromNode,
                                             nodeResources.at(info.nodeName)));
            break;
        case NodeKind::EXIT: {
            auto node = std::make_unique<ExitNode<ResponseType>>(response, getOutputsInfo(), info.gatherFromNode, useSharedOutputContentFn(request), getName());
            exit = node.get();
            nodes.emplace(info.nodeName, std::move(node));
            break;
        }
        default:
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Requested pipeline: {} contains unknown node kind", getName());
            throw std::invalid_argument("unknown node kind");
        }
    }
    for (const auto& kv : connections) {
        const auto& dependantNode = nodes.at(kv.first);
        for (const auto& pair : kv.second) {
            const auto& dependencyNode = nodes.at(pair.first);
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Connecting pipeline: {}, from: {}, to: {}", getName(), dependencyNode->getName(), dependantNode->getName());
            Pipeline::connect(*dependencyNode, *dependantNode, pair.second);
        }
    }
#pragma warning(push)
#pragma warning(disable : 6011)
    pipeline = std::make_unique<Pipeline>(*entry, *exit, *this->reporter, getName());
#pragma warning(pop)
    for (auto& kv : nodes) {
        pipeline->push(std::move(kv.second));
    }
    return status;
}
}  // namespace ovms
