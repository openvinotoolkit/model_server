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
#include "pipeline_config_parser.hpp"

#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/logging.hpp"
#include "src/status.hpp"
#include "custom_node_library_manager.hpp"
#include "entry_node.hpp"
#include "exit_node.hpp"
#include "nodeinfo.hpp"
#include "pipeline_factory.hpp"

namespace ovms {

Status toNodeKind(const std::string& str, NodeKind& nodeKind) {
    if (str == DL_NODE_CONFIG_TYPE) {
        nodeKind = NodeKind::DL;
        return StatusCode::OK;
    }
    if (str == CUSTOM_NODE_CONFIG_TYPE) {
        nodeKind = NodeKind::CUSTOM;
        return StatusCode::OK;
    }
    SPDLOG_LOGGER_ERROR(modelmanager_logger, "Unsupported node type: {}", str);
    return StatusCode::PIPELINE_NODE_WRONG_KIND_CONFIGURATION;
}

static void processNodeInputs(const std::string nodeName, const rapidjson::Value::ConstMemberIterator& itro, pipeline_connections_t& connections) {
    for (const auto& nodeInput : itro->value.GetArray()) {
        for (const auto& objectNameValue : nodeInput.GetObject()) {
            const std::string inputName = objectNameValue.name.GetString();
            const std::string sourceNodeName = objectNameValue.value.GetObject()["node_name"].GetString();
            const std::string sourceOutputName = objectNameValue.value.GetObject()["data_item"].GetString();
            SPDLOG_DEBUG("Creating node dependencies mapping request. Node: {} input: {} <- SourceNode: {} output: {}",
                nodeName, inputName, sourceNodeName, sourceOutputName);
            if (connections.find(nodeName) == connections.end()) {
                connections[nodeName] = {
                    {sourceNodeName,
                        {{sourceOutputName, inputName}}}};
            } else {
                if (connections[nodeName].find(sourceNodeName) == connections[nodeName].end()) {
                    connections[nodeName].insert({sourceNodeName,
                        {{sourceOutputName, inputName}}});
                } else {
                    connections[nodeName][sourceNodeName].push_back({sourceOutputName, inputName});
                }
            }
        }
    }
}

static void processPipelineInputs(const rapidjson::Value::ConstMemberIterator& pipelineInputsPtr, const std::string& nodeName, std::unordered_map<std::string, std::string>& nodeOutputNameAlias, const std::string& pipelineName) {
    for (const auto& pipelineInput : pipelineInputsPtr->value.GetArray()) {
        const std::string pipelineInputName = pipelineInput.GetString();
        SPDLOG_DEBUG("Mapping node:{} output:{}, under alias:{}",
            nodeName, pipelineInputName, pipelineInputName);
        auto result = nodeOutputNameAlias.insert({pipelineInputName, pipelineInputName});
        if (!result.second) {
            SPDLOG_ERROR("Pipeline {} has duplicated input declaration", pipelineName);
        }
    }
}

static void processNodeOutputs(const rapidjson::Value::ConstMemberIterator& nodeOutputsItr, const std::string& nodeName, const std::string& modelName, std::unordered_map<std::string, std::string>& nodeOutputNameAlias) {
    for (const auto& nodeOutput : nodeOutputsItr->value.GetArray()) {
        const std::string modelOutputName = nodeOutput.GetObject()["data_item"].GetString();
        const std::string nodeOutputName = nodeOutput.GetObject()["alias"].GetString();
        SPDLOG_DEBUG("Mapping node: {} model_name: {} output: {}, under alias: {}",
            nodeName, modelName, modelOutputName, nodeOutputName);
        nodeOutputNameAlias[nodeOutputName] = modelOutputName;
    }
}

static void processDLNodeConfig(const rapidjson::Value& nodeConfig, DLNodeInfo& info) {
    info.modelName = nodeConfig["model_name"].GetString();
    if (nodeConfig.HasMember("version")) {
        info.modelVersion = nodeConfig["version"].GetUint64();
    }
}

static Status processCustomNodeConfig(const rapidjson::Value& nodeConfig, CustomNodeInfo& info, const std::string& pipelineName, const CustomNodeLibraryManager& libraryManager) {
    std::string libraryName = nodeConfig["library_name"].GetString();
    auto status = libraryManager.getLibrary(libraryName, info.library);
    if (!status.ok()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Pipeline: {} refers to non existing custom node library: {}", pipelineName, libraryName);
    }
    if (nodeConfig.HasMember("params")) {
        for (const auto& param : nodeConfig["params"].GetObject()) {
            info.parameters.emplace(param.name.GetString(), param.value.GetString());
        }
    }
    return StatusCode::OK;
}

static Status processPipelineConfig(
    const rapidjson::Value& pipelineConfig,
    std::set<std::string>& pipelinesInConfigFile,
    PipelineFactory& factory,
    ModelInstanceProvider& modelInstanceProvider,
    ServableNameChecker& nameChecker,
    DagResourceManager& resourceMgr,
    const CustomNodeLibraryManager& libraryManager,
    MetricRegistry* metricRegistry,
    const MetricConfig* metricConfig) {
    const std::string pipelineName = pipelineConfig["name"].GetString();
    if (pipelinesInConfigFile.find(pipelineName) != pipelinesInConfigFile.end()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Duplicated pipeline names: {} defined in config file. Only first definition will be loaded.", pipelineName);
        return StatusCode::OK;
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Reading pipeline: {} configuration", pipelineName);
    std::set<std::string> demultiplexerNodes;
    std::set<std::string> gatheredDemultiplexerNodes;
    std::optional<int32_t> demultiplyCountEntry = std::nullopt;
    auto demultiplyCountEntryIt = pipelineConfig.FindMember("demultiply_count");
    if (demultiplyCountEntryIt != pipelineConfig.MemberEnd()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Pipeline: {} does have demultiply at entry node", pipelineName);
        int32_t parsedDemultiplyCount = pipelineConfig["demultiply_count"].GetInt();
        if (parsedDemultiplyCount == 0) {
            parsedDemultiplyCount = -1;
            SPDLOG_LOGGER_WARN(modelmanager_logger, "demultiply_count 0 will be deprecated. For dynamic count use -1.");
        }
        demultiplyCountEntry = parsedDemultiplyCount;
        demultiplexerNodes.insert(ENTRY_NODE_NAME);
    } else {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Pipeline: {} does not have demultiply at entry node", pipelineName);
    }

    std::vector<NodeInfo> info;
    NodeInfo entryInfo{NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {}, demultiplyCountEntry};
    info.emplace_back(std::move(entryInfo));
    processPipelineInputs(pipelineConfig.FindMember("inputs"), ENTRY_NODE_NAME, info[0].outputNameAliases, pipelineName);
    pipeline_connections_t connections;

    auto nodesItr = pipelineConfig.FindMember("nodes");
    for (const auto& nodeConfig : nodesItr->value.GetArray()) {
        std::string nodeName;
        nodeName = nodeConfig["name"].GetString();

        const std::string nodeKindStr = nodeConfig["type"].GetString();
        NodeKind nodeKind;
        auto status = toNodeKind(nodeKindStr, nodeKind);
        if (!status.ok()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Parsing node kind failed: {} for pipeline: {}", nodeKindStr, pipelineName);
            return status;
        }

        DLNodeInfo dlNodeInfo;
        CustomNodeInfo customNodeInfo;
        if (nodeKind == NodeKind::DL) {
            processDLNodeConfig(nodeConfig, dlNodeInfo);
        } else if (nodeKind == NodeKind::CUSTOM) {
            status = processCustomNodeConfig(nodeConfig, customNodeInfo, pipelineName, libraryManager);
            if (!status.ok()) {
                return status;
            }
        } else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Pipeline {} contains unknown node kind", pipelineName);
            throw std::invalid_argument("unknown node kind");
        }

        auto nodeOutputsItr = nodeConfig.FindMember("outputs");
        if (nodeOutputsItr == nodeConfig.MemberEnd() || !nodeOutputsItr->value.IsArray()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Pipeline: {} does not have valid outputs configuration", pipelineName);
            return status;
        }
        std::unordered_map<std::string, std::string> nodeOutputNameAlias;  // key:alias, value realName
        processNodeOutputs(nodeOutputsItr, nodeName, dlNodeInfo.modelName, nodeOutputNameAlias);
        std::optional<int32_t> demultiplyCount;
        if (nodeConfig.HasMember("demultiply_count")) {
            int32_t parsedDemultiplyCount = nodeConfig["demultiply_count"].GetInt();
            if (parsedDemultiplyCount == 0) {
                parsedDemultiplyCount = -1;
                SPDLOG_LOGGER_WARN(modelmanager_logger, "demultiply_count 0 will be deprecated. For dynamic count use -1.");
            }
            demultiplyCount = parsedDemultiplyCount;
            demultiplexerNodes.insert(nodeName);
        }
        std::set<std::string> gatherFromNode;
        if (nodeConfig.HasMember("gather_from_node")) {
            std::string nodeToGatherFrom = nodeConfig["gather_from_node"].GetString();
            gatherFromNode.insert(nodeToGatherFrom);
            gatheredDemultiplexerNodes.insert(nodeToGatherFrom);
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Creating node: {} type: {} model_name: {} modelVersion: {}",
            nodeName, nodeKindStr, dlNodeInfo.modelName, dlNodeInfo.modelVersion.value_or(0));
        info.emplace_back(
            nodeKind,
            nodeName,
            dlNodeInfo.modelName,
            dlNodeInfo.modelVersion,
            nodeOutputNameAlias,
            demultiplyCount,
            gatherFromNode,
            customNodeInfo.library,
            customNodeInfo.parameters);
        auto nodeInputItr = nodeConfig.FindMember("inputs");
        processNodeInputs(nodeName, nodeInputItr, connections);
    }
    const auto iteratorOutputs = pipelineConfig.FindMember("outputs");
    // pipeline outputs are node exit inputs
    processNodeInputs(EXIT_NODE_NAME, iteratorOutputs, connections);
    std::set<std::string> nonGatheredDemultiplexerNodes;
    std::set_difference(demultiplexerNodes.begin(), demultiplexerNodes.end(),
        gatheredDemultiplexerNodes.begin(), gatheredDemultiplexerNodes.end(),
        std::inserter(nonGatheredDemultiplexerNodes, nonGatheredDemultiplexerNodes.begin()));
    info.emplace_back(std::move(NodeInfo(NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, nonGatheredDemultiplexerNodes)));
    if (!factory.definitionExists(pipelineName)) {
        SPDLOG_DEBUG("Pipeline:{} was not loaded so far. Triggering load", pipelineName);
        auto status = factory.createDefinition(pipelineName, info, connections, modelInstanceProvider, nameChecker, resourceMgr, metricRegistry, metricConfig);
        pipelinesInConfigFile.insert(pipelineName);
        return status;
    }
    SPDLOG_DEBUG("Pipeline:{} is already loaded. Triggering reload", pipelineName);
    auto status = factory.reloadDefinition(pipelineName,
        std::move(info),
        std::move(connections),
        modelInstanceProvider, nameChecker, resourceMgr);
    pipelinesInConfigFile.insert(pipelineName);
    return status;
}

Status loadPipelinesConfig(
    rapidjson::Document& configJson,
    PipelineFactory& factory,
    ModelInstanceProvider& modelInstanceProvider,
    ServableNameChecker& nameChecker,
    DagResourceManager& resourceMgr,
    const CustomNodeLibraryManager& customNodeLibraryManager,
    MetricRegistry* metricRegistry,
    const MetricConfig* metricConfig) {
    const auto itrp = configJson.FindMember("pipeline_config_list");
    if (itrp == configJson.MemberEnd() || !itrp->value.IsArray()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Configuration file doesn't have pipelines property.");
        factory.retireOtherThan({}, modelInstanceProvider);
        return StatusCode::OK;
    }
    std::set<std::string> pipelinesInConfigFile;
    Status firstErrorStatus = StatusCode::OK;
    for (const auto& pipelineConfig : itrp->value.GetArray()) {
        auto status = processPipelineConfig(pipelineConfig, pipelinesInConfigFile, factory,
            modelInstanceProvider, nameChecker, resourceMgr, customNodeLibraryManager,
            metricRegistry, metricConfig);
        if (status != StatusCode::OK) {
            if (firstErrorStatus.ok()) {
                firstErrorStatus = status;
            }
        }
    }
    factory.retireOtherThan(std::move(pipelinesInConfigFile), modelInstanceProvider);
    return firstErrorStatus;
}

}  // namespace ovms
