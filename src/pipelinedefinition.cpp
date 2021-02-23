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

#include <chrono>
#include <set>
#include <thread>

#include "custom_node.hpp"
#include "dl_node.hpp"
#include "entry_node.hpp"
#include "exit_node.hpp"
#include "logging.hpp"
#include "modelmanager.hpp"
#include "node_library_utils.hpp"
#include "pipeline.hpp"
#include "pipelinedefinitionunloadguard.hpp"
#include "prediction_service_utils.hpp"

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

Status PipelineDefinition::validate(ModelManager& manager) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Started validation of pipeline: {}", getName());
    ValidationResultNotifier notifier(status, loadedNotify);
    auto& models = manager.getModels();
    if (std::find_if(models.begin(), models.end(), [this](auto pair) { return this->pipelineName == pair.first; }) != models.end()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Pipeline name: {} is already occupied by model.", pipelineName);
        return StatusCode::PIPELINE_NAME_OCCUPIED;
    }

    Status validationResult = validateNodes(manager);
    if (!validationResult.ok()) {
        return validationResult;
    }

    validationResult = validateForCycles();
    if (!validationResult.ok()) {
        return validationResult;
    }
    notifier.passed = true;
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Finished validation of pipeline: {}", getName());
    return validationResult;
}

Status PipelineDefinition::reload(ModelManager& manager, const std::vector<NodeInfo>&& nodeInfos, const pipeline_connections_t&& connections) {
    // block creating new unloadGuards
    this->status.handle(ReloadEvent());
    resetSubscriptions(manager);
    while (requestsHandlesCounter > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    this->nodeInfos = std::move(nodeInfos);
    this->connections = std::move(connections);
    makeSubscriptions(manager);

    return validate(manager);
}

void PipelineDefinition::retire(ModelManager& manager) {
    resetSubscriptions(manager);
    this->status.handle(RetireEvent());
    while (requestsHandlesCounter > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    this->nodeInfos.clear();
    this->connections.clear();
}

Status PipelineDefinition::waitForLoaded(std::unique_ptr<PipelineDefinitionUnloadGuard>& unloadGuard, const uint waitForLoadedTimeoutMicroseconds) {
    unloadGuard = std::make_unique<PipelineDefinitionUnloadGuard>(*this);

    const uint waitLoadedTimestepMicroseconds = 100;
    const uint waitCheckpoints = waitForLoadedTimeoutMicroseconds / waitLoadedTimestepMicroseconds;
    uint waitCheckpointsCounter = waitCheckpoints;
    std::mutex cvMtx;
    std::unique_lock<std::mutex> cvLock(cvMtx);
    while (waitCheckpointsCounter-- != 0) {
        if (status.isAvailable()) {
            SPDLOG_DEBUG("Successfully waited for pipeline definition: {}", getName());
            return StatusCode::OK;
        }
        unloadGuard.reset();
        if (!status.canEndLoaded()) {
            if (status.getStateCode() != PipelineDefinitionStateCode::RETIRED) {
                SPDLOG_DEBUG("Waiting for pipeline definition: {} ended due to timeout.", getName());
                return StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET;
            } else {
                SPDLOG_DEBUG("Waiting for pipeline definition: {} ended since it failed to load.", getName());
                return StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE;
            }
        }
        SPDLOG_DEBUG("Waiting for available state for pipeline: {}, with timestep: {}us timeout: {}us check count: {}",
            getName(), waitLoadedTimestepMicroseconds, waitForLoadedTimeoutMicroseconds, waitCheckpointsCounter);
        loadedNotify.wait_for(cvLock,
            std::chrono::microseconds(waitLoadedTimestepMicroseconds),
            [this]() {
                return this->status.isAvailable() ||
                       !this->status.canEndLoaded();
            });
        unloadGuard = std::make_unique<PipelineDefinitionUnloadGuard>(*this);
    }
    if (!status.isAvailable()) {
        if (status.getStateCode() != PipelineDefinitionStateCode::RETIRED) {
            SPDLOG_DEBUG("Waiting for pipeline definition: {} ended due to timeout.", getName());
            return StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET;
        } else {
            SPDLOG_DEBUG("Waiting for pipeline definition: {} ended since it failed to load.", getName());
            return StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_DEBUG("Succesfully waited for pipeline definition: {}", getName());
    return StatusCode::OK;
}

Status PipelineDefinition::create(std::unique_ptr<Pipeline>& pipeline,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    ModelManager& manager) {
    std::unique_ptr<PipelineDefinitionUnloadGuard> unloadGuard;
    Status status = waitForLoaded(unloadGuard);
    if (!status.ok()) {
        return status;
    }

    std::unordered_map<std::string, std::unique_ptr<Node>> nodes;
    EntryNode* entry = nullptr;
    ExitNode* exit = nullptr;

    for (const auto& info : nodeInfos) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Creating pipeline: {}. Adding nodeName: {}, modelName: {}",
            getName(), info.nodeName, info.modelName);
        switch (info.kind) {
        case NodeKind::ENTRY: {
            auto node = std::make_unique<EntryNode>(request);
            entry = node.get();
            nodes.emplace(info.nodeName, std::move(node));
            break;
        }
        case NodeKind::DL:
            nodes.emplace(info.nodeName, std::make_unique<DLNode>(
                                             info.nodeName,
                                             info.modelName,
                                             info.modelVersion,
                                             manager,
                                             info.outputNameAliases,
                                             info.demultiplyCount.value_or(0),
                                             info.gatherFromNode));
            break;
        case NodeKind::CUSTOM:
            nodes.emplace(info.nodeName, std::make_unique<CustomNode>(
                                             info.nodeName,
                                             info.library,
                                             info.parameters,
                                             info.outputNameAliases,
                                             info.demultiplyCount.value_or(0),
                                             info.gatherFromNode));
            break;
        case NodeKind::EXIT: {
            auto node = std::make_unique<ExitNode>(response, info.gatherFromNode);
            exit = node.get();
            nodes.emplace(info.nodeName, std::move(node));
            break;
        }
        default:
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Requested pipeline {} contains unknown node kind", getName());
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
    pipeline = std::make_unique<Pipeline>(*entry, *exit, pipelineName);
    for (auto& kv : nodes) {
        pipeline->push(std::move(kv.second));
    }
    return status;
}

void PipelineDefinition::resetSubscriptions(ModelManager& manager) {
    for (auto& [modelName, modelVersion] : subscriptions) {
        if (modelVersion) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Unsubscribing pipeline: {} from model: {}, version: {}",
                getName(), modelName, modelVersion);
            manager.findModelByName(modelName)->getModelInstanceByVersion(modelVersion)->unsubscribe(*this);
        } else {  // using default version
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Unsubscribing pipeline: {} from model: {}",
                getName(), modelName);
            manager.findModelByName(modelName)->unsubscribe(*this);
        }
    }
    subscriptions.clear();
}

static std::string createSubscriptionErrorMessage(const std::string& pipelineName, const NodeInfo& nodeInfo) {
    std::stringstream ss;
    ss << "Pipeline: " << pipelineName << " Failed to make subscription to model: " << nodeInfo.modelName;
    if (nodeInfo.modelVersion) {
        ss << " version: " << nodeInfo.modelVersion.value();
    }
    ss << " because it was missing";
    return ss.str();
}

void PipelineDefinition::makeSubscriptions(ModelManager& manager) {
    for (auto& node : nodeInfos) {
        if (node.kind == NodeKind::DL) {
            if (subscriptions.find({node.modelName, node.modelVersion.value_or(0)}) != subscriptions.end()) {
                continue;
            }
            auto model = manager.findModelByName(node.modelName);
            if (nullptr == model) {
                SPDLOG_LOGGER_WARN(modelmanager_logger, createSubscriptionErrorMessage(getName(), node));
                continue;
            }
            if (node.modelVersion) {
                auto modelInstance = model->getModelInstanceByVersion(node.modelVersion.value());
                if (nullptr == modelInstance) {
                    SPDLOG_LOGGER_WARN(modelmanager_logger, createSubscriptionErrorMessage(getName(), node));
                    continue;
                }
                modelInstance->subscribe(*this);
            } else {
                model->subscribe(*this);
            }
            subscriptions.insert({node.modelName, node.modelVersion.value_or(0)});
        }
    }
}

class NodeValidator {
    const std::string& pipelineName;
    ModelManager& manager;
    const NodeInfo& dependantNodeInfo;
    const pipeline_connections_t& connections;
    const std::vector<NodeInfo>& nodeInfos;
    const bool isMultiBatchAllowed;

    std::unique_ptr<ModelInstanceUnloadGuard> dependantModelUnloadGuard;
    std::shared_ptr<ModelInstance> dependantModelInstance;
    std::set<std::string> remainingUnconnectedDependantInputs;

    tensor_map_t inputsInfo, outputsInfo;
    tensor_map_t dependencyInputsInfo, dependencyOutputsInfo;

public:
    NodeValidator(
        const std::string& pipelineName,
        ModelManager& manager,
        const NodeInfo& dependantNodeInfo,
        const pipeline_connections_t& connections,
        const std::vector<NodeInfo>& nodeInfos,
        const bool isMultiBatchAllowed = true) :
        pipelineName(pipelineName),
        manager(manager),
        dependantNodeInfo(dependantNodeInfo),
        connections(connections),
        nodeInfos(nodeInfos),
        isMultiBatchAllowed(isMultiBatchAllowed) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Validation of pipeline: {}; node name: {}; node kind: {}",
            pipelineName,
            dependantNodeInfo.nodeName,
            dependantNodeInfo.kind);
    }

    Status fetchUnderlyingModelInstance() {
        if (!manager.getModelInstance(
                        dependantNodeInfo.modelName,
                        dependantNodeInfo.modelVersion.value_or(0),
                        dependantModelInstance,
                        dependantModelUnloadGuard)
                 .ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Missing model: {}; version: {}",
                pipelineName,
                dependantNodeInfo.modelName,
                dependantNodeInfo.modelVersion.value_or(0));
            return StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL;
        }
        return StatusCode::OK;
    }

    Status getDependencyNodeInfo(const std::string& dependencyNodeName, std::vector<NodeInfo>::const_iterator& dependencyNodeInfo) {
        // Find dependency node info object.
        dependencyNodeInfo = std::find_if(
            std::begin(this->nodeInfos),
            std::end(this->nodeInfos),
            [dependencyNodeName](const NodeInfo& nodeInfo) { return nodeInfo.nodeName == dependencyNodeName; });
        if (dependencyNodeInfo == std::end(this->nodeInfos)) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Node (name:{}) is connected to missing dependency node (name:{})",
                pipelineName,
                dependantNodeInfo.nodeName,
                dependencyNodeName);
            return StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_NODE;
        }

        if (dependencyNodeInfo->kind == NodeKind::EXIT) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Exit node used as dependency node",
                pipelineName);
            return StatusCode::PIPELINE_EXIT_USED_AS_NODE_DEPENDENCY;
        }

        return StatusCode::OK;
    }

    Status checkForForbiddenDynamicParameters() {
        const auto& config = dependantModelInstance->getModelConfig();
        if (config.getBatchingMode() == Mode::AUTO || config.anyShapeSetToAuto()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Node name {} used model name {} with dynamic batch/shape parameter which is forbidden.",
                pipelineName,
                dependantNodeInfo.nodeName,
                dependantNodeInfo.modelName);
            return StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER;
        }
        return StatusCode::OK;
    }

    Status checkForRestrictedBatchSize() {
        if (!isMultiBatchAllowed) {
            for (auto& [inputName, tensorInfo] : this->inputsInfo) {
                if (!tensorInfo->getShape().empty() && tensorInfo->getShape()[0] >= 2) {
                    return StatusCode::PIPELINE_DEMULTIPLEXER_MULTIPLE_BATCH_SIZE;
                }
            }
            if (dependantModelInstance && dependantModelInstance->getBatchSize() >= 2) {
                return StatusCode::PIPELINE_DEMULTIPLEXER_MULTIPLE_BATCH_SIZE;
            }
        }
        return StatusCode::OK;
    }

    Status validateGatherNode(const NodeInfo& dependantNodeInfo) const {
        for (const auto& gather : dependantNodeInfo.gatherFromNode) {
            auto it = std::find_if(nodeInfos.begin(), nodeInfos.end(), [gather](const NodeInfo& nodeInfo) { return nodeInfo.nodeName == gather; });
            if (it == nodeInfos.end()) {
                return StatusCode::PIPELINE_NODE_GATHER_FROM_NOT_EXISTING_NODE;
            }
            if (!it->demultiplyCount) {
                return StatusCode::PIPELINE_NODE_GATHER_FROM_NOT_DEMULTIPLEXER;
            }
            if (it->kind == NodeKind::ENTRY) {
                return StatusCode::PIPELINE_NODE_GATHER_FROM_ENTRY_NODE;
            }
        }
        return StatusCode::OK;
    }

    Status checkConnectionMappedToExistingDataSource(const NodeInfo& dependencyNodeInfo, const std::string& dataSource) {
        // Check whether dependency node is configured to have required output.
        if (dependencyNodeInfo.outputNameAliases.count(dataSource) == 0) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Missing dependency node:{} data item:{} for dependant node:{}",
                pipelineName,
                dependencyNodeInfo.nodeName,
                dataSource,
                dependantNodeInfo.nodeName);
            return StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_DATA_SOURCE;
        }

        // If dependency node is of type DL model, make sure there is underlying model output present.
        if (dependencyNodeInfo.kind == NodeKind::DL || dependencyNodeInfo.kind == NodeKind::CUSTOM) {
            // Check whether underlying model contains required output.
            const auto& modelOutputName = dependencyNodeInfo.outputNameAliases.at(dataSource);
            if (this->dependencyOutputsInfo.count(modelOutputName) == 0) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Missing output:{} of dependency node:{}; data source {}",
                    pipelineName,
                    modelOutputName,
                    dependencyNodeInfo.nodeName,
                    dataSource);
                return StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL_OUTPUT;  // TODO: Check status for Model phrase, might need to have add custom too
            }
        }

        return StatusCode::OK;
    }

    Status influenceShapeWithDemultiplexer(shape_t& shape, const NodeInfo& demultiplicatorNodeInfo) {
        if (shape.size() < 3) {
            return StatusCode::UNKNOWN_ERROR;  // TODO: specific error? Add test?
        }
        if (shape[1] != demultiplicatorNodeInfo.demultiplyCount.value()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Demultiply count: {} of node: {} does not match tensor second dimenson value: {}",
                this->pipelineName,
                demultiplicatorNodeInfo.demultiplyCount.value(),
                demultiplicatorNodeInfo.nodeName,
                shape[1]);
            return StatusCode::PIPELINE_DEMULTIPLY_COUNT_DOES_NOT_MATCH_BLOB_SHARD_COUNT;
        }
        shape.erase(shape.begin() + 1);
        return StatusCode::OK;
    }

    Status checkConnectionMetadataCorrectness(const NodeInfo& dependencyNodeInfo, const std::string& modelInputName, const std::string& modelOutputName) {
        // If validated connection pair connects two DL model/Custom nodes,
        // check if both input/output exist and its metadata (shape, precision) matches.
        // Affect shape by demultiplexer/gather if applies.
        const auto& tensorInput = this->inputsInfo.at(modelInputName);
        const auto& tensorOutput = this->dependencyOutputsInfo.at(modelOutputName);
        shape_t tensorInputShape = tensorInput->getShape();
        shape_t tensorOutputShape = tensorOutput->getShape();
        if (dependencyNodeInfo.demultiplyCount) {
            auto result = influenceShapeWithDemultiplexer(tensorOutputShape, dependencyNodeInfo);
            if (!result.ok()) {
                return result;
            }
        }
        if (dependantNodeInfo.gatherFromNode.size() == 1) {
            std::vector<NodeInfo>::const_iterator demultiplicatorNode;
            auto result = getDependencyNodeInfo(*dependantNodeInfo.gatherFromNode.begin(), demultiplicatorNode);
            if (!result.ok()) {
                return result;
            }
            result = influenceShapeWithDemultiplexer(tensorInputShape, *demultiplicatorNode);
            if (!result.ok()) {
                return result;
            }
        } else if (dependantNodeInfo.gatherFromNode.size() > 1) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Manual gathering from multiple nodes is not supported in node name: {}",
                this->pipelineName,
                dependantNodeInfo.nodeName);
            return StatusCode::PIPELINE_MANUAL_GATHERING_FROM_MULTIPLE_NODES_NOT_SUPPORTED;
        }
        if (tensorInputShape != tensorOutputShape) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Shape mismatch between: dependant node:{}; input:{}; shape:{} vs dependency node:{}; output:{}; shape:{}",
                pipelineName,
                dependantNodeInfo.nodeName,
                modelInputName,
                TensorInfo::shapeToString(tensorInput->getShape()),
                dependencyNodeInfo.nodeName,
                modelOutputName,
                TensorInfo::shapeToString(tensorOutput->getShape()));
            return StatusCode::INVALID_SHAPE;
        }
        if (tensorInput->getPrecision() != tensorOutput->getPrecision()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Precision mismatch between: dependant node:{}; input:{}; precision:{} vs dependency node:{}; output:{}; precision:{}",
                pipelineName,
                dependantNodeInfo.nodeName,
                modelInputName,
                tensorInput->getPrecisionAsString(),
                dependencyNodeInfo.nodeName,
                modelOutputName,
                tensorOutput->getPrecisionAsString());
            return StatusCode::INVALID_PRECISION;
        }
        return StatusCode::OK;
    }

    void prepareRemainingUnconnectedDependantInputsSet() {
        // Save set of inputs which are required by underlying model/custom node of currently validated node.
        // This is later used to make sure we feed each input exactly one data source.
        std::transform(
            this->inputsInfo.begin(),
            this->inputsInfo.end(),
            std::inserter(
                remainingUnconnectedDependantInputs,
                remainingUnconnectedDependantInputs.end()),
            [](auto pair) { return pair.first; });
    }

    Status ensureAllModelInputsOfValidatedNodeHaveDataSource() {
        // Make sure all model inputs of validated node is fed with some data source.
        if (remainingUnconnectedDependantInputs.size() > 0) {
            std::stringstream ss;
            for (const auto& input : remainingUnconnectedDependantInputs) {
                ss << input << ", ";
            }
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Node:{} has inputs:({}) not connected to any source",
                pipelineName,
                dependantNodeInfo.nodeName,
                ss.str());
            return StatusCode::PIPELINE_NOT_ALL_INPUTS_CONNECTED;
        }
        return StatusCode::OK;
    }

    Status markInputAsConnected(const std::string& name) {
        // If currently validated node is of type DL model or Custom, mark its input as connected
        // by erasing from previously gathered input set.
        // If such input cannot be found in the map, it means we refer
        // to non existing model input or we already connected it to some other data source which is invalid.
        if (this->inputsInfo.count(name) == 0) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Node:{} has no input with name:{}",
                pipelineName,
                dependantNodeInfo.nodeName,
                name);
            return StatusCode::PIPELINE_CONNECTION_TO_MISSING_MODEL_INPUT;
        }
        if (remainingUnconnectedDependantInputs.erase(name) == 0) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Node:{} input name:{} is connected to more than one data source",
                pipelineName,
                dependantNodeInfo.nodeName,
                name);
            return StatusCode::PIPELINE_MODEL_INPUT_CONNECTED_TO_MULTIPLE_DATA_SOURCES;
        }
        return StatusCode::OK;
    }

    Status validateConnection(const NodeInfo& dependencyNodeInfo, const Aliases& mapping) {
        // At this point dependency node can only be either DL model node, Custom node or entry node.
        // Take care when adding new node types.
        std::unique_ptr<ModelInstanceUnloadGuard> dependencyModelUnloadGuard;
        std::shared_ptr<ModelInstance> dependencyModelInstance;
        if (dependencyNodeInfo.kind == NodeKind::DL) {
            if (!manager.getModelInstance(
                            dependencyNodeInfo.modelName,
                            dependencyNodeInfo.modelVersion.value_or(0),
                            dependencyModelInstance,
                            dependencyModelUnloadGuard)
                     .ok()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline({}) definition failed. Dependency DL model node refers to unavailable model - name:{}; version:{}",
                    pipelineName,
                    dependencyNodeInfo.modelName,
                    dependencyNodeInfo.modelVersion.value_or(0));
                return StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL;
            }
            retrieveModelNodeDependencyMetadata(dependencyModelInstance);
        }

        if (dependencyNodeInfo.kind == NodeKind::CUSTOM) {
            auto result = retrieveCustomNodeDependencyMetadata(dependencyNodeInfo);
            if (!result.ok()) {
                return result;
            }
        }

        for (const auto& [alias, realName] : mapping) {
            if (dependantNodeInfo.kind == NodeKind::DL || dependantNodeInfo.kind == NodeKind::CUSTOM) {
                auto result = markInputAsConnected(realName);
                if (!result.ok()) {
                    return result;
                }
            }

            auto result = checkConnectionMappedToExistingDataSource(dependencyNodeInfo, alias);
            if (!result.ok()) {
                return result;
            }

            if (
                (dependantNodeInfo.kind == NodeKind::DL || dependantNodeInfo.kind == NodeKind::CUSTOM) &&
                (dependencyNodeInfo.kind == NodeKind::DL || dependencyNodeInfo.kind == NodeKind::CUSTOM)) {
                result = checkConnectionMetadataCorrectness(dependencyNodeInfo, realName, dependencyNodeInfo.outputNameAliases.at(alias));
                if (!result.ok()) {
                    return result;
                }
            }
        }

        return StatusCode::OK;
    }

    Status retrieveDependantMetadata() {
        if (dependantNodeInfo.kind == NodeKind::DL) {
            this->inputsInfo = this->dependantModelInstance->getInputsInfo();
            this->outputsInfo = this->dependantModelInstance->getOutputsInfo();
            return StatusCode::OK;
        } else if (dependantNodeInfo.kind == NodeKind::CUSTOM) {
            auto result = PipelineDefinition::getCustomNodeMetadata(
                dependantNodeInfo,
                this->inputsInfo,
                dependantNodeInfo.library.getInputsInfo,
                this->pipelineName);
            if (!result.ok()) {
                return result;
            }
            result = PipelineDefinition::getCustomNodeMetadata(
                dependantNodeInfo,
                this->outputsInfo,
                dependantNodeInfo.library.getOutputsInfo,
                this->pipelineName);
            if (!result.ok()) {
                return result;
            }
        }
        return StatusCode::OK;
    }

    void retrieveModelNodeDependencyMetadata(const std::shared_ptr<ModelInstance>& dependencyModelInstance) {
        this->dependencyInputsInfo = dependencyModelInstance->getInputsInfo();
        this->dependencyOutputsInfo = dependencyModelInstance->getOutputsInfo();
    }

    Status retrieveCustomNodeDependencyMetadata(const NodeInfo& dependencyNodeInfo) {
        auto result = PipelineDefinition::getCustomNodeMetadata(
            dependencyNodeInfo,
            this->dependencyInputsInfo,
            dependencyNodeInfo.library.getInputsInfo,
            this->pipelineName);
        if (!result.ok()) {
            return result;
        }
        result = PipelineDefinition::getCustomNodeMetadata(
            dependencyNodeInfo,
            this->dependencyOutputsInfo,
            dependencyNodeInfo.library.getOutputsInfo,
            this->pipelineName);
        if (!result.ok()) {
            return result;
        }
        return StatusCode::OK;
    }

    Status validate() {
        if (dependantNodeInfo.kind == NodeKind::DL) {
            auto result = fetchUnderlyingModelInstance();
            if (!result.ok()) {
                return result;
            }

            result = retrieveDependantMetadata();
            if (!result.ok()) {
                return result;
            }

            result = checkForForbiddenDynamicParameters();
            if (!result.ok()) {
                return result;
            }

            result = checkForRestrictedBatchSize();
            if (!result.ok()) {
                return result;
            }

            prepareRemainingUnconnectedDependantInputsSet();
        }

        if (dependantNodeInfo.kind == NodeKind::CUSTOM) {
            if (!dependantNodeInfo.library.isValid()) {
                return StatusCode::PIPELINE_DEFINITION_INVALID_NODE_LIBRARY;
            }

            auto result = retrieveDependantMetadata();
            if (!result.ok()) {
                return result;
            }

            result = checkForRestrictedBatchSize();
            if (!result.ok()) {
                return result;
            }

            prepareRemainingUnconnectedDependantInputsSet();
        }

        if (!dependantNodeInfo.gatherFromNode.empty()) {
            auto result = validateGatherNode(dependantNodeInfo);
            if (!result.ok()) {
                return result;
            }
        }

        if (dependantNodeInfo.kind == NodeKind::ENTRY && dependantNodeInfo.demultiplyCount) {
            return StatusCode::PIPELINE_DEMULTIPLY_ENTRY_NODE;
        }

        if (connections.count(dependantNodeInfo.nodeName) > 0) {
            for (const auto& [dependencyNodeName, mapping] : connections.at(dependantNodeInfo.nodeName)) {
                if (mapping.size() == 0) {
                    return StatusCode::UNKNOWN_ERROR;
                }

                this->dependencyInputsInfo.clear();
                this->dependencyOutputsInfo.clear();
                std::vector<NodeInfo>::const_iterator dependencyNodeInfo;
                auto result = getDependencyNodeInfo(dependencyNodeName, dependencyNodeInfo);
                if (!result.ok()) {
                    return result;
                }

                result = validateConnection(*dependencyNodeInfo, mapping);
                if (!result.ok()) {
                    return result;
                }
            }
        }

        return ensureAllModelInputsOfValidatedNodeHaveDataSource();
    }
};

Status PipelineDefinition::validateNode(ModelManager& manager, const NodeInfo& dependantNodeInfo, const bool isMultiBatchAllowed) {
    NodeValidator validator(this->pipelineName, manager, dependantNodeInfo, connections, nodeInfos, isMultiBatchAllowed);
    return validator.validate();
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
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Pipeline {} does not contain response node.", getName());
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
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Node: {} is connected to itself in pipeline: {}", nodeName, getName());
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
                    SPDLOG_LOGGER_ERROR(modelmanager_logger, "In pipeline: {}, following nodes creates cycle: {}", getName(), cycleNodes);
                    return StatusCode::PIPELINE_CYCLE_FOUND;
                }
            }
        }

        if (!unvisistedFound) {
            if (parentNodes.size() == 0) {
                anyUnvisitedLeft = false;
                if (visited.size() != nodeInfos.size()) {
                    SPDLOG_LOGGER_ERROR(modelmanager_logger, "In pipeline: {}, there are not connected nodes", getName());
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
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Validation of pipeline definition: {} nodes started.", getName());

    int entryNodeCount = std::count_if(
        this->nodeInfos.begin(),
        this->nodeInfos.end(),
        [](const NodeInfo& info) { return info.kind == NodeKind::ENTRY; });

    int exitNodeCount = std::count_if(
        this->nodeInfos.begin(),
        this->nodeInfos.end(),
        [](const NodeInfo& info) { return info.kind == NodeKind::EXIT; });

    if (entryNodeCount <= 0) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "PipelineDefinition: {} is missing request node", pipelineName);
        return StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT;
    }

    if (exitNodeCount <= 0) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "PipelineDefinition: {} is missing response node", pipelineName);
        return StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT;
    }

    if (entryNodeCount > 1) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "PipelineDefinition: {} has multiple request nodes", pipelineName);
        return StatusCode::PIPELINE_MULTIPLE_ENTRY_NODES;
    }

    if (exitNodeCount > 1) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "PipelineDefinition: {} has multiple response nodes", pipelineName);
        return StatusCode::PIPELINE_MULTIPLE_EXIT_NODES;
    }

    const bool isMultiBatchAllowed = !std::any_of(nodeInfos.begin(), nodeInfos.end(), [](const auto& node) { return node.demultiplyCount; });
    for (const auto& node : nodeInfos) {
        auto findByName = [node](const NodeInfo& nodeInfo) {
            return nodeInfo.nodeName == node.nodeName;
        };

        if (std::count_if(nodeInfos.begin(), nodeInfos.end(), findByName) > 1) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "PipelineDefinition: {} has multiple nodes with name {}", pipelineName, node.nodeName);
            return StatusCode::PIPELINE_NODE_NAME_DUPLICATE;
        }

        auto result = validateNode(manager, node, isMultiBatchAllowed);
        if (!result.ok()) {
            return result;
        }
    }
    return StatusCode::OK;
}

Status PipelineDefinition::getInputsInfo(tensor_map_t& inputsInfo, const ModelManager& manager) const {
    // Assumptions: this can only be called on available pipeline definition.
    // Add check if available when pipeline status will be implemented.

    static const auto byName = [](const std::string& name) {
        return [name](const NodeInfo& nodeInfo) {
            return nodeInfo.nodeName == name;
        };
    };
    for (const auto& [dependantNodeName, allMappings] : connections) {
        const auto& dependantNodeInfo = std::find_if(std::begin(nodeInfos), std::end(nodeInfos), byName(dependantNodeName));
        for (const auto& [dependencyNodeName, specificDependencyMapping] : allMappings) {
            const auto& dependencyNodeInfo = std::find_if(std::begin(nodeInfos), std::end(nodeInfos), byName(dependencyNodeName));
            if (dependencyNodeInfo->kind != NodeKind::ENTRY) {
                continue;
            }

            switch (dependantNodeInfo->kind) {
            case NodeKind::EXIT: {
                for (const auto& [alias, realName] : specificDependencyMapping) {
                    inputsInfo.insert({alias, TensorInfo::getUnspecifiedTensorInfo()});
                }
                break;
            }
            case NodeKind::DL: {
                auto instance = manager.findModelInstance(dependantNodeInfo->modelName, dependantNodeInfo->modelVersion.value_or(0));
                if (!instance) {
                    SPDLOG_DEBUG("Model: {} was unavailable during pipeline: {} inputs info fetching", dependantNodeInfo->modelName, this->getName());
                    return StatusCode::MODEL_MISSING;
                }
                std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;
                auto status = instance->waitForLoaded(0, unloadGuard);
                if (!status.ok()) {
                    SPDLOG_DEBUG("Model: {} was unavailable during pipeline: {} inputs info fetching", instance->getName(), this->getName());
                    return status;
                }

                for (const auto& [alias, realName] : specificDependencyMapping) {
                    inputsInfo[alias] = instance->getInputsInfo().at(realName);
                }
                break;
            }
            case NodeKind::CUSTOM: {
                if (!dependantNodeInfo->library.isValid()) {
                    return StatusCode::NODE_LIBRARY_MISSING;
                }

                tensor_map_t info;
                auto status = getCustomNodeMetadata(*dependantNodeInfo, info, dependantNodeInfo->library.getInputsInfo, this->getName());
                if (!status.ok()) {
                    return status;
                }

                for (const auto& [alias, realName] : specificDependencyMapping) {
                    inputsInfo[alias] = info.at(realName);
                }
                break;
            }
            default: {
                // Pipeline validation does not allow connections into entry node.
                SPDLOG_ERROR("Unexpected dependant node kind (name: {})", this->getName());
                return StatusCode::UNKNOWN_ERROR;
            }
            }
        }
    }
    return StatusCode::OK;
}

std::shared_ptr<TensorInfo> applyGatherShapeForTensor(const std::shared_ptr<TensorInfo>& tensorInfo, const shape_t& gatherShape) {
    if (gatherShape.size() == 0) {
        return tensorInfo;
    }
    shape_t newShape = tensorInfo->getShape();
    newShape.insert(newShape.begin() + 1, gatherShape.begin(), gatherShape.end());
    return tensorInfo->createCopyWithNewShape(newShape);
}

Status PipelineDefinition::populateOutputsInfoWithDLModelOutputs(const NodeInfo& dependencyNodeInfo, const ModelManager& manager, tensor_map_t& outputsInfo, const Aliases& specificDependencyMapping, const shape_t& gatherShape) const {
    auto instance = manager.findModelInstance(dependencyNodeInfo.modelName, dependencyNodeInfo.modelVersion.value_or(0));
    if (!instance) {
        SPDLOG_DEBUG("Model: {} was unavailable during pipeline: {} outputs info fetching", dependencyNodeInfo.modelName, this->getName());
        return StatusCode::MODEL_MISSING;
    }
    std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;
    auto status = instance->waitForLoaded(0, unloadGuard);
    if (!status.ok()) {
        SPDLOG_DEBUG("Model: {} was unavailable during pipeline: {} outputs info fetching", instance->getName(), this->getName());
        return status;
    }
    for (const auto& [alias, realName] : specificDependencyMapping) {
        const auto& finalName = dependencyNodeInfo.outputNameAliases.count(alias) > 0 ? dependencyNodeInfo.outputNameAliases.at(alias) : alias;
        outputsInfo[realName] = applyGatherShapeForTensor(instance->getOutputsInfo().at(finalName), gatherShape);
    }
    return StatusCode::OK;
}

Status PipelineDefinition::populateOutputsInfoWithCustomNodeOutputs(const NodeInfo& dependencyNodeInfo, const ModelManager& manager, tensor_map_t& outputsInfo, const Aliases& specificDependencyMapping, const shape_t& gatherShape) const {
    if (!dependencyNodeInfo.library.isValid()) {
        return StatusCode::NODE_LIBRARY_MISSING;
    }
    tensor_map_t info;
    auto status = getCustomNodeMetadata(dependencyNodeInfo, info, dependencyNodeInfo.library.getOutputsInfo, this->getName());
    if (!status.ok()) {
        return status;
    }
    for (const auto& [alias, realName] : specificDependencyMapping) {
        const auto& finalName = dependencyNodeInfo.outputNameAliases.count(alias) > 0 ? dependencyNodeInfo.outputNameAliases.at(alias) : alias;
        outputsInfo[realName] = applyGatherShapeForTensor(info.at(finalName), gatherShape);
    }
    return StatusCode::OK;
}

Status PipelineDefinition::getOutputsInfo(tensor_map_t& outputsInfo, const ModelManager& manager) const {
    // Assumptions: this can only be called on available pipeline definition.
    // Add check if available when pipeline status will be implemented.

    static const auto byName = [](const std::string& name) {
        return [name](const NodeInfo& nodeInfo) {
            return nodeInfo.nodeName == name;
        };
    };
    for (const auto& [dependantNodeName, allMappings] : connections) {
        const auto& dependantNodeInfo = std::find_if(std::begin(nodeInfos), std::end(nodeInfos), byName(dependantNodeName));
        if (dependantNodeInfo->kind != NodeKind::EXIT) {
            continue;
        }

        auto gatherShape = this->getNodeGatherShape(*dependantNodeInfo);

        for (const auto& [dependencyNodeName, specificDependencyMapping] : allMappings) {
            const auto& dependencyNodeInfo = std::find_if(std::begin(nodeInfos), std::end(nodeInfos), byName(dependencyNodeName));

            switch (dependencyNodeInfo->kind) {
            case NodeKind::ENTRY: {
                for (const auto& [alias, realName] : specificDependencyMapping) {
                    outputsInfo.insert({realName, TensorInfo::getUnspecifiedTensorInfo()});
                }
                break;
            }
            case NodeKind::DL: {
                auto status = populateOutputsInfoWithDLModelOutputs(
                    *dependencyNodeInfo, manager, outputsInfo, specificDependencyMapping, gatherShape);
                if (!status.ok()) {
                    return status;
                }
                break;
            }
            case NodeKind::CUSTOM: {
                auto status = populateOutputsInfoWithCustomNodeOutputs(
                    *dependencyNodeInfo, manager, outputsInfo, specificDependencyMapping, gatherShape);
                if (!status.ok()) {
                    return status;
                }
                break;
            }
            default: {
                // Pipeline validation does not allow connections from exit node.
                SPDLOG_ERROR("Unexpected dependency node kind (name: {})", this->getName());
                return StatusCode::UNKNOWN_ERROR;
            }
            }
        }
    }
    return StatusCode::OK;
}

Status PipelineDefinition::getCustomNodeMetadata(const NodeInfo& customNodeInfo, tensor_map_t& inputsInfo, metadata_fn callback, const std::string& pipelineName) {
    struct CustomNodeTensorInfo* info = nullptr;
    int infoLength = 0;
    auto paramArray = createCustomNodeParamArray(customNodeInfo.parameters);  // TODO: not create it in every call? prepare it once?
    int paramArrayLength = customNodeInfo.parameters.size();
    int result = callback(&info, &infoLength, paramArray.get(), paramArrayLength);
    if (result != 0) {
        SPDLOG_ERROR("Metadata call to custom node {} in pipeline {} returned error code: {}",
            customNodeInfo.nodeName, pipelineName, result);
        return StatusCode::NODE_LIBRARY_METADATA_FAILED;
    }
    return createTensorInfoMap(info, infoLength, inputsInfo, customNodeInfo.library.release);
}

const NodeInfo& PipelineDefinition::findNodeByName(const std::string& name) const {
    return *std::find_if(std::begin(this->nodeInfos), std::end(this->nodeInfos), [&name](const NodeInfo& nodeInfo) {
        return nodeInfo.nodeName == name;
    });
}

shape_t PipelineDefinition::getNodeGatherShape(const NodeInfo& info) const {
    if (info.gatherFromNode.size() == 0) {
        return {};
    }
    shape_t shape;
    shape.reserve(info.gatherFromNode.size());

    std::function<void(const std::string&)> search;
    search = [this, &info, &search, &shape](const std::string& nodeName) {
        if (this->connections.count(nodeName) == 0) {
            return;
        }
        if (info.gatherFromNode.count(nodeName) > 0) {
            shape.emplace_back(this->findNodeByName(nodeName).demultiplyCount.value());
        }
        for (const auto& [previousNodeName, _] : this->connections.at(nodeName)) {
            search(previousNodeName);
        }
    };

    search(info.nodeName);

    if (info.gatherFromNode.size() != shape.size()) {
        SPDLOG_ERROR("Pipeline {} node {} is misconfigured, gather shape has different number of dimensions that gather from node elements: {} vs {}",
            this->getName(), info.nodeName, shape.size(), info.gatherFromNode.size());
        throw std::invalid_argument("Gather shape has different number of dimensions that gather from node elements");
    }

    std::reverse(shape.begin(), shape.end());
    return std::move(shape);
}

}  // namespace ovms
