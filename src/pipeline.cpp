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
#include "pipeline.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <utility>

#include "logging.hpp"
#include "threadsafequeue.hpp"

namespace ovms {

void printNodeConnections(const std::string& nodeName, const std::string& sourceNode, const InputPairs& pairs) {
    if (spdlog::default_logger()->level() > spdlog::level::debug) {
        return;
    }
    std::stringstream ss;
    ss << "Links from:" << sourceNode << " to:" << nodeName << ":\n";
    for (auto& pair : pairs) {
        ss << "\t" << nodeName << "[" << pair.second << "]=" << sourceNode << "[" << pair.first << "]\n";
    }
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, ss.str());
}

std::map<const std::string, bool> Pipeline::prepareStatusMap() const {
    std::map<const std::string, bool> nameFlagMap;
    for (const auto& node : nodes) {
        nameFlagMap.emplace(std::make_pair(node->getName(), false));
    }
    return std::move(nameFlagMap);
}

void setFailIfNotFailEarlier(ovms::Status& earlierStatusCode, ovms::Status& newFailStatus) {
    if (earlierStatusCode.ok()) {
        earlierStatusCode = newFailStatus;
    }
}

#define IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE \
    if (!firstErrorStatus.ok()) {                                                       \
        if (finishedExecute == startedExecute) {                                        \
            break;                                                                      \
        } else {                                                                        \
            continue;                                                                   \
        }                                                                               \
    }

#define CHECK_AND_LOG_ERROR(NODE)                                                                  \
    if (!status.ok()) {                                                                            \
        setFailIfNotFailEarlier(firstErrorStatus, status);                                         \
        SPDLOG_LOGGER_WARN(dag_executor_logger, "Executing pipeline: {} node: {} failed with: {}", \
            getName(), NODE.getName(), status.string());                                           \
    }

Status Pipeline::execute() {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Started execution of pipeline: {}", getName());
    ThreadSafeQueue<std::reference_wrapper<Node>> finishedNodeQueue;
    ovms::Status firstErrorStatus{ovms::StatusCode::OK};
    auto startedExecute{prepareStatusMap()};
    auto finishedExecute{prepareStatusMap()};
    startedExecute.at(entry.getName()) = true;
    ovms::Status status = entry.execute(finishedNodeQueue);  // first node will triger first message
    if (!status.ok()) {
        SPDLOG_LOGGER_WARN(dag_executor_logger, "Executing pipeline: {} node: {} failed with: {}",
            getName(), entry.getName(), status.string());
        return status;
    }
    std::vector<std::reference_wrapper<Node>> nodesWaitingForIdleInferenceStreamId;  // consider replacing with std::vector
    // even though we can remove with random sequence it is probable that we will remove those in sequence
    const uint WAIT_FOR_FINISHED_NODE_TIMEOUT_MICROSECONDS = 5000;
    const uint WAIT_FOR_DEFERRED_NODE_DISARM_TIMEOUT_MICROSECONDS = 500;
    // process finished nodes and if no one is finished check if any node with deferred execution
    // has necessary resources already
    while (true) {
        spdlog::trace("Pipeline: {} waiting for message that node finished.", getName());
        auto optionallyFinishedNode = finishedNodeQueue.tryPull(WAIT_FOR_FINISHED_NODE_TIMEOUT_MICROSECONDS);
        if (optionallyFinishedNode) {
            Node& finishedNode = optionallyFinishedNode.value().get();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Pipeline: {} got message that node: {} finished.", getName(), finishedNode.getName());
            finishedExecute.at(finishedNode.getName()) = true;
            if (!firstErrorStatus.ok()) {
                finishedNode.release();
            }
            IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
            BlobMap finishedNodeOutputBlobMap;
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Fetching results of pipeline: {} node: {}", getName(), finishedNode.getName());
            status = finishedNode.fetchResults(finishedNodeOutputBlobMap);
            CHECK_AND_LOG_ERROR(finishedNode)
            IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
            if (std::all_of(finishedExecute.begin(), finishedExecute.end(), [](auto pair) { return pair.second; })) {
                break;
            }
            auto& nextNodesFromFinished = finishedNode.getNextNodes();
            for (auto& nextNode : nextNodesFromFinished) {
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "setting pipeline: {} node: {} outputs as inputs for node: {}",
                    getName(), finishedNode.getName(), nextNode.get().getName());
                status = nextNode.get().setInputs(finishedNode, finishedNodeOutputBlobMap);
                CHECK_AND_LOG_ERROR(nextNode.get())
                if (!firstErrorStatus.ok()) {
                    break;
                }
            }
            finishedNodeOutputBlobMap.clear();
            for (auto& nextNode : nextNodesFromFinished) {
                if (nextNode.get().isReady()) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Started execution of pipeline: {} node: {}", getName(), nextNode.get().getName());
                    startedExecute.at(nextNode.get().getName()) = true;
                    status = nextNode.get().execute(finishedNodeQueue);
                    if (status == StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET) {
                        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} not ready for execution yet", nextNode.get().getName());
                        nodesWaitingForIdleInferenceStreamId.push_back(nextNode.get());
                        status = StatusCode::OK;
                    }
                    CHECK_AND_LOG_ERROR(nextNode.get())
                    if (!firstErrorStatus.ok()) {
                        break;
                    }
                }
            }
        } else {
            // If error occurred earlier, disarm stream id guards of all deferred nodes and exit
            if (!firstErrorStatus.ok()) {
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will try to disarm all stream id guards of all {} deferred nodes due to previous error in pipeline", nodesWaitingForIdleInferenceStreamId.size());
                // Check if there are deferred nodes in queue to free
                if (nodesWaitingForIdleInferenceStreamId.size() > 0) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Trying to disarm {} remaining deferred nodes...", nodesWaitingForIdleInferenceStreamId.size());
                    for (auto it = nodesWaitingForIdleInferenceStreamId.begin(); it != nodesWaitingForIdleInferenceStreamId.end();) {
                        auto& node = (*it).get();
                        if (node.tryDisarmStreamIdGuard(WAIT_FOR_DEFERRED_NODE_DISARM_TIMEOUT_MICROSECONDS)) {
                            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Stream id guard disarm of node {} has succeeded", node.getName());
                            finishedExecute.at(node.getName()) = true;
                            it = nodesWaitingForIdleInferenceStreamId.erase(it);
                        } else {
                            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Cannot disarm stream id guard of node {} yet, will try again later", node.getName());
                            it++;
                        }
                    }
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Disarming iteration completed, remaining deferred nodes count: {}", nodesWaitingForIdleInferenceStreamId.size());
                }
                // Check for deferred node queue size again to indicate if all nodes got freed
                if (nodesWaitingForIdleInferenceStreamId.size() > 0) {
                    continue;
                } else {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Disarming all stream id guards of deferred nodes completed, pipeline will shut down");
                    IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
                }
            }

            // else scope could be executed always however it seems most reasonable at the time to
            // free blocked inferRequests from exeuction first rather than free models for reloading
            for (auto it = nodesWaitingForIdleInferenceStreamId.begin(); it != nodesWaitingForIdleInferenceStreamId.end();) {
                auto& node = (*it).get();
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Trying to trigger node: {} execution", node.getName());
                status = node.execute(finishedNodeQueue);
                if (status.ok()) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} ready yet:", node.getName());
                    it = nodesWaitingForIdleInferenceStreamId.erase(it);
                    continue;
                }
                it++;
                if (status == StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} not ready for execution yet", node.getName());
                    status = StatusCode::OK;
                } else {
                    CHECK_AND_LOG_ERROR(node)
                }
            }
        }
    }
    return firstErrorStatus;
}
}  // namespace ovms
