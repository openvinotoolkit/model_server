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
#include <set>
#include <string>
#include <utility>

#include "entry_node.hpp"
#include "exit_node.hpp"
#include "logging.hpp"
#include "node.hpp"
#include "pipelineeventqueue.hpp"

namespace ovms {
Pipeline::~Pipeline() = default;

Pipeline::Pipeline(EntryNode& entry, ExitNode& exit, const std::string& name) :
    name(name),
    entry(entry),
    exit(exit) {}

void Pipeline::push(std::unique_ptr<Node> node) {
    nodes.emplace_back(std::move(node));
}
void Pipeline::connect(Node& from, Node& to, const Aliases& blobNamesMapping) {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Connecting from: {}, to: {}", from.getName(), to.getName());
    printNodeConnections(to.getName(), from.getName(), blobNamesMapping);
    from.addDependant(to);
    to.addDependency(from, blobNamesMapping);
}

void printNodeConnections(const std::string& nodeName, const std::string& sourceNode, const Aliases& pairs) {
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

void setFailIfNotFailEarlier(ovms::Status& earlierStatusCode, ovms::Status& newFailStatus) {
    if (earlierStatusCode.ok()) {
        earlierStatusCode = newFailStatus;
    }
}

#define IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE \
    if (!firstErrorStatus.ok()) {                                                       \
        if (finishedSessions.size() == startedSessions.size()) {                        \
            break;                                                                      \
        } else {                                                                        \
            continue;                                                                   \
        }                                                                               \
    }

#define CHECK_AND_LOG_ERROR(NODE)                                                                                                          \
    if (!status.ok()) {                                                                                                                    \
        setFailIfNotFailEarlier(firstErrorStatus, status);                                                                                 \
        SPDLOG_LOGGER_WARN(dag_executor_logger, "Executing pipeline: {} node: {} session: {} failed with ret code: {}, error message: {}", \
            getName(), NODE.getName(), sessionKey, status.getCode(), status.string());                                                     \
    }

Status Pipeline::execute() {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Started execution of pipeline: {}", getName());
    PipelineEventQueue finishedNodeQueue;
    ovms::Status firstErrorStatus{ovms::StatusCode::OK};
    std::set<std::string> startedSessions;
    std::set<std::string> finishedSessions;
    NodeSessionMetadata meta;
    // entry node does not have setInputsCalled so it has no
    // session created. Here is just assumption that this meta has the same key
    // that the one in EntryNode::execute();
    auto entrySessionKey = meta.getSessionKey();
    startedSessions.emplace(entry.getName() + entrySessionKey);
    ovms::Status status = entry.execute(entrySessionKey, finishedNodeQueue);  // first node will triger first message
    if (!status.ok()) {
        SPDLOG_LOGGER_WARN(dag_executor_logger, "Executing pipeline: {} node: {} failed with: {}",
            getName(), entry.getName(), status.string());
        return status;
    }
    std::vector<std::pair<std::reference_wrapper<Node>, session_key_t>> deferredNodeSessions;
    const uint WAIT_FOR_FINISHED_NODE_TIMEOUT_MICROSECONDS = 5000;
    const uint WAIT_FOR_DEFERRED_NODE_DISARM_TIMEOUT_MICROSECONDS = 500;
    // process finished session nodes and if no one is finished check if any node session with deferred execution
    // has necessary resources already
    while (true) {
        spdlog::trace("Pipeline: {} waiting for message that node finished.", getName());
        auto optionallyFinishedNode = finishedNodeQueue.tryPull(WAIT_FOR_FINISHED_NODE_TIMEOUT_MICROSECONDS);
        if (optionallyFinishedNode) {
            auto& [finishedNodeRef, sessionKey] = optionallyFinishedNode.value();
            Node& finishedNode = finishedNodeRef.get();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Pipeline: {} got message that node: {} session: {} finished.", getName(), finishedNode.getName(), sessionKey);
            finishedSessions.emplace(finishedNode.getName() + sessionKey);
            if (!firstErrorStatus.ok()) {
                finishedNode.release(sessionKey);
            }
            IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
            TensorMap finishedNodeOutputTensorMap;
            SessionResults sessionResults;
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Fetching results of pipeline: {} node: {} session: {}", getName(), finishedNode.getName(), sessionKey);
            status = finishedNode.fetchResults(sessionKey, sessionResults);
            CHECK_AND_LOG_ERROR(finishedNode)
            IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
            auto& nextNodesFromFinished = finishedNode.getNextNodes();
            for (auto& nextNode : nextNodesFromFinished) {
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "setting pipeline: {} node: {} session: {} outputs as inputs for node: {}",
                    getName(), finishedNode.getName(), sessionKey, nextNode.get().getName());
                status = nextNode.get().setInputs(finishedNode, sessionResults);
                CHECK_AND_LOG_ERROR(nextNode.get())
                if (!firstErrorStatus.ok()) {
                    break;
                }
            }
            finishedNodeOutputTensorMap.clear();
            for (auto& nextNode : nextNodesFromFinished) {
                auto readySessions = nextNode.get().getReadySessions();
                for (auto sessionKey : readySessions) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Started execution of pipeline: {} node: {} session: {}", getName(), nextNode.get().getName(), sessionKey);
                    startedSessions.emplace(nextNode.get().getName() + sessionKey);
                    status = nextNode.get().execute(sessionKey, finishedNodeQueue);
                    if (status == StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET) {
                        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} not ready for execution yet", nextNode.get().getName(), sessionKey);
                        deferredNodeSessions.emplace_back(nextNode.get(), sessionKey);
                        status = StatusCode::OK;
                    }
                    CHECK_AND_LOG_ERROR(nextNode.get())
                    if (!firstErrorStatus.ok()) {
                        break;
                    }
                }
            }
            if (startedSessions.size() == finishedSessions.size()) {
                break;
            }
        } else {
            // If error occurred earlier, disarm stream id guards of all deferred nodes and exit
            if (!firstErrorStatus.ok()) {
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Will try to disarm all stream id guards of all {} deferred node sessions due to previous error in pipeline", deferredNodeSessions.size());
                if (deferredNodeSessions.size() > 0) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Trying to disarm {} remaining deferred node sessions ...", deferredNodeSessions.size());
                    for (auto it = deferredNodeSessions.begin(); it != deferredNodeSessions.end();) {
                        auto& [nodeRef, sessionKey] = *it;
                        auto& node = nodeRef.get();
                        if (node.tryDisarm(sessionKey, WAIT_FOR_DEFERRED_NODE_DISARM_TIMEOUT_MICROSECONDS)) {
                            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Stream id guard disarm of node {} session: {} has succeeded", node.getName(), sessionKey);
                            finishedSessions.emplace(node.getName() + sessionKey);
                            it = deferredNodeSessions.erase(it);
                        } else {
                            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Cannot disarm stream id guard of node: {}, session: {} yet, will try again later", node.getName(), sessionKey);
                            it++;
                        }
                    }
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Disarming iteration completed, remaining deferred node sessions count: {}", deferredNodeSessions.size());
                }
                // Check for deferred node queue size again to indicate if all nodes got freed
                if (deferredNodeSessions.size() > 0) {
                    continue;
                } else {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Disarming all stream id guards of deferred nodes completed, pipeline will shut down");
                    IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
                }
            }
            // else scope could be executed always however it seems most reasonable at the time to
            // free blocked inferRequests from exeuction first rather than free models for reloading
            for (auto it = deferredNodeSessions.begin(); it != deferredNodeSessions.end();) {
                auto& [nodeRef, sessionKey] = *it;
                auto& node = nodeRef.get();
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Trying to trigger node: {} session: {} execution", node.getName(), sessionKey);
                status = node.execute(sessionKey, finishedNodeQueue);
                if (status.ok()) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} is ready", node.getName(), sessionKey);
                    it = deferredNodeSessions.erase(it);
                    continue;
                }
                it++;
                if (status == StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET) {
                    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} not ready for execution yet", node.getName(), sessionKey);
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
