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
    SPDLOG_DEBUG(ss.str());
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

#define CHECK_AND_LOG_ERROR(NODE)                                   \
    if (!status.ok()) {                                             \
        setFailIfNotFailEarlier(firstErrorStatus, status);          \
        SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}", \
            getName(), NODE.getName(), status.string());            \
    }

Status Pipeline::execute() {
    SPDLOG_INFO("Started execution of pipeline:", getName());
    ThreadSafeQueue<std::reference_wrapper<Node>> finishedNodeQueue;
    ovms::Status firstErrorStatus{ovms::StatusCode::OK};
    auto startedExecute{prepareStatusMap()};
    auto finishedExecute{prepareStatusMap()};
    startedExecute.at(entry.getName()) = true;
    ovms::Status status = entry.execute(finishedNodeQueue);  // first node will triger first message
    if (!status.ok()) {
        SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
            getName(), entry.getName(), status.string());
        return status;
    }
    while (true) {
        SPDLOG_DEBUG("Pipeline:{} before get message", getName());
        auto& finishedNode = finishedNodeQueue.waitAndPull().get();
        SPDLOG_DEBUG("Pipeline:{} got message that node:{} finished.", getName(), finishedNode.getName());
        finishedExecute.at(finishedNode.getName()) = true;
        IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
        BlobMap finishedNodeOutputBlobMap;
        status = finishedNode.fetchResults(finishedNodeOutputBlobMap);
        CHECK_AND_LOG_ERROR(finishedNode)
        IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
        if (std::all_of(finishedExecute.begin(), finishedExecute.end(), [](auto pair) { return pair.second; })) {
            break;
        }
        auto& nextNodesFromFinished = finishedNode.getNextNodes();
        for (auto& nextNode : nextNodesFromFinished) {
            status = nextNode.get().setInputs(finishedNode, finishedNodeOutputBlobMap);
            CHECK_AND_LOG_ERROR(nextNode.get())
            IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
        }
        finishedNodeOutputBlobMap.clear();
        for (auto& nextNode : nextNodesFromFinished) {
            if (nextNode.get().isReady()) {
                startedExecute.at(nextNode.get().getName()) = true;
                SPDLOG_DEBUG("Started execution of pipeline:{} node:{}", getName(), nextNode.get().getName());
                status = nextNode.get().execute(finishedNodeQueue);
                CHECK_AND_LOG_ERROR(nextNode.get())
                IF_ERROR_OCCURRED_EARLIER_THEN_BREAK_IF_ALL_STARTED_FINISHED_CONTINUE_OTHERWISE
            }
        }
    }
    return firstErrorStatus;
}
}  // namespace ovms
