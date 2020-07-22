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
std::map<const std::string, bool> Pipeline::prepareStatusMap() const {
    // void Pipeline::prepareStatusMap() const {
    std::map<const std::string, bool> nameFlagMap;
    for (const auto& node : nodes) {
        nameFlagMap.emplace(std::make_pair(node->getName(), false));
    }
    return std::move(nameFlagMap);
}

Status Pipeline::execute() {
    ThreadSafeQueue<std::reference_wrapper<Node>> finishedNodeQueue;
    bool errorOccured = false;
    auto started{prepareStatusMap()};
    auto finished{prepareStatusMap()};
    started.at(entry.getName()) = true;
    ovms::Status status = entry.execute(finishedNodeQueue);  // first node will triger first message
    if (!status.ok()) {
        SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
            getName(), entry.getName(), status.string());
        return status;
    }
    while (true) {
        auto& finishedNode = finishedNodeQueue.waitAndPull().get();
        BlobMap finishedNodeOutputBlobMap;
        status = finishedNode.fetchResults(finishedNodeOutputBlobMap);
        finished.at(finishedNode.getName()) = true;
        if (!status.ok()) {
            errorOccured = true;
            SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
                getName(), finishedNode.getName(), status.string());
        }
        if (errorOccured) {
            if (finished == started) {
                break;
            } else {  // will wait for all triggered async inferences to finish
                continue;
            }
        }
        if (std::all_of(finished.begin(), finished.end(), [](auto pair) { return pair.second; })) {
            break;
        }
        auto& nextNodesFromFinished = finishedNode.getNextNodes();
        for (auto& nextNode : nextNodesFromFinished) {
            nextNode.get().setInputs(finishedNode, finishedNodeOutputBlobMap);
        }
        finishedNodeOutputBlobMap.clear();
        for (auto& nextNode : nextNodesFromFinished) {
            if (nextNode.get().isReady()) {
                started.at(nextNode.get().getName()) = true;
                status = nextNode.get().execute(finishedNodeQueue);
                if (!status.ok()) {
                    errorOccured = true;
                    SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
                        getName(), nextNode.get().getName(), status.string());
                }
            }
        }
    }
    if (errorOccured) {
        // TODO decide what status code it should be - either previous failure
        // or general PIPELINE_FAILED since previous issue was already reported
        return StatusCode::UNKNOWN_ERROR;
    }
    return std::move(status);
}
}  // namespace ovms
