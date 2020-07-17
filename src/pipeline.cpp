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
namespace ovms {
Status Pipeline::execute() {
    // before event loop
    // deserialize to OV Blobs
    // event loop
    // trigger first Node

    // ovms::Status status = entry.execute();
    // if(!status.ok()) {
    //     SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
    //         pipeline.getName(), node.getName(), status.string());
    //     return status;
    // }
    // // first node will triger first message
    // while (true)
    // {
    //     ovms::Message message{messageQueue.waitAndPull()};
    //     status = message.getStatus();
    //     if(!status.ok()) {
    //         SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
    //             pipeline.getName(), node.getName(), status.string());
    //         //cannot return need to check if any infer is in progress
    //     }
    //     auto finishedNode = message.getNode();
    //     std::map<string, OV:Blob> finishedNodeOutputBlobMap;
    //     status = finishedNode->getResultsAndClearInputs(std::move(message), finishedNodeOutputBlobMap);
    //     if (!status.ok()) {
    //         SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
    //             pipeline.getName(), node.getName(), status.string());
    //         //cannot return need to check if any infer is in progress
    //     }
    //     // either this or use pipeline vector of std::atomic/int to keep
    //     auto& finishedNodeDependants = finishedNode.getDependants();
    //     for (auto& dependant : finishedNodeDependants) {
    //         dependant.setInputs(finishedNode.getName(), finishedNodeOutputBlobMap);
    //     finishedNodeBlobs.clear();
    //     for (auto& dependant : finishedNodeDependants) {

    //         dependant.executeIfReady();
    //         status = dependant.execute();
    //         if (!status.ok()) {
    //             SPDLOG_INFO("Executing pipeline:{} node:{} failed with:{}",
    //                 pipeline.getName(), dependant.getName(), status.string());
    //         //cannot return need to check if any infer is in progress
    //         }
    //     }
    // need to extrac which node finished then unblock further infers by destroying message
    // }

    // after event loop
    // serialize to TFProto
    return StatusCode::OK;
}
}  // namespace ovms
