//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "gathernodeinputhandler.hpp"

#include <functional>

#include "logging.hpp"
#include "nodesessionmetadata.hpp"
#include "ov_utils.hpp"
#include "tensorinfo.hpp"

namespace ovms {

GatherNodeInputHandler::GatherNodeInputHandler(uint32_t inputsMissingCount, const CollapseDetails& collapsingDetails) :
    NodeInputHandler(inputsMissingCount),
    collapsingDetails(std::make_unique<CollapseDetails>(collapsingDetails)) {
    remainingDependencies = std::accumulate(
        collapsingDetails.collapsedSessionSizes.begin(),
        collapsingDetails.collapsedSessionSizes.end(),
        remainingDependencies,
        std::multiplies<session_id_t>());
}

Status GatherNodeInputHandler::setInput(const std::string& inputName, InferenceEngine::Blob::Ptr& ptr, session_id_t shardId) {
    auto inputsShardsIt = shardsStorage.find(inputName);
    if (inputsShardsIt == shardsStorage.end()) {
        shard_map_t shardMap{{shardId, ptr}};
        auto itDidInsertPair = shardsStorage.emplace(inputName, std::move(shardMap));
        if (!itDidInsertPair.second) {
            throw std::runtime_error("Tried to insert the same input twice with the same shard id");
        }
    } else {
        auto firstShardTensor = inputsShardsIt->second.begin()->second;
        if (firstShardTensor->getTensorDesc() != ptr->getTensorDesc()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Shard: {} tensor description differ. First shard desc: {}, current shard desc: {}",
                shardId,
                TensorInfo("firstShard", firstShardTensor->getTensorDesc()).getPrintableString(),
                TensorInfo("currentShard", ptr->getTensorDesc()).getPrintableString());
            return StatusCode::PIPELINE_INCONSISTENT_SHARD_DIMENSIONS;
        }
        auto itDidEmplacePair = inputsShardsIt->second.emplace(shardId, ptr);
        if (!itDidEmplacePair.second) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to put the same input shard twice");  // TODO  improve error msg
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    return StatusCode::OK;
}

Status GatherNodeInputHandler::notifyFinishedDependency() {
    NodeInputHandler::notifyFinishedDependency();
    if (remainingDependencies > 0) {
        return StatusCode::OK;
    }
    for (auto& [inputName, shardMap] : shardsStorage) {
        const auto shardsCount = shardMap.size();
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Consolidating: {} shards for input: {}", shardsCount, inputName);
        session_id_t firstShardId = 0;
        auto firstShardTensorDesc = shardMap.at(firstShardId)->getTensorDesc();
        auto shardDims = firstShardTensorDesc.getDims();
        auto newDims = shardDims;
        // we leave batch size dimension untouched (pos=0) hence + 1
        newDims.insert(newDims.begin() + 1,
            collapsingDetails->collapsedSessionSizes.begin(),
            collapsingDetails->collapsedSessionSizes.end());
        const InferenceEngine::TensorDesc consolidatedBlobDesc(
            firstShardTensorDesc.getPrecision(),
            newDims,
            InferenceEngine::Layout::ANY);
        InferenceEngine::Blob::Ptr consolidatedBlob;
        auto status = createSharedBlob(consolidatedBlob, consolidatedBlobDesc);
        if (!status.ok()) {
            return status;
        }
        for (auto& [shardId, blob] : shardMap) {
            auto shardTensorDesc = blob->getTensorDesc();
            if (shardTensorDesc != firstShardTensorDesc) {
                SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to consolidate blob: {} shards in gather node. First shard has different tensor description: {} than current shard: {}",
                    inputName, TensorInfo(inputName, firstShardTensorDesc).getPrintableString(), TensorInfo(inputName, shardTensorDesc).getPrintableString());
                return StatusCode::PIPELINE_INCONSISTENT_SHARD_DIMENSIONS;
            }
            const auto memstep = blob->byteSize();
            size_t offset = shardId * memstep;
            memcpy((char*)consolidatedBlob->buffer() + offset, blob->cbuffer(), memstep);
        }
        inputBlobs.insert({inputName, consolidatedBlob});
    }
    return StatusCode::OK;
}
}  // namespace ovms
