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

#include "logging.hpp"
#include "ov_utils.hpp"

namespace ovms {

GatherNodeInputHandler::GatherNodeInputHandler(uint32_t inputsMissingCount, session_id_t shardsCount) :
    NodeInputHandler(inputsMissingCount) {
    remainingDependencies *= shardsCount;
}

void GatherNodeInputHandler::setInput(const std::string& inputName, InferenceEngine::Blob::Ptr& ptr, session_id_t shardId) {
    auto it = shardsStorage.find(inputName);
    if (it == shardsStorage.end()) {
        shard_map_t shardMap{{shardId, ptr}};
        auto itDidInsertPair = shardsStorage.emplace(inputName, std::move(shardMap));
        if (!itDidInsertPair.second) {
            throw std::runtime_error("Tried to insert the same input twice with the same shard id");
        }
    } else {
        it->second.emplace(shardId, ptr);
    }
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
        auto tensorDesc = shardMap.at(firstShardId)->getTensorDesc();
        auto newDims = tensorDesc.getDims();
        newDims.insert(newDims.begin(), 1);
        newDims.at(1) = shardsCount;
        const InferenceEngine::TensorDesc consolidatedBlobDesc(
            tensorDesc.getPrecision(),
            newDims,
            InferenceEngine::Layout::ANY);
        InferenceEngine::Blob::Ptr consolidatedBlob;
        auto status = createSharedBlob(consolidatedBlob, consolidatedBlobDesc);
        if (!status.ok()) {
            return status;
        }
        for (auto& [shardId, blob] : shardMap) {
            const auto memstep = blob->byteSize();
            size_t offset = shardId * memstep;
            memcpy((char*)consolidatedBlob->buffer() + offset, blob->cbuffer(), memstep);
        }
        inputBlobs.insert({inputName, consolidatedBlob});
    }
    return StatusCode::OK;
}
}  // namespace ovms
