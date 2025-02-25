//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include <memory>
#include <optional>
#include <thread>

#include "requestprocessor.hpp"
#include "sequence_manager.hpp"
#include "sequence.hpp"
#include "sequence_processing_spec.hpp"
#include "status.hpp"

namespace ovms {
template <typename RequestType, typename ResponseType>
struct StatefulRequestProcessor : public RequestProcessor<RequestType, ResponseType> {
    SequenceManager& sequenceManager;
    std::unique_ptr<std::unique_lock<std::mutex>> sequenceManagerLock;
    std::unique_ptr<std::unique_lock<std::mutex>> sequenceLock;
    SequenceProcessingSpec sequenceProcessingSpec;
    Sequence* sequence{nullptr};
    std::optional<uint64_t> sequenceId;

    StatefulRequestProcessor(SequenceManager& sequenceManager) : sequenceManager(sequenceManager) {}
    Status prepare() override {
        sequenceManagerLock = std::make_unique<std::unique_lock<std::mutex>>(sequenceManager.getMutex());
        auto status = sequenceManager.processRequestedSpec(sequenceProcessingSpec);
        if (!status.ok())
            return status;
        this->sequenceId = sequenceProcessingSpec.getSequenceId();
        if (!sequenceManager.sequenceExists(this->sequenceId.value()))
            return StatusCode::INTERNAL_ERROR;
        sequence = &sequenceManager.getSequence(this->sequenceId.value());
    
        sequenceLock = std::make_unique<std::unique_lock<std::mutex>>(sequence->getMutex());
        sequenceManagerLock->unlock();
        return StatusCode::OK;
    }
    Status preInferenceProcessing(ov::InferRequest& inferRequest) override {
        if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_START) {
            // On SEQUENCE_START reset memory state of infer request to default
            for (auto&& state : inferRequest.query_state()) {
                state.reset();
            }
        } else {
            // For next requests in the sequence set infer request memory state to the last state saved by the sequence
            const sequence_memory_state_t& sequenceMemoryState = sequence->getMemoryState();
            for (auto&& state : inferRequest.query_state()) {
                auto stateName = state.get_name();
                if (!sequenceMemoryState.count(stateName))
                    return StatusCode::INTERNAL_ERROR;
                state.set_state(sequenceMemoryState.at(stateName));
            }
        }
        return StatusCode::OK;
    }
    // TODO @atobisze force check status
    Status extractRequestParameters(const RequestType* request) override { return StatusCode::NOT_IMPLEMENTED;};
    Status postInferenceProcessing(ResponseType* response, ov::InferRequest& inferRequest) override { return StatusCode::NOT_IMPLEMENTED;};
    Status release() override { return StatusCode::NOT_IMPLEMENTED;};
};
}  // namespace ovms
