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
#include "statefulmodelinstance.hpp"

#include "sequence.hpp"

using namespace InferenceEngine;

namespace ovms {

const Status extractSequenceId(const tensorflow::TensorProto& proto, uint64_t& sequenceId) {
    if (proto.uint64_val_size() == 1) {
        sequenceId = proto.uint64_val(0);
        return StatusCode::OK;
    }
    return StatusCode::SEQUENCE_ID_BAD_TYPE;
}

const Status extractSequenceControlInput(const tensorflow::TensorProto& proto, uint32_t& sequenceControlInput) {
    if (proto.uint32_val_size() == 1) {
        sequenceControlInput = proto.uint32_val(0);
        return StatusCode::OK;
    }
    return StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE;
}

const Status StatefulModelInstance::validateNumberOfInputs(const tensorflow::serving::PredictRequest* request, const size_t expectedNumberOfInputs) {
    // Begin with number of inputs required by the model and increase it with special inputs for sequence handling
    auto completeInputsNumber = expectedNumberOfInputs;
    for (auto specialInputName : SPECIAL_INPUT_NAMES) {
        if (request->inputs().count(specialInputName))
            completeInputsNumber++;
    }
    return ModelInstance::validateNumberOfInputs(request, completeInputsNumber);
}

const Status StatefulModelInstance::validateSpecialKeys(const tensorflow::serving::PredictRequest* request, ProcessingSpec* processingSpecPtr) {
    uint64_t sequenceId = 0;
    uint32_t sequenceControlInput = 0;
    Status status;
    auto it = request->inputs().find("sequence_id");
    if (it != request->inputs().end()) {
        status = extractSequenceId(it->second, sequenceId);
        if (!status.ok())
            return status;
    }
    it = request->inputs().find("sequence_control_input");
    if (it != request->inputs().end()) {
        status = extractSequenceControlInput(it->second, sequenceControlInput);
        if (!status.ok())
            return status;
    }

    if (sequenceControlInput != SEQUENCE_END && sequenceControlInput != NO_CONTROL_INPUT && sequenceControlInput != SEQUENCE_START) {
        return StatusCode::INVALID_SEQUENCE_CONTROL_INPUT;
    }
    if ((sequenceControlInput == SEQUENCE_END || sequenceControlInput == NO_CONTROL_INPUT) && sequenceId == 0) {
        return StatusCode::SEQUENCE_ID_NOT_PROVIDED;
    }

    processingSpecPtr->setSequenceProcessingSpec(sequenceControlInput, sequenceId);
    return StatusCode::OK;
}

const Status StatefulModelInstance::validate(const tensorflow::serving::PredictRequest* request, ProcessingSpec* processingSpecPtr) {
    auto status = validateSpecialKeys(request, processingSpecPtr);
    if (!status.ok())
        return status;

    return ModelInstance::validate(request, processingSpecPtr);
}

Status StatefulModelInstance::infer(const tensorflow::serving::PredictRequest* requestProto,
    tensorflow::serving::PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    return StatusCode::OK;
}

const Status StatefulModelInstance::preInferenceProcessing(InferenceEngine::InferRequest& inferRequest, SequenceProcessingSpec& sequenceProcessingSpec) {
    if (sequenceProcessingSpec.sequenceControlInput == SEQUENCE_START) {
        // On SEQUENCE_START reset memory state of infer request to default
        for (auto&& state : inferRequest.QueryState()) {
            state.Reset();
        }
    } else {
        // For next requests in the sequence set infer request memory state to the last state saved by the sequence
        const sequence_memory_state_t& sequenceMemoryState = sequenceManager.getSequenceMemoryState(sequenceProcessingSpec.sequenceId);
        for (auto&& state : inferRequest.QueryState()) {
            auto stateName = state.GetName();
            if (!sequenceMemoryState.count(stateName))
                return StatusCode::INTERNAL_ERROR;
            state.SetState(sequenceMemoryState.at(stateName));
        }
    }
    return StatusCode::OK;
}

const Status StatefulModelInstance::postInferenceProcessing(tensorflow::serving::PredictResponse* response,
    InferenceEngine::InferRequest& inferRequest, SequenceProcessingSpec& sequenceProcessingSpec) {

    SequenceProcessingSpec& sequenceSpec = sequenceProcessingSpec.getSequenceProcessingSpec();
    // Reset inferRequest states on SEQUENCE_END
    if (sequenceSpec.sequenceControlInput == SEQUENCE_END) {
        spdlog::debug("Received SEQUENCE_END signal. Reseting model state and removing sequence");
        for (auto &&state : inferRequest.QueryState()) {
            state.Reset();
        }
    }
    else {
        auto modelState = inferRequest.QueryState();
        sequenceManager.updateSequenceMemoryState(sequenceSpec.sequenceId, modelState);
    }

    // Include sequence_id in server response
    auto& tensorProto = (*response->mutable_outputs())["sequence_id"];
    tensorProto.add_uint64_val(sequenceSpec.sequenceId);

    return StatusCode::OK;
}
}  // namespace ovms
