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
#include "statefulmodelinstance.hpp"


using namespace InferenceEngine;

namespace ovms {

uint64_t extractSequenceId(const tensorflow::TensorProto& proto) {
    if(proto.uint64_val_size() == 1)
        return proto.uint64_val(0);
    return 0;
}

uint32_t extractSequenceControlInput(const tensorflow::TensorProto& proto) {
    if(proto.uint32_val_size() == 1)
        return proto.uint32_val(0);
    return 0;
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

    auto it = request->inputs().find("sequence_id");
    if (it != request->inputs().end())
        sequenceId = extractSequenceId(it->second);
    it = request->inputs().find("sequence_control_input");
    if (it != request->inputs().end())
        sequenceControlInput = extractSequenceControlInput(it->second);

    spdlog::info("Received sequence id: {}", sequenceId);

    if (sequenceControlInput == 1) { // First request in the sequence 
        // if sequenceId == 0, sequenceManager will need to generate unique id 
        if (sequenceId != 0) {
            // TO DO: Validate if sequenceId is already in use
        }
        processingSpecPtr->setSequenceProcessingSpec(sequenceControlInput, sequenceId);
        return StatusCode::OK;
    } else if (sequenceControlInput == 2 || sequenceControlInput == 0) { // Intermediate and last request in the sequence 
        if(sequenceId != 0) {
            // TO DO: check if sequence with provided ID exists
            processingSpecPtr->setSequenceProcessingSpec(sequenceControlInput, sequenceId);
            return StatusCode::OK;
        } else {
            // return 404 Missing sequence status
            return StatusCode::NOT_IMPLEMENTED;
        }
    }  else { 
        // 400 Bad Request - invalid control input
        return StatusCode::NOT_IMPLEMENTED;
    }
}

const Status StatefulModelInstance::validate(const tensorflow::serving::PredictRequest* request, ProcessingSpec* processingSpecPtr) {
    auto status = validateSpecialKeys(request, processingSpecPtr);
    if (!status.ok())
        return status; 
    return ModelInstance::validate(request, processingSpecPtr);
}


const Status StatefulModelInstance::preInferenceProcessing(const tensorflow::serving::PredictRequest* request, 
    InferenceEngine::InferRequest& inferRequest, ProcessingSpec* processingSpecPtr) {
    // Set infer request memory state to the last state saved by the sequence
    SequenceProcessingSpec& sequenceSpec = processingSpecPtr->getSequenceProcessingSpec();

    if (sequenceSpec.sequenceControlInput == 1) {
        // On SEQUENCE_START just add new sequence to sequence manager
        sequenceManager.addSequence(sequenceSpec.sequenceId);
    } else {
        // For next requests in the sequence restore model state
        const sequence_memory_state_t& sequenceMemoryState = sequenceManager.getSequenceMemoryState(sequenceSpec.sequenceId);
        for (auto &&state : inferRequest.QueryState()) {
            auto stateName = state.GetName();
            state.SetState(sequenceMemoryState.at(stateName));
            // TO DO: Error handling
        }
    }
    return StatusCode::OK;
}

const Status StatefulModelInstance::postInferenceProcessing(tensorflow::serving::PredictResponse* response, 
    InferenceEngine::InferRequest& inferRequest, ProcessingSpec* processingSpecPtr) {
    
    SequenceProcessingSpec& sequenceSpec = processingSpecPtr->getSequenceProcessingSpec();
    // Reset inferRequest states on SEQUENCE_END
    // TO DO: Move 2 to const variable
    if (sequenceSpec.sequenceControlInput == 2) {
        spdlog::info("Received SEQUENCE_END signal. Reseting model state and removing sequence");
        for (auto &&state : inferRequest.QueryState()) {
            state.Reset();
        }
        sequenceManager.removeSequence(sequenceSpec.sequenceId);
    } else {
        auto modelState = inferRequest.QueryState();
        sequenceManager.updateSequenceMemoryState(sequenceSpec.sequenceId, modelState);
    }

    // Include sequence_id in server response
    auto& tensorProto = (*response->mutable_outputs())["sequence_id"];
    tensorProto.add_uint64_val(sequenceSpec.sequenceId);

    spdlog::info("Added sequence_id to the response");
    return StatusCode::OK;
}

} //namespace ovms