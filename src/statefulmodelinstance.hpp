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
#pragma once

#include <memory>
#include <string>

#include "modelconfig.hpp"
#include "modelinstance.hpp"
#include "sequence_manager.hpp"

namespace ovms {

const uint32_t NO_CONTROL_INPUT = 0;
const uint32_t SEQUENCE_START = 1;
const uint32_t SEQUENCE_END = 2;

/**
     * @brief This class contains all the information about inference engine model
     */
class StatefulModelInstance : public ModelInstance {
public:
    /**
         * @brief A default constructor
         */
    StatefulModelInstance(const std::string& name, model_version_t version) :
        ModelInstance(name, version) {}

private:
    static constexpr std::array<const char*, 2> SPECIAL_INPUT_NAMES{"sequence_id", "sequence_control_input"};
    SequenceManager sequenceManager;
    bool performLowLatencyTransformation;

protected:
    const Status preInferenceProcessing(InferenceEngine::InferRequest& inferRequest, SequenceProcessingSpec& sequenceProcessingSpec);

    const Status postInferenceProcessing(tensorflow::serving::PredictResponse* response,
        InferenceEngine::InferRequest& inferRequest, ProcessingSpec* processingSpecPtr);

    const Status validateNumberOfInputs(const tensorflow::serving::PredictRequest* request,
        const size_t expectedNumberOfInputs) override;

    const Status validate(const tensorflow::serving::PredictRequest* request, ProcessingSpec* processingSpecPtr) override;

    const Status validateSpecialKeys(const tensorflow::serving::PredictRequest* request, ProcessingSpec* processingSpecPtr);

    Status infer(const tensorflow::serving::PredictRequest* requestProto,
        tensorflow::serving::PredictResponse* responseProto,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) override;
};
}  // namespace ovms
