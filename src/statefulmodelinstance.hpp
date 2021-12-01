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
#include <set>
#include <string>

#include "global_sequences_viewer.hpp"
#include "modelconfig.hpp"
#include "modelinstance.hpp"
#include "sequence_manager.hpp"

namespace ovms {

class StatefulModelInstance : public ModelInstance {
    static const std::set<const char*> SPECIAL_INPUT_NAMES;

public:
    /**
         * @brief A default constructor
         */
    StatefulModelInstance(const std::string& name, model_version_t version, InferenceEngine::Core& ieCore, ov::runtime::Core& ieCore_2, GlobalSequencesViewer* globalSequencesViewer) :
        ModelInstance(name, version, ieCore, ieCore_2),
        globalSequencesViewer(globalSequencesViewer) {
        sequenceManager = std::make_shared<SequenceManager>(config.getMaxSequenceNumber(), name, version);
    }

    const std::shared_ptr<SequenceManager>& getSequenceManager() const {
        return this->sequenceManager;
    }

    const Status extractSequenceId(const tensorflow::TensorProto& proto, uint64_t& sequenceId);

    const Status extractSequenceControlInput(const tensorflow::TensorProto& proto, uint32_t& sequenceControlInput);
    /*
    Performs pre inference operations:
        - for SEQUENCE_START control input - reset InferRequest memory state
        - for SEQUENCE_END control input or for no control input - load sequence memory state into InferRequest

        Always returns StatusCode::OK
    */
    const Status preInferenceProcessing(InferenceEngine::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec);
    const Status preInferenceProcessing_2(ov::runtime::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec);

    /*
    Performs pre inference operations:
        - for SEQUENCE_START or for no control input - save InferRequest memory state in sequence memory state
        - for SEQUENCE_END control input - reset InferRequest memory state
        - for all requests - append sequence id to the response

        Always returns StatusCode::OK
    */
    const Status postInferenceProcessing(tensorflow::serving::PredictResponse* response,
        InferenceEngine::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec);
    const Status postInferenceProcessing_2(tensorflow::serving::PredictResponse* response,
        ov::runtime::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec);

    Status infer(const tensorflow::serving::PredictRequest* requestProto,
        tensorflow::serving::PredictResponse* responseProto,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) override;

    Status loadModel(const ModelConfig& config) override;

    Status reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter()) override;

    void retireModel(bool isPermanent = true) override;

    void cleanupFailedLoad() override;

protected:
    std::shared_ptr<SequenceManager> sequenceManager;

    bool performLowLatencyTransformation;

    bool autoCleanupEnabled;

    GlobalSequencesViewer* globalSequencesViewer;

    const Status validate(const tensorflow::serving::PredictRequest* request, SequenceProcessingSpec& processingSpec);

    Status loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter()) override;

    Status loadOVExecutableNetwork(const ModelConfig& config) override;

private:
    const Status validateSpecialKeys(const tensorflow::serving::PredictRequest* request, SequenceProcessingSpec& sequenceProcessingSpec);
};
}  // namespace ovms
