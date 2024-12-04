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
#include <optional>
#include <set>
#include <string>

#include "global_sequences_viewer.hpp"
#include "modelinstance.hpp"
#include "sequence_manager.hpp"
#include "sequence_processing_spec.hpp"

namespace ovms {
class ModelConfig;
class StatefulModelInstance : public ModelInstance {
    static const std::set<std::string> SPECIAL_INPUT_NAMES;

public:
    /**
         * @brief A default constructor
         */
    StatefulModelInstance(const std::string& name, model_version_t version, ov::Core& ieCore, MetricRegistry* registry, const MetricConfig* metricsConfig = nullptr, GlobalSequencesViewer* globalSequencesViewer = nullptr) :
        ModelInstance(name, version, ieCore, registry, metricsConfig),
        globalSequencesViewer(globalSequencesViewer) {
        sequenceManager = std::make_shared<SequenceManager>(config.getMaxSequenceNumber(), name, version);
    }

    const std::shared_ptr<SequenceManager>& getSequenceManager() const override {
        return this->sequenceManager;
    }

    static const Status extractSequenceId(const tensorflow::TensorProto& proto, uint64_t& sequenceId);

    static const Status extractSequenceControlInput(const tensorflow::TensorProto& proto, uint32_t& sequenceControlInput);
    /*
    Performs pre inference operations:
        - for SEQUENCE_START control input - reset InferRequest memory state
        - for SEQUENCE_END control input or for no control input - load sequence memory state into InferRequest

        Always returns StatusCode::OK
    */
    const Status preInferenceProcessing(ov::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec);

    /*
    Performs pre inference operations:
        - for SEQUENCE_START or for no control input - save InferRequest memory state in sequence memory state
        - for SEQUENCE_END control input - reset InferRequest memory state
        - for all requests - append sequence id to the response

        Always returns StatusCode::OK
    */
    const Status postInferenceProcessing(tensorflow::serving::PredictResponse* response,
        ov::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec);

    Status loadModel(const ModelConfig& config) override;

    Status reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter()) override;

    void retireModel(bool isPermanent = true) override;

    void cleanupFailedLoad() override;

protected:
    std::shared_ptr<SequenceManager> sequenceManager{nullptr};

    bool performLowLatencyTransformation = false;

    GlobalSequencesViewer* globalSequencesViewer;

    Status loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter()) override;

    Status loadOVCompiledModel(const ModelConfig& config) override;

public:
    template <typename RequestType>
    static const Status extractSpecialKeys(const RequestType* request, SequenceProcessingSpec& sequenceProcessingSpec);

    // TODO @atobisze std::unique_ptr<RequestProcessor<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>> createRequestProcessor(const tensorflow::serving::PredictRequest*, tensorflow::serving::PredictResponse*) override;
    const std::set<std::string>& getOptionalInputNames() override;
};
}  // namespace ovms
