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

#include <openvino/openvino.hpp>
#include <openvino/pass/low_latency.hpp>

#include "executingstreamidguard.hpp"
#include "logging.hpp"
#include "model_metric_reporter.hpp"
#include "modelconfig.hpp"
#include "predict_request_validation_utils.hpp"
#include "profiler.hpp"
#include "sequence_processing_spec.hpp"
#include "timer.hpp"

namespace ovms {

const std::set<std::string> StatefulModelInstance::SPECIAL_INPUT_NAMES{"sequence_id", "sequence_control_input"};

Status StatefulModelInstance::loadModel(const ModelConfig& config) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);

    Status status = ModelInstance::loadModel(config);
    if (!status.ok())
        return status;

    if (this->config.getIdleSequenceCleanup()) {
        status = globalSequencesViewer->registerForCleanup(getName(), getVersion(), sequenceManager);
        if (!status.ok())
            return status;
    }
    return StatusCode::OK;
}

Status StatefulModelInstance::reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    Status status;
    if (this->config.getIdleSequenceCleanup() && this->status.getState() == ModelVersionState::AVAILABLE) {
        status = globalSequencesViewer->unregisterFromCleanup(getName(), getVersion());
        if (!status.ok())
            return status;
    }
    status = ModelInstance::reloadModel(config, parameter);
    if (!status.ok())
        return status;

    if (this->config.getIdleSequenceCleanup()) {
        status = globalSequencesViewer->registerForCleanup(getName(), getVersion(), sequenceManager);
        if (!status.ok())
            return status;
    }
    return StatusCode::OK;
}

void StatefulModelInstance::retireModel(bool isPermanent) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    if (isPermanent && this->config.getIdleSequenceCleanup()) {
        globalSequencesViewer->unregisterFromCleanup(getName(), getVersion());
    }
    ModelInstance::retireModel(isPermanent);
    sequenceManager.reset();
}

void StatefulModelInstance::cleanupFailedLoad() {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    ModelInstance::cleanupFailedLoad();
    sequenceManager.reset();
}

Status StatefulModelInstance::loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter) {
    performLowLatencyTransformation = config.isLowLatencyTransformationUsed();
    sequenceManager = std::make_shared<SequenceManager>(config.getMaxSequenceNumber(), config.getName(), config.getVersion());
    return ModelInstance::loadModelImpl(config, parameter);
}

Status StatefulModelInstance::loadOVCompiledModel(const ModelConfig& config) {
    if (performLowLatencyTransformation) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "[Model: {} version: {}] Performing Low Latency Transformation on the model", getName(), getVersion());
        try {
            ov::pass::LowLatency2().run_on_model(model);
        } catch (ov::Exception& ex) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error: {}; occurred during low latency transformation on model: {} version: {}", ex.what(), getName(), getVersion());
            return StatusCode::INTERNAL_ERROR;
        } catch (std::exception& ex) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error: {}; occurred during low latency transformation on model: {} version: {}", ex.what(), getName(), getVersion());
            return StatusCode::INTERNAL_ERROR;
        }
    }
    return ModelInstance::loadOVCompiledModel(config);
}

const std::set<std::string>& StatefulModelInstance::getOptionalInputNames() {
    return SPECIAL_INPUT_NAMES;
}

const Status StatefulModelInstance::preInferenceProcessing(ov::InferRequest& inferRequest, Sequence& sequence,
    SequenceProcessingSpec& sequenceProcessingSpec) {
    if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_START) {
        // On SEQUENCE_START reset memory state of infer request to default
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    } else {
        // For next requests in the sequence set infer request memory state to the last state saved by the sequence
        const sequence_memory_state_t& sequenceMemoryState = sequence.getMemoryState();
        for (auto&& state : inferRequest.query_state()) {
            auto stateName = state.get_name();
            if (!sequenceMemoryState.count(stateName))
                return StatusCode::INTERNAL_ERROR;
            state.set_state(sequenceMemoryState.at(stateName));
        }
    }
    return StatusCode::OK;
}
}  // namespace ovms
