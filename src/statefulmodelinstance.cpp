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

#include "deserialization.hpp"
#include "executingstreamidguard.hpp"
#include "logging.hpp"
#include "serialization.hpp"
#include "timer.hpp"

using namespace InferenceEngine;

namespace ovms {

const Status StatefulModelInstance::extractSequenceId(const tensorflow::TensorProto& proto, uint64_t& sequenceId) {
    if (!proto.tensor_shape().dim_size()) {
        SPDLOG_DEBUG("[Model: {} version: {}] Sequence id tensor proto does not contain tensor shape information", getName(), getVersion());
        return StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE;
    } else if (proto.tensor_shape().dim_size() != 1) {
        SPDLOG_DEBUG("[Model: {} version: {}] Sequence id tensor proto shape has invalid number of dimensions. Expecting shape with one dimension", getName(), getVersion());
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, "Required shape for sequence_id is: (1)");
    }

    if (proto.tensor_shape().dim(0).size() != 1) {
        SPDLOG_DEBUG("[Model: {} version: {}] Sequence id tensor proto shape has invalid shape. Expecting shape: (1)", getName(), getVersion());
        return Status(StatusCode::INVALID_SHAPE, "Required shape for sequence_id is: (1)");
    }

    if (proto.uint64_val_size() == 1) {
        sequenceId = proto.uint64_val(0);
        return StatusCode::OK;
    }
    return StatusCode::SEQUENCE_ID_BAD_TYPE;
}

const Status StatefulModelInstance::extractSequenceControlInput(const tensorflow::TensorProto& proto, uint32_t& sequenceControlInput) {
    if (proto.tensor_shape().dim_size() == 0) {
        SPDLOG_DEBUG("[Model: {} version: {}] Sequence control tensor proto does not contain tensor shape information", getName(), getVersion());
        return StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE;
    } else if (proto.tensor_shape().dim_size() != 1) {
        SPDLOG_DEBUG("[Model: {} version: {}] Sequence control tensor proto shape has invalid number of dimensions. Expecting shape with one dimension.", getName(), getVersion());
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, "Required shape for sequence_control_input is: (1)");
    }

    if (proto.tensor_shape().dim(0).size() != 1) {
        SPDLOG_DEBUG("[Model: {} version: {}] Sequence control tensor proto shape has invalid shape. Expecting shape: (1)", getName(), getVersion());
        return Status(StatusCode::INVALID_SHAPE, "Required shape for sequence_control_input is: (1)");
    }

    if (proto.uint32_val_size() == 1) {
        sequenceControlInput = proto.uint32_val(0);
        return StatusCode::OK;
    }
    return StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE;
}

Status StatefulModelInstance::loadModel(const ModelConfig& config) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    autoCleanupEnabled = config.getIdleSequenceCleanup();

    Status status = ModelInstance::loadModel(config);
    if (!status.ok())
        return status;

    if (autoCleanupEnabled) {
        status = globalSequencesViewer->registerForCleanup(getName(), getVersion(), sequenceManager);
        if (!status.ok())
            return status;
    }
    return StatusCode::OK;
}

Status StatefulModelInstance::reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    Status status;
    if (autoCleanupEnabled && this->status.getState() != ModelVersionState::END) {
        status = globalSequencesViewer->unregisterFromCleanup(getName(), getVersion());
        if (!status.ok())
            return status;
    }
    status = ModelInstance::reloadModel(config, parameter);
    if (!status.ok())
        return status;
    autoCleanupEnabled = config.getIdleSequenceCleanup();

    if (autoCleanupEnabled) {
        status = globalSequencesViewer->registerForCleanup(getName(), getVersion(), sequenceManager);
        if (!status.ok())
            return status;
    }
    return StatusCode::OK;
}

void StatefulModelInstance::retireModel(bool isPermanent) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    if (isPermanent && autoCleanupEnabled) {
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

Status StatefulModelInstance::loadOVExecutableNetwork(const ModelConfig& config) {
    if (performLowLatencyTransformation) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "[Model: {} version: {}] Performing Low Latency Transformation on the network", getName(), getVersion());
        try {
            InferenceEngine::lowLatency2(*network);
        } catch (std::exception& ex) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error: {}; occurred during low latency transformation on model: {} version: {}", ex.what(), getName(), getVersion());
            return StatusCode::INTERNAL_ERROR;
        }
    }
    return ModelInstance::loadOVExecutableNetwork(config);
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

const Status StatefulModelInstance::validateSpecialKeys(const tensorflow::serving::PredictRequest* request, SequenceProcessingSpec& sequenceProcessingSpec) {
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

    sequenceProcessingSpec.setSequenceId(sequenceId);
    sequenceProcessingSpec.setSequenceControlInput(sequenceControlInput);

    return StatusCode::OK;
}

const Status StatefulModelInstance::validate(const tensorflow::serving::PredictRequest* request, SequenceProcessingSpec& sequenceProcessingSpec) {
    auto status = validateSpecialKeys(request, sequenceProcessingSpec);
    if (!status.ok())
        return status;

    return ModelInstance::validate(request);
}

Status StatefulModelInstance::infer(const tensorflow::serving::PredictRequest* requestProto,
    tensorflow::serving::PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    Timer timer;
    using std::chrono::microseconds;
    SequenceProcessingSpec sequenceProcessingSpec;
    auto status = validate(requestProto, sequenceProcessingSpec);
    if (!status.ok())
        return status;

    std::unique_lock<std::mutex> sequenceManagerLock(sequenceManager->getMutex());
    status = sequenceManager->processRequestedSpec(sequenceProcessingSpec);
    if (!status.ok())
        return status;
    const uint64_t sequenceId = sequenceProcessingSpec.getSequenceId();
    if (!sequenceManager->sequenceExists(sequenceId))
        return StatusCode::INTERNAL_ERROR;
    Sequence& sequence = sequenceManager->getSequence(sequenceId);

    std::unique_lock<std::mutex> sequenceLock(sequence.getMutex());
    sequenceManagerLock.unlock();

    timer.start("get infer request");
    ExecutingStreamIdGuard executingStreamIdGuard(getInferRequestsQueue());
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    timer.stop("get infer request");
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("get infer request") / 1000);

    timer.start("preprocess");
    status = preInferenceProcessing(inferRequest, sequence, sequenceProcessingSpec);
    timer.stop("preprocess");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Preprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("preprocess") / 1000);

    timer.start("deserialize");
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, getInputsInfo(), inferRequest);
    timer.stop("deserialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("deserialize") / 1000);

    timer.start("prediction");
    status = performInference(inferRequest);
    timer.stop("prediction");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("prediction") / 1000);

    timer.start("serialize");
    status = serializePredictResponse(inferRequest, getOutputsInfo(), responseProto);
    timer.stop("serialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("serialize") / 1000);

    timer.start("postprocess");
    status = postInferenceProcessing(responseProto, inferRequest, sequence, sequenceProcessingSpec);
    timer.stop("postprocess");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Postprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("postprocess") / 1000);

    sequenceLock.unlock();
    if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_END) {
        sequenceManagerLock.lock();
        status = sequenceManager->removeSequence(sequenceId);
        if (!status.ok())
            return status;
    }

    return StatusCode::OK;
}

const Status StatefulModelInstance::preInferenceProcessing(InferenceEngine::InferRequest& inferRequest, Sequence& sequence,
    SequenceProcessingSpec& sequenceProcessingSpec) {
    if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_START) {
        // On SEQUENCE_START reset memory state of infer request to default
        for (auto&& state : inferRequest.QueryState()) {
            state.Reset();
        }
    } else {
        // For next requests in the sequence set infer request memory state to the last state saved by the sequence
        const sequence_memory_state_t& sequenceMemoryState = sequence.getMemoryState();
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
    InferenceEngine::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec) {
    // Reset inferRequest states on SEQUENCE_END
    if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_END) {
        spdlog::debug("Received SEQUENCE_END signal. Reseting model state and removing sequence");
        for (auto&& state : inferRequest.QueryState()) {
            state.Reset();
        }
    } else {
        auto modelState = inferRequest.QueryState();
        sequence.updateMemoryState(modelState);
    }

    // Include sequence_id in server response
    auto& tensorProto = (*response->mutable_outputs())["sequence_id"];
    tensorProto.mutable_tensor_shape()->add_dim()->set_size(1);
    tensorProto.set_dtype(tensorflow::DataType::DT_UINT64);
    tensorProto.add_uint64_val(sequenceProcessingSpec.getSequenceId());

    return StatusCode::OK;
}
}  // namespace ovms
