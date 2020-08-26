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
#include "dl_node.hpp"

#include <map>

#include <inference_engine.hpp>
#include <spdlog/spdlog.h>

#include "modelmanager.hpp"
#include "ovinferrequestsqueue.hpp"
#include "prediction_service_utils.hpp"

namespace ovms {

const uint WAIT_FOR_STREAM_ID_TIMEOUT_MICROSECONDS = 1;

Status DLNode::execute(ThreadSafeQueue<std::reference_wrapper<Node>>& notifyEndQueue) {
    Status status;
    if (this->nodeStreamIdGuard == nullptr) {
        status = requestExecuteRequiredResources();
        if (!status.ok()) {
            notifyEndQueue.push(*this);
            return status;
        }
    }
    auto streamId = this->nodeStreamIdGuard->tryGetId(WAIT_FOR_STREAM_ID_TIMEOUT_MICROSECONDS);
    if (!streamId) {
        SPDLOG_DEBUG("Node:{} could not acquire stream Id right away", getName());
        return StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET;
    }
    auto& inferRequestsQueue = this->model->getInferRequestsQueue();
    auto& inferRequest = inferRequestsQueue.getInferRequest(streamId.value());
    status = setInputsForInference(inferRequest);
    if (!status.ok()) {
        notifyEndQueue.push(*this);
        return status;
    }
    status = executeInference(notifyEndQueue, inferRequest);
    if (!status.ok()) {
        notifyEndQueue.push(*this);
        return status;
    }
    return status;
}

Status DLNode::requestExecuteRequiredResources() {
    Status status = StatusCode::OK;
    status = getModelInstance(
        this->modelManager,
        this->modelName,
        this->modelVersion.value_or(0),
        this->model,
        this->modelUnloadGuard);

    if (!status.ok()) {
        SPDLOG_INFO("Getting modelInstance failed for node:{} with:{}", getName(), status.string());
        return status;
    }

    status = prepareInputsAndModelForInference();
    if (!status.ok()) {
        SPDLOG_INFO("DLNode::execute (Node name {}); error occurred during input/model preparation: {}", getName(), status.string());
        return status;
    }
    auto& inferRequestsQueue = this->model->getInferRequestsQueue();
    this->nodeStreamIdGuard = std::make_unique<NodeStreamIdGuard>(inferRequestsQueue);
    return status;
}

Status DLNode::setInputsForInference(InferenceEngine::InferRequest& infer_request) {
    Status status = StatusCode::OK;
    try {
        // Prepare inference request, fill with input blobs
        for (const auto& kv : this->inputBlobs) {
            infer_request.SetBlob(kv.first, kv.second);
        }
        // OV implementation the InferenceEngineException is not
        // a base class for all other exceptions thrown from OV.
        // OV can throw exceptions derived from std::logic_error.
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_INFO("DLNode::execute (Node name {}); error during InferRequest::SetBlob: {}; exception message: {}", getName(), status.string(), e.what());
    } catch (std::logic_error& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_INFO("DLNode::execute (Node name {}); error during InferRequest::SetBlob: {}; exception message: {}", getName(), status.string(), e.what());
    } catch (...) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_ERROR("DLNode::execute (Node name {}); error during InferRequest::SetBlob: {}; with unknown exception", getName(), status.string());
    }
    return status;
}

Status DLNode::executeInference(ThreadSafeQueue<std::reference_wrapper<Node>>& notifyEndQueue, InferenceEngine::InferRequest& infer_request) {
    try {
        SPDLOG_DEBUG("Setting completion callback for node name: {}", this->getName());
        infer_request.SetCompletionCallback([this, &notifyEndQueue, &infer_request]() {
            SPDLOG_DEBUG("Completion callback received for node name: {}", this->getName());
            // After inference is completed, input blobs are not needed anymore
            this->inputBlobs.clear();
            notifyEndQueue.push(*this);
            infer_request.SetCompletionCallback([]() {});  // reset callback on infer request
        });
        SPDLOG_DEBUG("Starting infer async for node name: {}", getName());
        infer_request.StartAsync();
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        SPDLOG_INFO("Exception occured when started async inference or setting completion callback on node:{}, modelName:{}, error:{}",
            getName(), modelName, e.what());
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    } catch (const std::exception& e) {
        SPDLOG_INFO("Exception occured when started async inference or setting completion callback on node:{}, modelName:{}, error:{}",
            getName(), modelName, e.what());
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    } catch (...) {
        SPDLOG_ERROR("Unknown exception occured when started async or setting completion callback inference on node:{}, modelName:{}",
            getName(), modelName);
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    }
    return StatusCode::OK;
}

Status DLNode::fetchResults(BlobMap& outputs) {
    // ::execute needs to be executed before ::fetchResults
    if (this->model == nullptr) {
        SPDLOG_INFO("Calling DLNode::fetchResults failed because execution failed (Node: {})", getName());
        return StatusCode::UNKNOWN_ERROR;
    }

    // Get infer request corresponding to this node model
    auto streamId = this->nodeStreamIdGuard->tryGetId();
    if (!streamId) {
        SPDLOG_ERROR("Calling DLNode::fetchResults failed - node had stream Id never assigned (Node: {})", getName());
        return StatusCode::UNKNOWN_ERROR;
    }
    auto& infer_request = this->model->getInferRequestsQueue().getInferRequest(streamId.value());
    // Wait for blob results
    SPDLOG_DEBUG("Waiting for infer request with streamId:{} to finish", streamId.value());
    auto ov_status = infer_request.Wait(InferenceEngine::IInferRequest::RESULT_READY);
    SPDLOG_DEBUG("Infer request with streamId:{} finished", streamId.value());
    this->inputBlobs.clear();
    if (ov_status != InferenceEngine::StatusCode::OK) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_ERROR("Async infer failed: {}; OV StatusCode: {}", status.string(), ov_status);
        return status;
    }

    // Fill outputs map with result blobs. Fetch only those that are required in following nodes.
    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& output_name = pair.first;
            // Multiple next nodes can have the same dependency, do not prepare the same blob multiple times
            // TODO what if different following nodes expect different outputs but under the same name?
            if (outputs.count(output_name) == 1) {
                continue;
            }

            try {
                auto aliasItr = nodeOutputNameAlias.find(output_name);
                const std::string realModelOutputName = ((aliasItr != nodeOutputNameAlias.end()) ? (*aliasItr).second : output_name);
                SPDLOG_DEBUG("Getting blob from model:{}, inferRequestStreamId:{}, blobName:{}", modelName, streamId.value(), realModelOutputName);
                outputs[output_name] = infer_request.GetBlob(realModelOutputName);
            } catch (const InferenceEngine::details::InferenceEngineException& e) {
                Status status = StatusCode::OV_INTERNAL_SERIALIZATION_ERROR;
                SPDLOG_ERROR("DLNode::fetchResults (Node name {}); error during InferRequest::GetBlob: {}; exception message: {}", getName(), status.string(), e.what());
                return status;
            }
            SPDLOG_DEBUG("DLNode::fetchResults (Node name {}): blob with name [{}] has been prepared", getName(), output_name);
        }
    }
    // After results are fetched, model and inference request are not needed anymore
    this->nodeStreamIdGuard.reset();
    this->model.reset();
    this->modelUnloadGuard.reset();
    return StatusCode::OK;
}

Status DLNode::validate(const InferenceEngine::Blob::Ptr& blob, const TensorInfo& info) {
    if (info.getPrecision() != blob->getTensorDesc().getPrecision()) {
        return StatusCode::INVALID_PRECISION;
    }

    // If batch size differes, check if remaining dimensions are equal
    if (info.getShape()[0] != blob->getTensorDesc().getDims()[0]) {
        // If remaining dimensions are equal, it is invalid batch size
        if (std::equal(info.getShape().begin() + 1, info.getShape().end(), blob->getTensorDesc().getDims().begin() + 1)) {
            return StatusCode::INVALID_BATCH_SIZE;
        } else {
            // Otherwise whole shape is incorrect
            return StatusCode::INVALID_SHAPE;
        }
    }

    if (info.getShape() != blob->getTensorDesc().getDims()) {
        return StatusCode::INVALID_SHAPE;
    }

    return StatusCode::OK;
}

Status DLNode::prepareInputsAndModelForInference() {
    size_t requestedBatchSize = 0;
    std::map<std::string, shape_t> requestedReshapes;

    // Validate each blob against its OV tensor info
    const auto& inputsInfo = this->model->getInputsInfo();
    for (const auto& kv : this->inputBlobs) {
        const auto& name = kv.first;
        auto& blob = kv.second;

        if (inputsInfo.count(name) == 0) {
            return StatusCode::INVALID_MISSING_INPUT;
        }
        auto& inputInfo = *inputsInfo.at(name);
        auto status = validate(blob, inputInfo);
        if (status.ok()) {
            continue;
        }

        // If precision is incorrect, perform conversion
        if (status == StatusCode::INVALID_PRECISION) {
            // TODO: Create new blob with proper precision
            // https://jira.devtools.intel.com/browse/CVS-35616
            return status;
        }

        // If batch size is incorrect, perform network batch size change if allowed (shape mode=auto or batch size=auto)
        if (status == StatusCode::INVALID_BATCH_SIZE) {
            if (this->model->getModelConfig().getBatchingMode() == Mode::AUTO) {
                requestedBatchSize = blob->getTensorDesc().getDims()[0];
            } else if (this->model->getModelConfig().isShapeAuto(name)) {
                requestedReshapes[name] = blob->getTensorDesc().getDims();
            } else {
                return status;
            }
        }

        // If shape is incorrect, perform reshape if allowed (mode=auto)
        if (status == StatusCode::INVALID_SHAPE) {
            if (!this->model->getModelConfig().isShapeAuto(name)) {
                return status;
            }
            requestedReshapes[name] = blob->getTensorDesc().getDims();
        }
    }
    if (requestedReshapes.size() > 0) {
        auto status = this->model->reloadModel(0, requestedReshapes, this->modelUnloadGuard);
        if (!status.ok()) {
            return status;
        }
    } else if (requestedBatchSize > 0) {
        auto status = this->model->reloadModel(requestedBatchSize, {}, this->modelUnloadGuard);
        if (!status.ok()) {
            return status;
        }
    }
    return StatusCode::OK;
}

}  // namespace ovms
