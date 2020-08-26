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

#include <spdlog/spdlog.h>

#include "executinstreamidguard.hpp"
#include "modelmanager.hpp"
#include "prediction_service_utils.hpp"

namespace ovms {

Status DLNode::execute(ThreadSafeQueue<std::reference_wrapper<Node>>& notifyEndQueue) {
    // Start inference asynchronously
    auto status = getModelInstance(
        this->modelManager,
        this->modelName,
        this->modelVersion.value_or(0),
        this->model,
        this->modelUnloadGuard);

    if (!status.ok()) {
        SPDLOG_INFO("Getting modelInstance failed. {}", status.string());
        notifyEndQueue.push(*this);
        return status;
    }

    status = prepareInputsAndModelForInference();
    if (!status.ok()) {
        SPDLOG_DEBUG("DLNode::execute (Node name {}); error occurred during input/model preparation: {}", status.string());
        notifyEndQueue.push(*this);
        return status;
    }

    // Acquire infer request from pool
    auto& ir_queue = this->model->getInferRequestsQueue();
    this->streamIdGuard = std::make_unique<ExecutingStreamIdGuard>(ir_queue);
    auto& infer_request = ir_queue.getInferRequest(this->streamIdGuard->getId());

    try {
        // Prepare inference request, fill with input blobs
        for (const auto& kv : this->inputBlobs) {
            infer_request.SetBlob(kv.first, kv.second);
        }
        // OV implementation the InferenceEngineException is not
        // a base class for all other exceptions thrown from OV.
        // OV can throw exceptions derived from std::logic_error.
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_ERROR("DLNode::execute (Node name {}); error during InferRequest::SetBlob: {}; exception message: {}", getName(), status.string(), e.what());
        notifyEndQueue.push(*this);
        return status;
    } catch (std::logic_error& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_ERROR("DLNode::execute (Node name {}); error during InferRequest::SetBlob: {}; exception message: {}", getName(), status.string(), e.what());
        notifyEndQueue.push(*this);
        return status;
    }

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
        notifyEndQueue.push(*this);
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    } catch (const std::exception& e) {
        SPDLOG_INFO("Exception occured when started async inference or setting completion callback on node:{}, modelName:{}, error:{}",
            getName(), modelName, e.what());
        notifyEndQueue.push(*this);
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    } catch (...) {
        SPDLOG_INFO("Unknown exception occured when started async or setting completion callback inference on node:{}, modelName:{}",
            getName(), modelName);
        notifyEndQueue.push(*this);
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
    auto& infer_request = this->model->getInferRequestsQueue().getInferRequest(this->streamIdGuard->getId());

    // Wait for blob results
    auto ov_status = infer_request.Wait(InferenceEngine::IInferRequest::RESULT_READY);
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
    this->streamIdGuard.reset();
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
