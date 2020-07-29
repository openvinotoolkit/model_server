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
#include <map>

#include <spdlog/spdlog.h>

#include "dl_node.hpp"
#include "executinstreamidguard.hpp"
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
        notifyEndQueue.push(*this);
        return status;
    }

    // Acquire infer request from pool
    auto& ir_queue = this->model->getInferRequestsQueue();
    this->streamIdGuard = std::make_unique<ExecutingStreamIdGuard>(ir_queue);
    auto& infer_request = ir_queue.getInferRequest(this->streamIdGuard->getId());

    // Prepare inference request, fill with input blobs
    for (const auto& kv : this->inputBlobs) {
        infer_request.SetBlob(kv.first, kv.second);
    }
    SPDLOG_DEBUG("Setting completion callback for node name: {}", this->getName());
    infer_request.SetCompletionCallback([this, &notifyEndQueue]() {
        SPDLOG_DEBUG("Completion callback received for node name: {}", this->getName());
        // After inference is completed, input blobs are not needed anymore
        this->inputBlobs.clear();
        notifyEndQueue.push(*this);
    });

    SPDLOG_DEBUG("Starting infer async for node name: {}", getName());
    infer_request.StartAsync();

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
    infer_request.Wait(InferenceEngine::IInferRequest::RESULT_READY);

    // Fill outputs map with result blobs. Fetch only those that are required in following nodes.
    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& output_name = pair.first;
            outputs[output_name] = infer_request.GetBlob(output_name);
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

    // If batch size differes, check if remaining dimensionsa re equal
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
            return status;
        }

        // If batch size is incorrect, perform network batch size change if allowed (mode=auto)
        if (status == StatusCode::INVALID_BATCH_SIZE) {
            if (this->model->getModelConfig().getBatchingMode() != Mode::AUTO) {
                return status;
            }
            requestedBatchSize = blob->getTensorDesc().getDims()[0];
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
