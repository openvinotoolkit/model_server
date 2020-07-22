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

#include <spdlog/spdlog.h>

#include "executinstreamidguard.hpp"
#include "prediction_service_utils.hpp"

namespace ovms {

Status DLNode::execute() {
    // Start inference asynchronously
    auto status = getModelInstance(
        this->modelManager,
        this->modelName,
        this->modelVersion.value_or(0),
        this->model,
        this->modelUnloadGuard);

    if (!status.ok()) {
        SPDLOG_INFO("Getting modelInstance failed. {}", status.string());
        return status;
    }

    // TODO: If precision does not match, create new blob with proper precision

    // Validate each blob against its OV tensor info
    const auto& inputsInfo = this->model->getInputsInfo();
    for (const auto& kv : this->inputBlobs) {
        const auto& name = kv.first;
        auto& blob = kv.second;

        if (inputsInfo.count(name) == 0) {
            return StatusCode::INVALID_MISSING_INPUT;
        }

        status = validate(blob, *inputsInfo.at(name));
        if (!status.ok()) {
            return status;
        }
    }

    // Acquire infer request from pool
    auto& ir_queue = this->model->getInferRequestsQueue();
    this->streamIdGuard = std::make_unique<ExecutingStreamIdGuard>(ir_queue);
    auto& infer_request = ir_queue.getInferRequest(this->streamIdGuard->getId());

    // Prepare inference request, fill with input blobs
    for (const auto& kv : this->inputBlobs) {
        infer_request.SetBlob(kv.first, kv.second);
    }

    infer_request.SetCompletionCallback([this]() {
        SPDLOG_INFO("Completion callback received for node name: {}", this->getName());
        // After inference is completed, input blobs are not needed anymore
        this->inputBlobs.clear();
        // TODO: queue.push(this);
    });

    SPDLOG_INFO("Starting infer async for node name: {}", getName());
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

    // Fill outputs map with result blobs. Fetch only those that are required in following nodes.
    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& output_name = pair.first;
            outputs[output_name] = infer_request.GetBlob(output_name);
            SPDLOG_ERROR("DLNode::fetchResults (Node name {}): blob with name [{}] has been prepared", getName(), output_name);
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

    if (info.getShape() != blob->getTensorDesc().getDims()) {
        return StatusCode::INVALID_SHAPE;
    }

    return StatusCode::OK;
}

}  // namespace ovms
