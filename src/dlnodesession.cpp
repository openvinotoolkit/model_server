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

#include "dlnodesession.hpp"

#include <map>
#include <string>

#include "logging.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "nodeinputhandler.hpp"
#include "nodeoutputhandler.hpp"
#include "nodestreamidguard.hpp"
#include "ov_utils.hpp"
#include "tensorinfo.hpp"
#include "timer.hpp"

namespace ovms {
DLNodeSession::DLNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ModelManager& manager, const std::string& modelName, model_version_t modelVersion) :
    NodeSession(metadata, nodeName, inputsCount, collapsingDetails),
    modelManager(manager),
    modelName(modelName),
    modelVersion(modelVersion) {}

DLNodeSession::DLNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ModelManager& manager, const std::string& modelName, model_version_t modelVersion) :
    NodeSession(std::move(metadata), nodeName, inputsCount, collapsingDetails),
    modelManager(manager),
    modelName(modelName),
    modelVersion(modelVersion) {}

DLNodeSession::~DLNodeSession() = default;

void DLNodeSession::clearInputs() {
    this->inputHandler->clearInputs();
}

ModelInstance& DLNodeSession::getModelInstance() {
    return *this->model;
}

ov::runtime::InferRequest& DLNodeSession::getInferRequest(const uint microseconds) {
    auto& inferRequestsQueue = this->model->getInferRequestsQueue_2();
    auto streamIdOpt = this->nodeStreamIdGuard->tryGetId(microseconds);
    if (!streamIdOpt) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to get streamId on already executed node: {} session: {}", getName(), getSessionKey());
        throw std::logic_error("Stream id is empty on already executed node");
    }
    return inferRequestsQueue.getInferRequest(streamIdOpt.value());
}

Status DLNodeSession::requestExecuteRequiredResources() {
    Status status = modelManager.getModelInstance(
        modelName,
        modelVersion,
        this->model,
        this->modelUnloadGuard);

    if (!status.ok()) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Getting modelInstance failed for node: {} session: {} with: {}", getName(), getSessionKey(), status.string());
        return status;
    }

    status = prepareInputsAndModelForInference();
    if (!status.ok()) {
        return status;
    }
    this->nodeStreamIdGuard = std::make_unique<NodeStreamIdGuard>(model->getInferRequestsQueue());
    return status;
}

Status DLNodeSession::prepareInputsAndModelForInference() {
    size_t requestedBatchSize = 0;
    std::map<std::string, shape_t> requestedReshapes;

    // Validate each tensor against its OV tensor info
    const auto& inputsInfo = this->model->getInputsInfo();
    for (const auto& kv : this->inputHandler->getInputs()) {
        const auto& name = kv.first;
        auto& tensor = kv.second;

        auto it = inputsInfo.find(name);
        if (it == inputsInfo.end()) {
            std::stringstream ss;
            ss << "Required input: " << name;
            const std::string details = ss.str();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Missing input with specific name - {}", getName(), details);
            return Status(StatusCode::INVALID_MISSING_INPUT, details);
        }
        auto& inputInfo = *it->second;
        auto status = validate(tensor, inputInfo);
        if (status.ok()) {
            continue;
        }

        // If precision is incorrect, perform conversion
        if (status == StatusCode::INVALID_PRECISION) {
            return status;
        }

        // If batch size is incorrect, perform network batch size change if allowed (shape mode=auto or batch size=auto)
        if (status == StatusCode::INVALID_BATCH_SIZE) {
            if (this->model->getModelConfig().getBatchingMode() == Mode::AUTO) {
                requestedBatchSize = tensor->get_shape()[0];
            } else if (this->model->getModelConfig().isShapeAuto(name)) {
                requestedReshapes[name] = tensor->get_shape();
            } else {
                return status;
            }
        }

        // If shape is incorrect, perform reshape if allowed (mode=auto)
        if (status == StatusCode::INVALID_SHAPE) {
            if (!this->model->getModelConfig().isShapeAuto(name)) {
                return status;
            }
            requestedReshapes[name] = tensor->get_shape();
        }
    }
    if (requestedReshapes.size() > 0) {
        size_t bs = 0;
        auto status = this->model->reloadModel(bs, requestedReshapes, this->modelUnloadGuard);
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

Status DLNodeSession::validate(const std::shared_ptr<ov::runtime::Tensor>& tensor, const TensorInfo& tensorInfo) {
    if (ovmsPrecisionToIE2Precision(tensorInfo.getPrecision_2()) != tensor->get_element_type()) {
        std::stringstream ss;
        ss << "Node: " << getName() << " input: " << tensorInfo.getName()
           << " Invalid precision -"
           << " Expected: " << tensorInfo.getPrecisionAsString()
           << "; Actual: " << toString(ovElementTypeToOvmsPrecision(tensor->get_element_type()));
        const std::string details = ss.str();
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }

    // If batch size differs, check if remaining dimensions are equal
    const auto& dims = tensor->get_shape();
    if (tensorInfo.getEffectiveShape()[0] != dims[0]) {
        // If remaining dimensions are equal, it is invalid batch size
        std::stringstream ss;
        if (std::equal(tensorInfo.getEffectiveShape().begin() + 1, tensorInfo.getEffectiveShape().end(), dims.begin() + 1)) {
            ss << "Node: " << getName() << " input: " << tensorInfo.getName()
               << " Invalid batch size -"
               << " Expected: " << tensorInfo.getEffectiveShape()[0]
               << "; Actual: " << dims[0];
            const std::string details = ss.str();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
            return Status(StatusCode::INVALID_BATCH_SIZE, details);
        } else {
            // Otherwise whole shape is incorrect
            ss << "Node: " << getName() << " input: " << tensorInfo.getName()
               << " Invalid shape -"
               << " Expected: " << TensorInfo::shapeToString(tensorInfo.getEffectiveShape())
               << "; Actual: " << TensorInfo::shapeToString(dims);
            const std::string details = ss.str();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }

    if (tensorInfo.getEffectiveShape() != dims) {
        std::stringstream ss;
        ss << "Node: " << getName() << " input: " << tensorInfo.getName()
           << " Invalid shape -"
           << " Expected: " << TensorInfo::shapeToString(tensorInfo.getEffectiveShape())
           << "; Actual: " << TensorInfo::shapeToString(dims);
        const std::string details = ss.str();
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }

    return StatusCode::OK;
}

Status DLNodeSession::execute(PipelineEventQueue& notifyEndQueue, uint waitForStreamIdTimeoutMicroseconds, Node& node) {
    Status status;
    if (this->nodeStreamIdGuard == nullptr) {
        status = requestExecuteRequiredResources();
        if (!status.ok()) {
            notifyEndQueue.push({node, getSessionKey()});
            return status;
        }
    }
    auto streamIdOpt = this->nodeStreamIdGuard->tryGetId(waitForStreamIdTimeoutMicroseconds);
    if (!streamIdOpt) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Could not acquire stream Id right away", getName());
        return StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET;
    }
    auto& inferRequestsQueue = this->model->getInferRequestsQueue_2();
    auto& inferRequest = inferRequestsQueue.getInferRequest(streamIdOpt.value());
    status = setInputsForInference(inferRequest);
    if (!status.ok()) {
        notifyEndQueue.push({node, getSessionKey()});
        return status;
    }
    status = executeInference(notifyEndQueue, inferRequest, node);
    if (!status.ok()) {
        notifyEndQueue.push({node, getSessionKey()});
        return status;
    }
    return status;
}

Status DLNodeSession::getRealInputName(const std::string& alias, std::string* result) const {
    auto it = this->model->getInputsInfo().find(alias);
    if (it == this->model->getInputsInfo().end()) {
        return StatusCode::INVALID_MISSING_INPUT;
    }
    *result = it->second->getName();
    return StatusCode::OK;
}

Status DLNodeSession::setInputsForInference(ov::runtime::InferRequest& inferRequest) {
    Status status = StatusCode::OK;
    try {
        // Prepare inference request, fill with input blobs
        for (const auto& [name, tensor] : this->inputHandler->getInputs()) {
            std::string realModelInputName;
            if (!getRealInputName(name, &realModelInputName).ok()) {
                SPDLOG_LOGGER_WARN(dag_executor_logger, "DLNode::{} [Node name: {}]; cannot find real model input name for alias: {}",
                    __FUNCTION__, getName(), name);
                return StatusCode::INTERNAL_ERROR;
            }
            inferRequest.set_tensor(realModelInputName, *tensor);
        }
        // OV implementation the InferenceEngine::Exception is not
        // a base class for all other exceptions thrown from OV.
        // OV can throw exceptions derived from std::logic_error.
    } catch (const ov::Exception& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] {}; exception message: {}", getName(), status.string(), e.what());
    } catch (std::logic_error& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] {}; exception message: {}", getName(), status.string(), e.what());
    } catch (...) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] {}; with unknown exception", getName(), status.string());
    }
    return status;
}

Status DLNodeSession::executeInference(PipelineEventQueue& notifyEndQueue, ov::runtime::InferRequest& inferRequest, Node& node) {
    try {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Setting completion callback for node name: {}", this->getName());
        inferRequest.set_callback([this, &notifyEndQueue, &inferRequest, &node](std::exception_ptr exception_ptr) {
            this->timer->stop("inference");
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Completion callback received for node name: {}", this->getName());
            // After inference is completed, input blobs are not needed anymore
            this->inputHandler->clearInputs();
            notifyEndQueue.push({node, getSessionKey()});
            inferRequest.set_callback([](std::exception_ptr exception_ptr) {});  // reset callback on infer request
        });
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Starting infer async for node name: {}", getName());
        this->timer->start("inference");
        inferRequest.start_async();
    } catch (const ov::Exception& e) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Exception occured when starting async inference or setting completion callback on model: {}, error: {}",
            getName(), getModelName(), e.what());
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Exception occured when starting async inference or setting completion callback on  model: {}, error: {}",
            getName(), getModelName(), e.what());
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    } catch (...) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "[Node: {}] Unknown exception occured when starting async inference or setting completion callback on model: {}",
            getName(), getModelName());
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    }
    return StatusCode::OK;
}

void DLNodeSession::release() {
    this->nodeStreamIdGuard.reset();
    this->model.reset();
    this->modelUnloadGuard.reset();
}

bool DLNodeSession::tryDisarm(uint microseconds) {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Trying to disarm stream id guard of node: {}", getName());
    if (this->nodeStreamIdGuard == nullptr) {
        return true;
    }
    return this->nodeStreamIdGuard->tryDisarm(microseconds);
}
}  // namespace ovms
