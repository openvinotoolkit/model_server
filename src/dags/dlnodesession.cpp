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

#include "../logging.hpp"
#include "../modelinstance.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../modelmanager.hpp"
#include "../ov_utils.hpp"
#include "../profiler.hpp"
#include "../shape.hpp"
#include "../status.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "nodeinputhandler.hpp"
#include "nodeoutputhandler.hpp"
#include "nodestreamidguard.hpp"

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

ov::InferRequest& DLNodeSession::getInferRequest(const uint microseconds) {
    auto& inferRequestsQueue = this->model->getInferRequestsQueue();
    auto streamIdOpt = this->nodeStreamIdGuard->tryGetId(microseconds);
    if (!streamIdOpt) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to get streamId on already executed node: {} model: {} session: {}", getName(), getModelName(), getSessionKey());
        throw std::logic_error("Stream id is empty on already executed node");
    }
    return inferRequestsQueue.getInferRequest(streamIdOpt.value());
}

Status DLNodeSession::requestExecuteRequiredResources() {
    OVMS_PROFILE_FUNCTION();
    Status status = modelManager.getModelInstance(
        this->getModelName(),
        modelVersion,
        this->model,
        this->modelUnloadGuard);

    if (!status.ok()) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Getting model: {} instance failed for node: {} session: {} with: {}", getModelName(), getName(), getSessionKey(), status.string());
        return status;
    }

    status = prepareInputsAndModelForInference();
    if (!status.ok()) {
        return status;
    }
    this->timer->start(GET_INFER_REQUEST);
    this->nodeStreamIdGuard = std::make_unique<NodeStreamIdGuard>(model->getInferRequestsQueue(), model->getMetricReporter());
    return status;
}

Status DLNodeSession::prepareInputsAndModelForInference() {
    OVMS_PROFILE_FUNCTION();
    // Validate each tensor against its OV tensor info
    const auto& inputsInfo = this->model->getInputsInfo();
    Status status;
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
        status = validate(tensor, inputInfo);
        if (!status.ok()) {
            break;
        }
    }
    return status;
}

Status DLNodeSession::validate(const ov::Tensor& tensor, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    if (ovmsPrecisionToIE2Precision(tensorInfo.getPrecision()) != tensor.get_element_type()) {
        std::stringstream ss;
        ss << "Node: " << getName() << " input: " << tensorInfo.getName()
           << " Invalid precision -"
           << " Expected: " << tensorInfo.getPrecisionAsString()
           << "; Actual: " << toString(ovElementTypeToOvmsPrecision(tensor.get_element_type()));
        const std::string details = ss.str();
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }

    // If batch size differs, check if remaining dimensions are equal
    const auto& dims = tensor.get_shape();
    const auto batchIndex = tensorInfo.getLayout().getBatchIndex();
    if (!batchIndex.has_value() || batchIndex.value() >= tensorInfo.getShape().size() || batchIndex.value() >= dims.size()) {
        std::stringstream ss;
        ss << "Node: " << getName() << " input: " << tensorInfo.getName()
           << " Invalid batch size index";
        const std::string details = ss.str();
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
        return Status(StatusCode::INVALID_BATCH_DIMENSION, details);
    }
    if (!tensorInfo.getShape()[batchIndex.value()].match(dims[batchIndex.value()])) {
        // If remaining dimensions are equal, it is invalid batch size
        std::stringstream ss;
        if (tensorInfo.getShape().match(dims, batchIndex.value())) {
            ss << "Node: " << getName() << " input: " << tensorInfo.getName()
               << " Invalid batch size -"
               << " Expected: " << tensorInfo.getShape()[batchIndex.value()].toString()
               << "; Actual: " << dims[batchIndex.value()];
            const std::string details = ss.str();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
            return Status(StatusCode::INVALID_BATCH_SIZE, details);
        } else {
            // Otherwise whole shape is incorrect
            ss << "Node: " << getName() << " input: " << tensorInfo.getName()
               << " Invalid shape -"
               << " Expected: " << tensorInfo.getShape().toString()
               << "; Actual: " << shapeToString(dims);
            const std::string details = ss.str();
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }

    if (!tensorInfo.getShape().match(dims)) {
        std::stringstream ss;
        ss << "Node: " << getName() << " input: " << tensorInfo.getName()
           << " Invalid shape -"
           << " Expected: " << tensorInfo.getShape().toString()
           << "; Actual: " << shapeToString(dims);
        const std::string details = ss.str();
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, details);
        return Status(StatusCode::INVALID_SHAPE, details);
    }

    return StatusCode::OK;
}

Status DLNodeSession::execute(PipelineEventQueue& notifyEndQueue, uint waitForStreamIdTimeoutMicroseconds, Node& node) {
    OVMS_PROFILE_FUNCTION();
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
    auto& inferRequestsQueue = this->model->getInferRequestsQueue();
    auto& inferRequest = inferRequestsQueue.getInferRequest(streamIdOpt.value());
    this->timer->stop(GET_INFER_REQUEST);
    double getInferRequestTime = this->timer->elapsed<std::chrono::microseconds>(GET_INFER_REQUEST);
    OBSERVE_IF_ENABLED(this->model->getMetricReporter().waitForInferReqTime, getInferRequestTime);
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

Status DLNodeSession::setInputsForInference(ov::InferRequest& inferRequest) {
    OVMS_PROFILE_FUNCTION();
    Status status = StatusCode::OK;
    try {
        // Prepare inference request, fill with input tensors
        for (const auto& [name, tensor] : this->inputHandler->getInputs()) {
            std::string realModelInputName;
            if (!getRealInputName(name, &realModelInputName).ok()) {
                SPDLOG_LOGGER_WARN(dag_executor_logger, "DLNode::{} [Node name: {}]; cannot find real model:{} input name for alias: {}",
                    __FUNCTION__, getName(), getModelName(), name);
                return StatusCode::INTERNAL_ERROR;
            }
            OVMS_PROFILE_SCOPE("ov::InferRequest::set_tensor");
            inferRequest.set_tensor(realModelInputName, tensor);
        }
        // OV implementation the ov::Exception is not
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

Status DLNodeSession::executeInference(PipelineEventQueue& notifyEndQueue, ov::InferRequest& inferRequest, Node& node) {
    OVMS_PROFILE_FUNCTION();
    try {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Setting completion callback for node name: {}", this->getName());
        inferRequest.set_callback([this, &notifyEndQueue, &inferRequest, &node](std::exception_ptr exception_ptr) {
            OVMS_PROFILE_ASYNC_END("async inference", this);
            this->timer->stop(EXECUTE);
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Completion callback received for node name: {}", this->getName());
            // After inference is completed, input tensors are not needed anymore
            this->inputHandler->clearInputs();
            notifyEndQueue.push({node, getSessionKey()});
            inferRequest.set_callback([](std::exception_ptr exception_ptr) {});  // reset callback on infer request
        });
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Starting infer async for node name: {}", getName());
        this->timer->start(EXECUTE);
        OVMS_PROFILE_SYNC_BEGIN("ov::InferRequest::start_async");
        inferRequest.start_async();
        OVMS_PROFILE_SYNC_END("ov::InferRequest::start_async");
        OVMS_PROFILE_ASYNC_BEGIN("async inference", this);
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
