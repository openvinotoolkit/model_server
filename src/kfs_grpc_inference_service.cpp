//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "kfs_grpc_inference_service.hpp"

#include <iostream>
#include <memory>
#include <string>

#include "deserialization.hpp"
#include "modelinstance.hpp"
#include "modelmanager.hpp"
#include "ovinferrequestsqueue.hpp"
#include "pipelinedefinition.hpp"
#include "prediction_service_utils.hpp"
#include "serialization.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "tensorinfo.hpp"
#include "timer.hpp"
#include "version.hpp"

namespace ovms {

Status KFSInferenceServiceImpl::getModelInstance(const ::inference::ModelInferRequest* request,
    std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;  // TODO consider other + add details
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO if not succeed then return error
    auto& manager = servableManagerModule->getServableManager();
    model_version_t requestedVersion = 0;
    if (!request->model_version().empty()) {
        auto versionRead = stoi64(request->model_version());
        if (versionRead) {
            requestedVersion = versionRead.value();
        } else {
            SPDLOG_DEBUG("requested model: name {}; with version in invalid format: {}", request->model_name(), request->model_version());
            return StatusCode::MODEL_VERSION_INVALID_FORMAT;
        }
    }
    return manager.getModelInstance(request->model_name(), requestedVersion, modelInstance, modelInstanceUnloadGuardPtr);
}

Status KFSInferenceServiceImpl::getPipeline(const ::inference::ModelInferRequest* request,
    ::inference::ModelInferResponse* response,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr) {
    OVMS_PROFILE_FUNCTION();
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;  // TODO consider other + add details
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO if not succeed then return error
    auto& manager = servableManagerModule->getServableManager();
    return manager.createPipeline(pipelinePtr, request->model_name(), request, response);
}

const std::string PLATFORM = "OpenVINO";

::grpc::Status KFSInferenceServiceImpl::ServerLive(::grpc::ServerContext* context, const ::inference::ServerLiveRequest* request, ::inference::ServerLiveResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    bool isLive = this->ovmsServer.isLive();
    SPDLOG_DEBUG("Requested Server liveness state: {}", isLive);
    response->set_live(isLive);
    return grpc::Status::OK;
}

::grpc::Status KFSInferenceServiceImpl::ServerReady(::grpc::ServerContext* context, const ::inference::ServerReadyRequest* request, ::inference::ServerReadyResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    bool isReady = this->ovmsServer.isReady();
    SPDLOG_DEBUG("Requested Server readiness state: {}", isReady);
    response->set_ready(isReady);
    return grpc::Status::OK;
}

Status KFSInferenceServiceImpl::getModelReady(const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response, const ModelManager& manager) {
    // Return in response true/false
    // if no version requested give response for default version
    const auto& name = request->name();
    const auto& versionString = request->version();
    auto model = manager.findModelByName(name);
    SPDLOG_DEBUG("ModelReady requested name: {}, version: {}", name, versionString);
    if (model == nullptr) {
        SPDLOG_DEBUG("ModelReady requested model {} is missing, trying to find pipeline with such name", name);
        auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(name);
        if (!pipelineDefinition) {
            return Status(StatusCode::MODEL_NAME_MISSING);
        }
        return buildResponse(*pipelineDefinition, response);
    }
    std::shared_ptr<ModelInstance> instance = nullptr;
    if (!versionString.empty()) {
        SPDLOG_DEBUG("ModelReady requested model: name {}; version {}", name, versionString);
        model_version_t requestedVersion = 0;
        auto versionRead = stoi64(versionString);
        if (versionRead) {
            requestedVersion = versionRead.value();
        } else {
            SPDLOG_DEBUG("ModelReady requested model: name {}; with version in invalid format: {}", name, versionString);
            return Status(StatusCode::MODEL_VERSION_INVALID_FORMAT);
        }
        instance = model->getModelInstanceByVersion(requestedVersion);
        if (instance == nullptr) {
            SPDLOG_DEBUG("ModelReady requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    } else {
        SPDLOG_DEBUG("ModelReady requested model: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_DEBUG("ModelReady requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    }
    return buildResponse(instance, response);
}

::grpc::Status KFSInferenceServiceImpl::ModelReady(::grpc::ServerContext* context, const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response) {
    (void)context;
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, SERVABLE_MANAGER_MODULE_NAME + " module not started yet");
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO if not succeed then return error
    auto& manager = servableManagerModule->getServableManager();
    return this->getModelReady(request, response, manager).grpc();
}

::grpc::Status KFSInferenceServiceImpl::ServerMetadata(::grpc::ServerContext* context, const ::inference::ServerMetadataRequest* request, ::inference::ServerMetadataResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    response->set_name(PROJECT_NAME);
    response->set_version(PROJECT_VERSION);
    return grpc::Status::OK;
}

::grpc::Status KFSInferenceServiceImpl::ModelMetadata(::grpc::ServerContext* context, const ::inference::ModelMetadataRequest* request, ::inference::ModelMetadataResponse* response) {
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, SERVABLE_MANAGER_MODULE_NAME + " module not started yet");
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO if not succeed then return error
    auto& manager = servableManagerModule->getServableManager();
    const auto& name = request->name();
    const auto& versionString = request->version();

    auto model = manager.findModelByName(name);
    if (model == nullptr) {
        SPDLOG_DEBUG("GetModelMetadata: Model {} is missing, trying to find pipeline with such name", name);
        auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(name);
        if (!pipelineDefinition) {
            return Status(StatusCode::MODEL_NAME_MISSING).grpc();
        }
        return buildResponse(*pipelineDefinition, response);
    }
    std::shared_ptr<ModelInstance> instance = nullptr;
    if (!versionString.empty()) {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; version {}", name, versionString);
        model_version_t requestedVersion = 0;
        auto versionRead = stoi64(versionString);
        if (versionRead) {
            requestedVersion = versionRead.value();
        } else {
            SPDLOG_DEBUG("GetModelMetadata requested model: name {}; with version in invalid format: {}", name, versionString);
            return Status(StatusCode::MODEL_VERSION_INVALID_FORMAT).grpc();
        }
        instance = model->getModelInstanceByVersion(requestedVersion);
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING).grpc();
        }
    } else {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING).grpc();
        }
    }
    return buildResponse(*model, *instance, response).grpc();
}

::grpc::Status KFSInferenceServiceImpl::ModelInfer(::grpc::ServerContext* context, const ::inference::ModelInferRequest* request, ::inference::ModelInferResponse* response) {
    (void)context;
    OVMS_PROFILE_FUNCTION();
    Timer timer;
    timer.start("total");
    using std::chrono::microseconds;
    SPDLOG_DEBUG("Processing gRPC request for model: {}; version: {}",
        request->model_name(),
        request->model_version());

    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::Pipeline> pipelinePtr;

    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(request, modelInstance, modelInstanceUnloadGuard);
    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", request->model_name());
        status = getPipeline(request, response, pipelinePtr);
    }
    if (!status.ok()) {
        SPDLOG_DEBUG("Getting modelInstance or pipeline failed. {}", status.string());
        return status.grpc();
    }

    if (pipelinePtr) {
        status = pipelinePtr->execute();
    } else {
        status = modelInstance->infer(request, response, modelInstanceUnloadGuard);
    }

    if (!status.ok()) {
        return status.grpc();
    }

    timer.stop("total");
    SPDLOG_DEBUG("Total gRPC request processing time: {} ms", timer.elapsed<microseconds>("total") / 1000);
    return grpc::Status::OK;
}

Status KFSInferenceServiceImpl::buildResponse(
    std::shared_ptr<ModelInstance> instance,
    ::inference::ModelReadyResponse* response) {
    response->set_ready(instance->getStatus().getState() == ModelVersionState::AVAILABLE);
    return StatusCode::OK;
}

Status KFSInferenceServiceImpl::buildResponse(
    PipelineDefinition& pipelineDefinition,
    ::inference::ModelReadyResponse* response) {
    response->set_ready(pipelineDefinition.getStatus().isAvailable());
    return StatusCode::OK;
}

void addReadyVersions(Model& model,
    ::inference::ModelMetadataResponse* response) {
    auto modelVersions = model.getModelVersionsMapCopy();
    for (auto& [modelVersion, modelInstance] : modelVersions) {
        if (modelInstance.getStatus().getState() == ModelVersionState::AVAILABLE)
            response->add_versions(std::to_string(modelVersion));
    }
}

Status KFSInferenceServiceImpl::buildResponse(
    Model& model,
    ModelInstance& instance,
    ::inference::ModelMetadataResponse* response) {

    std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;

    // 0 meaning immediately return unload guard if possible, otherwise do not wait for available state
    auto status = instance.waitForLoaded(0, unloadGuard);
    if (!status.ok()) {
        return status;
    }

    response->Clear();
    response->set_name(instance.getName());
    addReadyVersions(model, response);
    response->set_platform(PLATFORM);

    for (const auto& input : instance.getInputsInfo()) {
        convert(input, response->add_inputs());
    }

    for (const auto& output : instance.getOutputsInfo()) {
        convert(output, response->add_outputs());
    }

    return StatusCode::OK;
}

KFSInferenceServiceImpl::KFSInferenceServiceImpl(const Server& server) :
    ovmsServer(server) {}

Status KFSInferenceServiceImpl::buildResponse(
    PipelineDefinition& pipelineDefinition,
    ::inference::ModelMetadataResponse* response) {

    std::unique_ptr<PipelineDefinitionUnloadGuard> unloadGuard;

    // 0 meaning immediately return unload guard if possible, otherwise do not wait for available state
    auto status = pipelineDefinition.waitForLoaded(unloadGuard, 0);
    if (!status.ok()) {
        return status;
    }

    response->Clear();
    response->set_name(pipelineDefinition.getName());
    response->add_versions("1");
    response->set_platform(PLATFORM);

    for (const auto& input : pipelineDefinition.getInputsInfo()) {
        convert(input, response->add_inputs());
    }

    for (const auto& output : pipelineDefinition.getOutputsInfo()) {
        convert(output, response->add_outputs());
    }

    return StatusCode::OK;
}

void KFSInferenceServiceImpl::convert(
    const std::pair<std::string, std::shared_ptr<TensorInfo>>& from,
    ::inference::ModelMetadataResponse::TensorMetadata* to) {
    to->set_name(from.first);
    to->set_datatype(from.second->getPrecisionAsKFSPrecision());
    for (auto dim : from.second->getShape()) {
        if (dim.isStatic()) {
            to->add_shape(dim.getStaticValue());
        } else {
            to->add_shape(DYNAMIC_DIMENSION);
        }
    }
}

}  // namespace ovms
