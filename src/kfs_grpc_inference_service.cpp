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
#include "metric.hpp"
#include "modelinstance.hpp"
#include "modelmanager.hpp"
#include "ovinferrequestsqueue.hpp"
#include "pipeline.hpp"
#include "pipelinedefinition.hpp"
#include "pipelinedefinitionstatus.hpp"
#include "pipelinedefinitionunloadguard.hpp"
#include "prediction_service_utils.hpp"
#include "serialization.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"
#include "timer.hpp"
#include "version.hpp"

namespace {
enum : unsigned int {
    TOTAL,
    TIMER_END
};
}

namespace ovms {

Status KFSInferenceServiceImpl::getModelInstance(const ::inference::ModelInferRequest* request,
    std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
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
    return this->modelManager.getModelInstance(request->model_name(), requestedVersion, modelInstance, modelInstanceUnloadGuardPtr);
}

Status KFSInferenceServiceImpl::getPipeline(const ::inference::ModelInferRequest* request,
    ::inference::ModelInferResponse* response,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr) {
    OVMS_PROFILE_FUNCTION();
    return this->modelManager.createPipeline(pipelinePtr, request->model_name(), request, response);
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

Status KFSInferenceServiceImpl::getModelReady(const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response, const ModelManager& manager, ExecutionContext executionContext) {
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
        auto status = buildResponse(*pipelineDefinition, response);
        INCREMENT_IF_ENABLED(pipelineDefinition->getMetricReporter().getModelReadyMetric(executionContext, status.ok()));
        return status;
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
    auto status = buildResponse(instance, response);
    INCREMENT_IF_ENABLED(instance->getMetricReporter().getModelReadyMetric(executionContext, status.ok()));
    return status;
}

::grpc::Status KFSInferenceServiceImpl::ModelReady(::grpc::ServerContext* context, const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response) {
    return ModelReadyImpl(context, request, response, ExecutionContext{ExecutionContext::Interface::GRPC, ExecutionContext::Method::ModelReady}).grpc();
}

Status KFSInferenceServiceImpl::ModelReadyImpl(::grpc::ServerContext* context, const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response, ExecutionContext executionContext) {
    (void)context;
    return this->getModelReady(request, response, this->modelManager, executionContext);
}

::grpc::Status KFSInferenceServiceImpl::ServerMetadata(::grpc::ServerContext* context, const ::inference::ServerMetadataRequest* request, ::inference::ServerMetadataResponse* response) {
    return ServerMetadataImpl(context, request, response).grpc();
}

Status KFSInferenceServiceImpl::ServerMetadataImpl(::grpc::ServerContext* context, const ::inference::ServerMetadataRequest* request, ::inference::ServerMetadataResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    response->set_name(PROJECT_NAME);
    response->set_version(PROJECT_VERSION);
    return StatusCode::OK;
}

::grpc::Status KFSInferenceServiceImpl::ModelMetadata(::grpc::ServerContext* context, const ::inference::ModelMetadataRequest* request, ::inference::ModelMetadataResponse* response) {
    return ModelMetadataImpl(context, request, response, ExecutionContext{ExecutionContext::Interface::GRPC, ExecutionContext::Method::ModelMetadata}).grpc();
}

Status KFSInferenceServiceImpl::ModelMetadataImpl(::grpc::ServerContext* context, const ::inference::ModelMetadataRequest* request, ::inference::ModelMetadataResponse* response, ExecutionContext executionContext) {
    const auto& name = request->name();
    const auto& versionString = request->version();

    auto model = this->modelManager.findModelByName(name);
    if (model == nullptr) {
        SPDLOG_DEBUG("GetModelMetadata: Model {} is missing, trying to find pipeline with such name", name);
        auto pipelineDefinition = this->modelManager.getPipelineFactory().findDefinitionByName(name);
        if (!pipelineDefinition) {
            return Status(StatusCode::MODEL_NAME_MISSING);
        }
        auto status = buildResponse(*pipelineDefinition, response);
        INCREMENT_IF_ENABLED(pipelineDefinition->getMetricReporter().getModelMetadataMetric(executionContext, status.ok()));
        return status;
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
            return Status(StatusCode::MODEL_VERSION_INVALID_FORMAT);
        }
        instance = model->getModelInstanceByVersion(requestedVersion);
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    } else {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    }
    auto status = buildResponse(*model, *instance, response);
    INCREMENT_IF_ENABLED(instance->getMetricReporter().getModelMetadataMetric(executionContext, status.ok()));

    return status;
}

::grpc::Status KFSInferenceServiceImpl::ModelInfer(::grpc::ServerContext* context, const ::inference::ModelInferRequest* request, ::inference::ModelInferResponse* response) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    SPDLOG_DEBUG("Processing gRPC request for model: {}; version: {}",
        request->model_name(),
        request->model_version());
    ServableMetricReporter* reporter = nullptr;
    auto status = this->ModelInferImpl(context, request, response, ExecutionContext{ExecutionContext::Interface::GRPC, ExecutionContext::Method::ModelInfer}, reporter);
    timer.stop(TOTAL);
    if (!status.ok()) {
        return status.grpc();
    }
    if (!reporter) {
        return Status(StatusCode::INTERNAL_ERROR).grpc();  // should not happen
    }
    double requestTotal = timer.elapsed<std::chrono::microseconds>(TOTAL);
    SPDLOG_DEBUG("Total gRPC request processing time: {} ms", requestTotal / 1000);
    OBSERVE_IF_ENABLED(reporter->requestTimeGrpc, requestTotal);
    return status.grpc();
}

Status KFSInferenceServiceImpl::ModelInferImpl(::grpc::ServerContext* context, const ::inference::ModelInferRequest* request, ::inference::ModelInferResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) {
    OVMS_PROFILE_FUNCTION();
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::Pipeline> pipelinePtr;

    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(request, modelInstance, modelInstanceUnloadGuard);
    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", request->model_name());
        status = getPipeline(request, response, pipelinePtr);
    }
    if (!status.ok()) {
        if (modelInstance) {
            INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().requestFailGrpcModelInfer);
        }
        SPDLOG_DEBUG("Getting modelInstance or pipeline failed. {}", status.string());
        return status;
    }

    if (pipelinePtr) {
        reporterOut = &pipelinePtr->getMetricReporter();
        status = pipelinePtr->execute(executionContext);
    } else {
        reporterOut = &modelInstance->getMetricReporter();
        status = modelInstance->infer(request, response, modelInstanceUnloadGuard);
    }

    INCREMENT_IF_ENABLED(reporterOut->getInferRequestMetric(executionContext, status.ok()));

    if (!status.ok()) {
        return status;
    }

    response->set_id(request->id());
    return StatusCode::OK;
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

static void addReadyVersions(Model& model,
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
    ovmsServer(server),
    modelManager(dynamic_cast<const ServableManagerModule*>(this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager()) {
    if (nullptr == this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME)) {
        const char* message = "Tried to create kserve inference service impl without servable manager module";
        SPDLOG_ERROR(message);
        throw std::logic_error(message);
    }
}

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
