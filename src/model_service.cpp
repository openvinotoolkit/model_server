//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include "model_service.hpp"

#include <memory>
#include <string>
#include <vector>

#include <google/protobuf/util/json_util.h>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#pragma GCC diagnostic pop

#include "dags/pipelinedefinition.hpp"
#include "execution_context.hpp"
#include "grpc_utils.hpp"
#if (MEDIAPIPE_DISABLE == 0)
#include "mediapipe_internal/mediapipefactory.hpp"
#include "mediapipe_internal/mediapipegraphdefinition.hpp"
#endif
#include "modelinstance.hpp"
#include "modelmanager.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "status.hpp"

using google::protobuf::util::JsonPrintOptions;
using google::protobuf::util::MessageToJsonString;

namespace ovms {

void addStatusToResponse(tensorflow::serving::GetModelStatusResponse* response, model_version_t version, const ModelVersionStatus& model_version_status) {
    SPDLOG_DEBUG("add_status_to_response version={} status={}", version, model_version_status.getStateString());
    auto status_to_fill = response->add_model_version_status();
    status_to_fill->set_state(static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(model_version_status.getState())));
    status_to_fill->set_version(version);
    status_to_fill->clear_status();
    status_to_fill->mutable_status()->set_error_code(static_cast<tensorflow::error::Code>(static_cast<int>(model_version_status.getErrorCode())));
    status_to_fill->mutable_status()->set_error_message(model_version_status.getErrorMsg());
}

void addStatusToResponse(tensorflow::serving::GetModelStatusResponse* response, const model_version_t version, const PipelineDefinitionStatus& pipeline_status) {
    auto [state, error_code] = pipeline_status.convertToModelStatus();
    SPDLOG_DEBUG("add_status_to_response state={} error_code", state, error_code);
    auto status_to_fill = response->add_model_version_status();
    status_to_fill->set_state(static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(state)));
    status_to_fill->set_version(version);
    status_to_fill->clear_status();
    status_to_fill->mutable_status()->set_error_code(static_cast<tensorflow::error::Code>(static_cast<int>(error_code)));
    status_to_fill->mutable_status()->set_error_message(ModelVersionStatusErrorCodeToString(error_code));
}

::grpc::Status ModelServiceImpl::GetModelStatus(
    ::grpc::ServerContext* context, const tensorflow::serving::GetModelStatusRequest* request,
    tensorflow::serving::GetModelStatusResponse* response) {
    return grpc(GetModelStatusImpl::getModelStatus(request, response, modelManager, ExecutionContext(ExecutionContext::Interface::GRPC, ExecutionContext::Method::GetModelStatus)));
}

Status GetModelStatusImpl::createGrpcRequest(std::string model_name, const std::optional<int64_t> model_version, tensorflow::serving::GetModelStatusRequest* request) {
    request->mutable_model_spec()->set_name(model_name);
    if (model_version.has_value()) {
        request->mutable_model_spec()->mutable_version()->set_value(model_version.value());
    }
    return StatusCode::OK;
}

Status GetModelStatusImpl::serializeResponse2Json(const tensorflow::serving::GetModelStatusResponse* response, std::string* output) {
    JsonPrintOptions opts;
    opts.add_whitespace = true;
    opts.always_print_primitive_fields = true;
    const auto& status = MessageToJsonString(*response, output, opts);
    if (!status.ok()) {
        SPDLOG_ERROR("Failed to convert proto to json. Error: ", status.ToString());
        return StatusCode::JSON_SERIALIZATION_ERROR;
    }
    return StatusCode::OK;
}

Status GetModelStatusImpl::getModelStatus(
    const tensorflow::serving::GetModelStatusRequest* request,
    tensorflow::serving::GetModelStatusResponse* response,
    ModelManager& manager,
    ExecutionContext context) {
    SPDLOG_DEBUG("model_service: request: {}", request->DebugString());

    bool has_requested_version = request->model_spec().has_version();
    auto requested_version = request->model_spec().version().value();
    std::string requested_model_name = request->model_spec().name();
    auto model_ptr = manager.findModelByName(requested_model_name);
    if (!model_ptr) {
        SPDLOG_DEBUG("GetModelStatus: Model {} is missing, trying to find pipeline with such name", requested_model_name);
        auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(requested_model_name);
        if (!pipelineDefinition) {
#if (MEDIAPIPE_DISABLE == 0)
            auto mediapipeGraphDefinition = manager.getMediapipeFactory().findDefinitionByName(requested_model_name);
            if (!mediapipeGraphDefinition) {
                return StatusCode::MODEL_NAME_MISSING;
            }
            addStatusToResponse(response, mediapipeGraphDefinition->getVersion(), mediapipeGraphDefinition->getStatus());
            SPDLOG_DEBUG("model_service: response: {}", response->DebugString());
            SPDLOG_DEBUG("MODEL_STATUS created a response for {} - {}", requested_model_name, requested_version);
            return StatusCode::OK;
#else
            return StatusCode::MODEL_NAME_MISSING;
#endif
        }
        INCREMENT_IF_ENABLED(pipelineDefinition->getMetricReporter().getGetModelStatusRequestSuccessMetric(context));

        addStatusToResponse(response, pipelineDefinition->getVersion(), pipelineDefinition->getStatus());
        SPDLOG_DEBUG("model_service: response: {}", response->DebugString());
        SPDLOG_DEBUG("MODEL_STATUS created a response for {} - {}", requested_model_name, requested_version);
        return StatusCode::OK;
    }

    SPDLOG_DEBUG("requested model: {}, has_version: {} (version: {})", requested_model_name, has_requested_version, requested_version);
    if (has_requested_version && requested_version != 0) {
        // return details only for a specific version of requested model; NOT_FOUND otherwise. If requested_version == 0, default is returned.
        std::shared_ptr<ModelInstance> modelInstance = model_ptr->getModelInstanceByVersion(requested_version);
        if (!modelInstance) {
            SPDLOG_DEBUG("requested model {} in version {} was not found.", requested_model_name, requested_version);
            return StatusCode::MODEL_VERSION_MISSING;
        }
        INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().getGetModelStatusRequestSuccessMetric(context));
        const auto& status = modelInstance->getStatus();
        SPDLOG_DEBUG("adding model {} - {} :: {} to response", requested_model_name, requested_version, status.getStateString());
        addStatusToResponse(response, requested_version, status);
    } else {
        // return status details of all versions of a requested model.
        auto modelVersionsInstances = model_ptr->getModelVersionsMapCopy();
        bool reported = false;
        for (const auto& [modelVersion, modelInstance] : modelVersionsInstances) {
            // GetModelStatus is tracked once for all versions (no version label) - this is why we only report once in a loop (each model instance metric refer to the same metric)
            if (!reported) {
                INCREMENT_IF_ENABLED(modelInstance.getMetricReporter().getGetModelStatusRequestSuccessMetric(context));
                reported = true;
            }
            const auto& status = modelInstance.getStatus();
            SPDLOG_DEBUG("adding model {} - {} :: {} to response", requested_model_name, modelVersion, status.getStateString());
            addStatusToResponse(response, modelVersion, status);
        }
    }
    SPDLOG_DEBUG("model_service: response: {}", response->DebugString());
    SPDLOG_DEBUG("MODEL_STATUS created a response for {} - {}", requested_model_name, requested_version);
    return StatusCode::OK;
}

Status GetModelStatusImpl::getAllModelsStatuses(std::map<std::string, tensorflow::serving::GetModelStatusResponse>& modelsStatuses, ModelManager& manager, ExecutionContext context) {
    std::shared_lock lock(manager.modelsMtx);
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatusesTmp;

    const std::map<std::string, std::shared_ptr<Model>>& models = manager.getModels();
    for (auto const& model : models) {
        std::optional<int64_t> noValueModelVersion;
        tensorflow::serving::GetModelStatusRequest request;
        GetModelStatusImpl::createGrpcRequest(model.first, noValueModelVersion, &request);
        tensorflow::serving::GetModelStatusResponse response;
        auto status = GetModelStatusImpl::getModelStatus(&request, &response, manager, context);
        if (status != StatusCode::OK) {
            // For now situation when getModelStatus return status other than OK cannot occur because we never remove models and pipelines from model manager.
            // However, if something in this matter will change we should handle this somehow.
            continue;
        }
        modelsStatusesTmp.insert({model.first, response});
    }
    lock.unlock();

    const std::vector<std::string>& pipelinesNames = manager.getPipelineFactory().getPipelinesNames();
    for (auto const& pipelineName : pipelinesNames) {
        std::optional<int64_t> noValueModelVersion;
        tensorflow::serving::GetModelStatusRequest request;
        GetModelStatusImpl::createGrpcRequest(pipelineName, noValueModelVersion, &request);
        tensorflow::serving::GetModelStatusResponse response;
        auto status = GetModelStatusImpl::getModelStatus(&request, &response, manager, context);
        if (status != StatusCode::OK) {
            // Same situation like with models.
            continue;
        }
        modelsStatusesTmp.insert({pipelineName, response});
    }
#if (MEDIAPIPE_DISABLE == 0)
    const std::vector<std::string>& mediapipePipelineNames = manager.getMediapipeFactory().getMediapipePipelinesNames();
    for (auto const& mediapipePipelineName : mediapipePipelineNames) {
        std::optional<int64_t> noValueModelVersion;
        tensorflow::serving::GetModelStatusRequest request;
        GetModelStatusImpl::createGrpcRequest(mediapipePipelineName, noValueModelVersion, &request);
        tensorflow::serving::GetModelStatusResponse response;
        auto status = GetModelStatusImpl::getModelStatus(&request, &response, manager, context);
        if (status != StatusCode::OK) {
            // Same situation like with models.
            continue;
        }
        modelsStatusesTmp.insert({mediapipePipelineName, response});
    }
#endif

    modelsStatuses.merge(modelsStatusesTmp);
    return StatusCode::OK;
}

Status GetModelStatusImpl::serializeModelsStatuses2Json(const std::map<std::string, tensorflow::serving::GetModelStatusResponse>& modelsStatuses, std::string& output) {
    std::string outputTmp;
    if (modelsStatuses.begin() == modelsStatuses.end()) {
        output = "{}";
        return StatusCode::OK;
    }

    outputTmp += "{\n";
    for (auto modelStatus = modelsStatuses.begin(); modelStatus != modelsStatuses.end(); modelStatus++) {
        outputTmp += ("\"" + modelStatus->first + "\" : \n");
        std::string responseStr;
        auto status = GetModelStatusImpl::serializeResponse2Json(&modelStatus->second, &responseStr);
        if (status != StatusCode::OK) {
            return status;
        }
        responseStr.pop_back();
        outputTmp += responseStr;
        if (std::next(modelStatus) != modelsStatuses.end()) {
            outputTmp += (",\n");
        }
    }
    outputTmp += "\n}";
    output = outputTmp;

    return StatusCode::OK;
}

ModelServiceImpl::ModelServiceImpl(ovms::Server& ovmsServer) :
    modelManager(dynamic_cast<const ServableManagerModule*>(ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager()) {
    if (nullptr == ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME)) {
        const char* message = "Tried to create model service impl without servable manager module";
        SPDLOG_ERROR(message);
        throw std::logic_error(message);
    }
}

::grpc::Status ModelServiceImpl::HandleReloadConfigRequest(
    ::grpc::ServerContext* context, const tensorflow::serving::ReloadConfigRequest* request,
    tensorflow::serving::ReloadConfigResponse* response) {
    SPDLOG_INFO("Requested HandleReloadConfigRequest - but this service is reloading config automatically by itself, therefore this operation has no *EXTRA* affect.");
    return grpc::Status::OK;  // we're reloading config all the time; for a total client compatibility, this means returning success here.
}

}  // namespace ovms
