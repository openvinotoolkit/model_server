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

#include <string>
#include <memory>

#include <google/protobuf/util/json_util.h>
#include <spdlog/spdlog.h>
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#include "tensorflow_serving/apis/get_model_status.pb.h"

#include "modelmanager.hpp"
#include "model_service.hpp"
#include "status.hpp"


using google::protobuf::util::MessageToJsonString;
using google::protobuf::util::JsonPrintOptions;

namespace ovms {

void addStatusToResponse(tensorflow::serving::GetModelStatusResponse* response, model_version_t version, const ModelVersionStatus& model_version_status) {
    SPDLOG_DEBUG("add_status_to_response version={} status={}", version, model_version_status.getStateString());
    auto status_to_fill = response->add_model_version_status();
    status_to_fill->set_state(static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(model_version_status.getState())));
    status_to_fill->set_version(version);
}

::grpc::Status ModelServiceImpl::GetModelStatus(
        ::grpc::ServerContext *context, const tensorflow::serving::GetModelStatusRequest *request,
        tensorflow::serving::GetModelStatusResponse *response) {
    return GetModelStatusImpl::getModelStatus(request, response).grpc();
}

Status  GetModelStatusImpl::createGrpcRequest(std::string model_name, const std::optional<int64_t> model_version, tensorflow::serving::GetModelStatusRequest * request ) {
    request->mutable_model_spec()->set_name(model_name);
    if (model_version.has_value()) {
        if (model_version.value() < 0) {
            spdlog::error("Version in GetModelStatus request cannot be negative. Provided value {}", model_version.value());
            return StatusCode::MODEL_VERSION_MISSING;
        }
        request->mutable_model_spec()->mutable_version()->set_value(model_version.value());
    }
    return StatusCode::OK;
}

Status GetModelStatusImpl::serializeResponse2Json(const tensorflow::serving::GetModelStatusResponse * response, std::string * output) {
    JsonPrintOptions opts;
    opts.add_whitespace = true;
    opts.always_print_primitive_fields = true;
    const auto& status = MessageToJsonString(*response, output, opts);
    if (!status.ok()) {
        spdlog::error("Failed to convert proto to json. Error: ", status.ToString());
        return StatusCode::JSON_SERIALIZATION_ERROR;
    }
    return StatusCode::OK;
}

Status GetModelStatusImpl::getModelStatus(const tensorflow::serving::GetModelStatusRequest * request, tensorflow::serving::GetModelStatusResponse * response) {
    SPDLOG_DEBUG("model_service: request: {}", request->DebugString());
    bool has_requested_version = request->model_spec().has_version();
    auto requested_version = request->model_spec().version().value();
    std::string requested_model_name = request->model_spec().name();
    auto model_ptr = ModelManager::getInstance().findModelByName(requested_model_name);
    if (!model_ptr) {
        SPDLOG_INFO("requested model {} was not found", requested_model_name);
        return StatusCode::MODEL_NAME_MISSING;
    }
    SPDLOG_DEBUG("requested model: {}, has_version: {} (version: {})", requested_model_name, has_requested_version, requested_version);
    if (has_requested_version || requested_version != 0) {
        // return details only for a specific version of requested model; NOT_FOUND otherwise. If requested_version == 0, default is returned.
        std::shared_ptr<ModelInstance> model_instance = model_ptr->getModelInstanceByVersion(requested_version);
        if (!model_instance) {
            SPDLOG_INFO("requested model {} in version {} was not found.", requested_model_name, requested_version);
            return StatusCode::MODEL_VERSION_MISSING;
        }
        const auto& status = model_instance->getStatus();
        SPDLOG_DEBUG("adding model {} - {} :: {} to response", requested_model_name, requested_version, status.getStateString());
        addStatusToResponse(response, requested_version, status);
    } else {
        // return status details of all versions of a requested model.
        auto model_versions = model_ptr->getModelVersions();
        for (const auto& [model_version, model_instance_ptr] : model_versions) {
            if (!model_instance_ptr) {
                spdlog::error("during model iteration, found null model instance pointer!");
                return StatusCode::MODEL_VERSION_MISSING;
            }
            const auto& status = model_instance_ptr->getStatus();
            SPDLOG_DEBUG("adding model {} - {} :: {} to response", requested_model_name, model_version, status.getStateString());
            addStatusToResponse(response, model_version, status);
        }
    }
    SPDLOG_DEBUG("model_service: response: {}", response->DebugString());
    SPDLOG_DEBUG("MODEL_STATUS created a response for {} - {}", requested_model_name, requested_version);
    return StatusCode::OK;
}

::grpc::Status ModelServiceImpl::HandleReloadConfigRequest(
        ::grpc::ServerContext *context, const tensorflow::serving::ReloadConfigRequest *request,
        tensorflow::serving::ReloadConfigResponse *response) {
    spdlog::info("Requested HandleReloadConfigRequest - but this service is reloading config automatically by itself, therefore this operation has no *EXTRA* affect.");
    return grpc::Status::OK;  // we're reloading config all the time; for a total client compatibility, this means returning success here.
}

}  // namespace ovms
