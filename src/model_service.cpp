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

#include "modelmanager.hpp"
#include "model_service.hpp"

#include <spdlog/spdlog.h>

namespace ovms {

void addStatusToResponse(tensorflow::serving::GetModelStatusResponse* response, model_version_t version, const ModelVersionStatus& model_version_status) {
  spdlog::debug("add_status_to_response version={} status={}", version, model_version_status.getStateString());
  auto status_to_fill = response->add_model_version_status();
  status_to_fill->set_state(static_cast<tensorflow::serving::ModelVersionStatus_State>(static_cast<int>(model_version_status.getState())));
  status_to_fill->set_version(version);
}

::grpc::Status ModelServiceImpl::GetModelStatus(
    ::grpc::ServerContext *context, const tensorflow::serving::GetModelStatusRequest *request,
    tensorflow::serving::GetModelStatusResponse *response) {
  spdlog::debug("model_service: request: {}", request->DebugString());
  bool has_requested_version = request->model_spec().has_version();
  auto requested_version = request->model_spec().version().value();
  std::string requested_model_name = request->model_spec().name();
  spdlog::debug("requested model: {}, has_version: {} (version: {})", requested_model_name, has_requested_version, requested_version);
  if(has_requested_version || requested_version != 0) {
    // return details only for a specific version of requested model; NOT_FOUND otherwise. If requested_version == 0, default is returned.
    std::shared_ptr<ModelInstance> model_instance = ModelManager::getInstance().findModelInstance(requested_model_name, requested_version);
    if(!model_instance) {
      spdlog::info("requested model {} in version {} was not found.", requested_model_name, requested_version);
      return grpc::Status(grpc::StatusCode::NOT_FOUND, GetModelStatus::getError(GetModelStatusCode::NO_SUCH_MODEL_VERSION));
    }
    const auto& status = model_instance->getStatus();
    spdlog::debug("adding model {} - {} :: {} to response", requested_model_name, requested_version, status.getStateString());
    addStatusToResponse(response, requested_version, status);
  } else {
    // return status details of all versions of a requested model.
    const std::shared_ptr<Model> model_ptr = ModelManager::getInstance().findModelByName(requested_model_name);
    if(!model_ptr) {
      spdlog::info("requested model {} was not found.", requested_model_name);
      return grpc::Status(grpc::StatusCode::NOT_FOUND, GetModelStatus::getError(GetModelStatusCode::NO_SUCH_MODEL_NAME));
    }
    auto model_versions = model_ptr->getModelVersions();
    for(const auto& [model_version, model_instance_ptr] : model_versions) {
      if(!model_instance_ptr) {
        spdlog::error("during model iteration, found null model instance pointer!");
        return grpc::Status(grpc::StatusCode::UNKNOWN, GetModelStatus::getError(GetModelStatusCode::INTERNAL_ERROR));
      }
      const auto& status = model_instance_ptr->getStatus();
      spdlog::debug("adding model {} - {} :: {} to response", requested_model_name, model_version, status.getStateString());
      addStatusToResponse(response, model_version, status);
    }
  }
  spdlog::debug("model_service: response: {}", response->DebugString());
  spdlog::debug("MODEL_STATUS created a response for {} - {}", requested_model_name, requested_version);
  return grpc::Status::OK;
}

::grpc::Status ModelServiceImpl::HandleReloadConfigRequest(
    ::grpc::ServerContext *context, const tensorflow::serving::ReloadConfigRequest *request,
    tensorflow::serving::ReloadConfigResponse *response) {
  spdlog::info("Requested HandleReloadConfigRequest - but this service is reloading config automatically by itself, therefore this operation has no *EXTRA* affect.");
  return grpc::Status::OK; // we're reloading config all the time; for a total client compatibility, this means returning success here.
}

} // namespace ovms
