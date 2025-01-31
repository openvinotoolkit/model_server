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
#include "get_model_metadata_impl.hpp"

#include <google/protobuf/util/json_util.h>

#include "dags/pipelinedefinition.hpp"
#include "dags/pipelinedefinitionstatus.hpp"
#include "dags/pipelinedefinitionunloadguard.hpp"
#include "execution_context.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "status.hpp"
#include "tfs_frontend/tfs_utils.hpp"

using google::protobuf::util::JsonPrintOptions;
using google::protobuf::util::MessageToJsonString;

namespace ovms {
GetModelMetadataImpl::GetModelMetadataImpl(ovms::Server& ovmsServer) :
    modelManager(dynamic_cast<const ServableManagerModule*>(ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager()) {
    if (nullptr == ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME)) {
        const char* message = "Tried to create model metadata impl without servable manager module";
        SPDLOG_ERROR("{}", message);
        throw std::logic_error(message);
    }
}

Status GetModelMetadataImpl::getModelStatus(
    const tensorflow::serving::GetModelMetadataRequest* request,
    tensorflow::serving::GetModelMetadataResponse* response,
    ExecutionContext context) const {
    auto status = validate(request);
    if (!status.ok()) {
        return status;
    }
    return getModelStatus(request, response, modelManager, context);
}

Status GetModelMetadataImpl::getModelStatus(
    const tensorflow::serving::GetModelMetadataRequest* request,
    tensorflow::serving::GetModelMetadataResponse* response,
    ModelManager& manager,
    ExecutionContext context) {
    const auto& name = request->model_spec().name();
    model_version_t version = request->model_spec().has_version() ? request->model_spec().version().value() : 0;

    auto model = manager.findModelByName(name);
    if (model == nullptr) {
        SPDLOG_DEBUG("GetModelMetadata: Model {} is missing, trying to find pipeline with such name", name);
        auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(name);
        if (!pipelineDefinition) {
            return StatusCode::MODEL_NAME_MISSING;
        }
        auto status = buildResponse(*pipelineDefinition, response, manager);
        INCREMENT_IF_ENABLED(pipelineDefinition->getMetricReporter().getGetModelMetadataRequestMetric(context, status.ok()));
        return status;
    }

    std::shared_ptr<ModelInstance> instance = nullptr;
    if (version != 0) {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; version {}", name, version);
        instance = model->getModelInstanceByVersion(version);
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; version {} is missing", name, version);
            return StatusCode::MODEL_VERSION_MISSING;
        }
    } else {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; default version is missing", name);
            return StatusCode::MODEL_VERSION_MISSING;
        }
    }

    auto status = buildResponse(instance, response);
    INCREMENT_IF_ENABLED(instance->getMetricReporter().getGetModelMetadataRequestMetric(context, status.ok()));
    return status;
}

Status GetModelMetadataImpl::validate(
    const tensorflow::serving::GetModelMetadataRequest* request) {

    if (!request->has_model_spec()) {
        return StatusCode::MODEL_SPEC_MISSING;
    }

    if (request->metadata_field_size() != 1) {
        return StatusCode::INVALID_SIGNATURE_DEF;
    }

    const auto& signature = request->metadata_field().at(0);

    if (signature != "signature_def") {
        return StatusCode::INVALID_SIGNATURE_DEF;
    }

    return StatusCode::OK;
}

void GetModelMetadataImpl::convert(
    const tensor_map_t& from,
    proto_signature_map_t* to) {
    for (const auto& [name, tensor] : from) {
        auto& input = (*to)[name];

        input.set_dtype(getPrecisionAsDataType(tensor->getPrecision()));

        // Since this method is used for models and pipelines we cannot rely on tensor getMappedName().
        // In both cases we can rely on tensor_map key values as final names.
        *input.mutable_name() = name;
        *input.mutable_tensor_shape() = tensorflow::TensorShapeProto();

        for (const auto& dim : tensor->getShape()) {
            if (dim.isStatic()) {
                input.mutable_tensor_shape()->add_dim()->set_size(dim.getStaticValue());
            } else {
                input.mutable_tensor_shape()->add_dim()->set_size(DYNAMIC_DIMENSION);
            }
        }
    }
}

Status GetModelMetadataImpl::buildResponse(
    std::shared_ptr<ModelInstance> instance,
    tensorflow::serving::GetModelMetadataResponse* response) {

    std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;

    // 0 meaning immediately return unload guard if possible, otherwise do not wait for available state
    auto status = instance->waitForLoaded(0, unloadGuard);
    if (!status.ok()) {
        return status;
    }

    response->Clear();
    response->mutable_model_spec()->set_name(instance->getName());
    response->mutable_model_spec()->mutable_version()->set_value(instance->getVersion());

    tensorflow::serving::SignatureDefMap def;
    convert(instance->getInputsInfo(), ((*def.mutable_signature_def())["serving_default"]).mutable_inputs());
    convert(instance->getOutputsInfo(), ((*def.mutable_signature_def())["serving_default"]).mutable_outputs());

    (*response->mutable_metadata())["signature_def"].PackFrom(def);
    return StatusCode::OK;
}

Status GetModelMetadataImpl::buildResponse(
    PipelineDefinition& pipelineDefinition,
    tensorflow::serving::GetModelMetadataResponse* response,
    const ModelManager& manager) {

    // 0 meaning immediately return unload guard if possible, otherwise do not wait for available state
    std::unique_ptr<PipelineDefinitionUnloadGuard> unloadGuard;
    auto status = pipelineDefinition.waitForLoaded(unloadGuard, 0);
    if (!status.ok()) {
        return status;
    }

    const tensor_map_t& inputs = pipelineDefinition.getInputsInfo();
    const tensor_map_t& outputs = pipelineDefinition.getOutputsInfo();

    response->Clear();
    response->mutable_model_spec()->set_name(pipelineDefinition.getName());
    response->mutable_model_spec()->mutable_version()->set_value(1);

    tensorflow::serving::SignatureDefMap def;
    convert(inputs, ((*def.mutable_signature_def())["serving_default"]).mutable_inputs());
    convert(outputs, ((*def.mutable_signature_def())["serving_default"]).mutable_outputs());

    (*response->mutable_metadata())["signature_def"].PackFrom(def);
    return StatusCode::OK;
}

Status GetModelMetadataImpl::createGrpcRequest(const std::string& model_name, std::optional<int64_t> model_version, tensorflow::serving::GetModelMetadataRequest* request) {
    request->mutable_model_spec()->set_name(model_name);
    if (model_version.has_value()) {
        request->mutable_model_spec()->mutable_version()->set_value(model_version.value());
    }
    request->mutable_metadata_field()->Add("signature_def");
    return StatusCode::OK;
}

Status GetModelMetadataImpl::serializeResponse2Json(const tensorflow::serving::GetModelMetadataResponse* response, std::string* output) {
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

}  // namespace ovms
