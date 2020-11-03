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

using google::protobuf::util::JsonPrintOptions;
using google::protobuf::util::MessageToJsonString;

namespace ovms {

Status GetModelMetadataImpl::getModelStatus(
    const tensorflow::serving::GetModelMetadataRequest* request,
    tensorflow::serving::GetModelMetadataResponse* response) {
    auto status = validate(request);
    if (!status.ok()) {
        return status;
    }

    const auto& name = request->model_spec().name();

    auto& manager = ovms::ModelManager::getInstance();

    auto model = manager.findModelByName(name);
    if (model == nullptr) {
        SPDLOG_INFO("model {} is  missing", name);
        return StatusCode::MODEL_NAME_MISSING;
    }

    std::shared_ptr<ModelInstance> instance = nullptr;
    if (request->model_spec().has_version() && request->model_spec().version().value() != 0) {
        ovms::model_version_t version = request->model_spec().version().value();
        SPDLOG_DEBUG("requested: name {}; version {}", name, version);
        instance = model->getModelInstanceByVersion(version);
        if (instance == nullptr) {
            SPDLOG_INFO("model {}; version {} is missing", name, version);
            return StatusCode::MODEL_VERSION_MISSING;
        }
    } else {
        SPDLOG_DEBUG("requested: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_INFO("model {}; default version is missing", name);
            return StatusCode::MODEL_VERSION_MISSING;
        }
    }

    return buildResponse(instance, response);
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
    for (const auto& pair : from) {
        auto tensor = pair.second;
        auto& input = (*to)[tensor->getMappedName()];

        input.set_dtype(tensor->getPrecisionAsDataType());

        *input.mutable_name() = tensor->getMappedName();
        *input.mutable_tensor_shape() = tensorflow::TensorShapeProto();

        for (auto dim : tensor->getShape()) {
            input.mutable_tensor_shape()->add_dim()->set_size(dim);
        }
    }
}

Status GetModelMetadataImpl::buildResponse(
    std::shared_ptr<ModelInstance> instance,
    tensorflow::serving::GetModelMetadataResponse* response) {

    const uint WAIT_FOR_LOADED_TIMEOUT_MILLISECONDS = 0;

    std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;
    auto status = instance->waitForLoaded(WAIT_FOR_LOADED_TIMEOUT_MILLISECONDS, unloadGuard);
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

Status GetModelMetadataImpl::createGrpcRequest(std::string model_name, std::optional<int64_t> model_version, tensorflow::serving::GetModelMetadataRequest* request) {
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
        spdlog::error("Failed to convert proto to json. Error: ", status.ToString());
        return StatusCode::JSON_SERIALIZATION_ERROR;
    }
    return StatusCode::OK;
}

}  // namespace ovms
