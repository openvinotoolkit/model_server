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

namespace ovms {

Status GetModelMetadataImpl::getModelStatus(
    const   tensorflow::serving::GetModelMetadataRequest*   request,
            tensorflow::serving::GetModelMetadataResponse*  response) {
    auto status = validate(request);
    if (!status.ok()) {
        return status;
    }

    const auto& name = request->model_spec().name();

    auto& manager = ovms::ModelManager::getInstance();

    ovms::model_version_t version = 0;
    if (request->model_spec().has_version()) {
        version = request->model_spec().version().value();
    }

    auto instance = manager.findModelInstance(name, version);
    if (instance == nullptr) {
        return version == 0 ? StatusCode::MODEL_NAME_MISSING : StatusCode::MODEL_VERSION_MISSING;
    }

    if (ModelVersionState::AVAILABLE != instance->getStatus().getState()) {
        return StatusCode::MODEL_MISSING;
    }

    buildResponse(instance, response);

    return StatusCode::OK;
}

Status GetModelMetadataImpl::validate(
    const   tensorflow::serving::GetModelMetadataRequest*   request) {

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
    const   tensor_map_t&           from,
            proto_signature_map_t*  to) {
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

void GetModelMetadataImpl::buildResponse(
    std::shared_ptr<ModelInstance>                  instance,
    tensorflow::serving::GetModelMetadataResponse*  response) {

    response->Clear();
    response->mutable_model_spec()->set_name(instance->getName());
    response->mutable_model_spec()->mutable_version()->set_value(instance->getVersion());

    tensorflow::serving::SignatureDefMap def;
    convert(instance->getInputsInfo(), ((*def.mutable_signature_def())["serving_default"]).mutable_inputs());
    convert(instance->getOutputsInfo(), ((*def.mutable_signature_def())["serving_default"]).mutable_outputs());

    (*response->mutable_metadata())["signature_def"].PackFrom(def);
}

}  // namespace ovms
