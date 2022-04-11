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
#include "modelmanager.hpp"
#include "pipelinedefinition.hpp"
#include "serialization.hpp"

namespace ovms {

using inference::GRPCInferenceService;

const std::string PLATFORM = "OpenVINO";

::grpc::Status KFSInferenceServiceImpl::ServerLive(::grpc::ServerContext* context, const ::inference::ServerLiveRequest* request, ::inference::ServerLiveResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    std::cout << __FUNCTION__ << ":" << __LINE__ << std::endl;
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status KFSInferenceServiceImpl::ServerReady(::grpc::ServerContext* context, const ::inference::ServerReadyRequest* request, ::inference::ServerReadyResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    std::cout << __FUNCTION__ << ":" << __LINE__ << std::endl;
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status KFSInferenceServiceImpl::ModelReady(::grpc::ServerContext* context, const ::inference::ModelReadyRequest* request, ::inference::ModelReadyResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    std::cout << __FUNCTION__ << ":" << __LINE__ << std::endl;
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status KFSInferenceServiceImpl::ServerMetadata(::grpc::ServerContext* context, const ::inference::ServerMetadataRequest* request, ::inference::ServerMetadataResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    std::cout << __FUNCTION__ << ":" << __LINE__ << std::endl;
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status KFSInferenceServiceImpl::ModelMetadata(::grpc::ServerContext* context, const ::inference::ModelMetadataRequest* request, ::inference::ModelMetadataResponse* response) {
    auto& manager = ModelManager::getInstance();
    const auto& name = request->name();
    const auto& version = request->version();

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
    if (!version.empty()) {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; version {}", name, version);
        instance = model->getModelInstanceByVersion(std::stoi(version));
        if (instance == nullptr) {
            SPDLOG_WARN("GetModelMetadata requested model {}; version {} is missing", name, version);
            return Status(StatusCode::MODEL_VERSION_MISSING).grpc();
        }
    } else {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_WARN("GetModelMetadata requested model {}; version {} is missing", name, version);
            return Status(StatusCode::MODEL_VERSION_MISSING).grpc();
        }
    }

    return buildResponse(instance, response).grpc();
}

::grpc::Status KFSInferenceServiceImpl::ModelInfer(::grpc::ServerContext* context, const ::inference::ModelInferRequest* request, ::inference::ModelInferResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    std::cout << __FUNCTION__ << ":" << __LINE__
              << " model:" << request->model_name()
              << " version:" << request->model_version()
              << " id:" << request->id()  // optional field - if specified should be put in response
              << std::endl;
    // TODO parameters - could hold eg. sequence id.
    // TODO inputs
    // TODO outputs
    auto inst = ModelManager::getInstance().findModelInstance("dummy", 1);
    inst->validate(request);
    int floats = 1;
    ov::Tensor tensor;
    std::shared_ptr<TensorInfo> tensorInfo;
    for (int i = 0; i < request->inputs_size(); i++) {
        floats = 1;
        auto input = request->inputs().at(i);
        std::cout << " name:" << input.name()
                  << " dataType:" << input.datatype()
                  << " shape:";
        auto sh = input.shape();
        for (int j = 0; j < sh.size(); j++) {
            std::cout << sh[j] << " ";
            floats *= sh[j];
        }
        std::cout << std::endl;

        tensorInfo = std::make_shared<TensorInfo>(input.name(), kfsPrecisionToOvmsPrecision(input.datatype()), ovms::Shape{1, 16});
        tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(input, tensorInfo, request->raw_input_contents()[i]);
    }

    std::cout << tensor.get_element_type() << tensor.get_shape() << tensor.data() << std::endl;
    // cast to expected input.datatype() to print data
    char* data = (char*)tensor.data();
    for (int i = 0; i < floats; ++i) {
        std::cout << "data2[" << i << "]=" << (*(data + i)) << " ";
    }
    std::cout << std::endl;

    // serialize
    auto output = response->add_outputs();
    serializeTensorToTensorProto(*output, response->add_raw_output_contents(), tensorInfo, tensor);
    response->set_id(request->id());

    return ::grpc::Status(::grpc::StatusCode::OK, "");
}

Status KFSInferenceServiceImpl::buildResponse(
    std::shared_ptr<ModelInstance> instance,
    ::inference::ModelMetadataResponse* response) {

    std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;

    // 0 meaning immediately return unload guard if possible, otherwise do not wait for available state
    auto status = instance->waitForLoaded(0, unloadGuard);
    if (!status.ok()) {
        return status;
    }

    response->Clear();
    response->set_name(instance->getName());
    response->add_versions(std::to_string(instance->getVersion()));
    response->set_platform(PLATFORM);

    for (const auto& input : instance->getInputsInfo()) {
        convert(input, response->add_inputs());
    }

    for (const auto& output : instance->getOutputsInfo()) {
        convert(output, response->add_outputs());
    }

    return StatusCode::OK;
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
    to->set_datatype(from.second->getPrecisionAsKfsPrecision());
    for (auto dim : from.second->getShape()) {
        if (dim.isStatic()) {
            to->add_shape(dim.getStaticValue());
        } else {
            to->add_shape(DYNAMIC_DIMENSION);
        }
    }
}

}  // namespace ovms
