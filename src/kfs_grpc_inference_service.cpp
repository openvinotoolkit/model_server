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

#include "modelmanager.hpp"
#include "pipelinedefinition.hpp"

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
    // TODO raw_input_contents
    auto it = request->inputs().begin();
    if (it == request->inputs().end())
        throw 1;
    std::cout << " name:" << it->name()
              << " dataType:" << it->datatype()
              << " shape:";
    auto sh = it->shape();
    int floats = 1;

    for (int i = 0; i < sh.size(); ++i) {
        std::cout << sh[i] << " ";
        floats *= sh[i];
    }
    std ::cout << " parameters:"
               << "";
    // TODO check if we want to use this or the  InferTensorContents
    auto size = it->contents().fp32_contents_size();
    std::cout << size

              << " contents size:"
              << "" << size << " content shapesize: " << floats << " content:"
              << std::endl;
    std::cout << "HER:" << it->has_contents() << " " << (!request->raw_input_contents().empty()) << std::endl;
    /*
    float* data = (float*)it->contents().fp32_contents().data();
    for (int i = 0; i < floats; ++i) {
      std::cout << *(data + i) << " ";
    }*/
    if (request->raw_input_contents().size() == 0)
        throw 2;
    const std::string& raw = request->raw_input_contents()[0];
    float* data = (float*)raw.c_str();
    std::cout << raw << std::endl;
    for (int i = 0; i < floats; ++i) {
        std::cout << "data[" << i << "]=" << (*(data + i)) << " ";
        // just showing influence of ovms
        *(data + i) *= 3;
    }
    std::cout << std::endl;
    // serialize
    auto responseTensor = response->outputs();
    // add first output
    auto output = response->add_outputs();
    output->mutable_shape()->Add(1);
    output->mutable_shape()->Add(10);
    output->set_name("a");
    output->set_datatype("FP32");
    std::string* raw_contents = response->add_raw_output_contents();
    size_t bytesize = sizeof(float) * floats;
    raw_contents->reserve(bytesize);
    std::string outputString = std::string(1 * 10 * sizeof(float), (char)0);
    raw_contents->assign((char*)data, bytesize);
    // raw_contents->assign((char*) outputString.c_str(), bytesize);
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
    to->set_datatype(from.second->getPrecisionAsString());
    for (auto dim : from.second->getShape()) {
        if (dim.isStatic()) {
            to->add_shape(dim.getStaticValue());
        } else {
            to->add_shape(DYNAMIC_DIMENSION);
        }
    }
}

}  // namespace ovms
