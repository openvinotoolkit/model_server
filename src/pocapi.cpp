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
#include "pocapi.hpp"

#include <cstdint>
#include <memory>
#include <string>

#include "buffer.hpp"
#include "inferenceparameter.hpp"
#include "inferencerequest.hpp"
#include "inferenceresponse.hpp"
#include "inferencetensor.hpp"
#include "model_service.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "profiler.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "server_options.hpp"
#include "status.hpp"
#include "timer.hpp"

using ovms::Buffer;
using ovms::InferenceParameter;
using ovms::InferenceRequest;
using ovms::InferenceResponse;
using ovms::InferenceTensor;
using ovms::ModelInstanceUnloadGuard;
using ovms::ModelManager;
using ovms::ServableManagerModule;
using ovms::Server;
using ovms::Status;
using ovms::StatusCode;
using ovms::Timer;
using std::chrono::microseconds;

OVMS_Status* OVMS_ServerGeneralOptionsNew(OVMS_ServerGeneralOptions** options) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    *options = reinterpret_cast<OVMS_ServerGeneralOptions*>(new ovms::GeneralOptionsImpl);
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsDelete(OVMS_ServerGeneralOptions* options) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    delete reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    return nullptr;
}

OVMS_Status* OVMS_ServerMultiModelOptionsNew(OVMS_ServerMultiModelOptions** options) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    *options = reinterpret_cast<OVMS_ServerMultiModelOptions*>(new ovms::MultiModelOptionsImpl);
    return nullptr;
}

OVMS_Status* OVMS_ServerMultiModelOptionsDelete(OVMS_ServerMultiModelOptions* options) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    delete reinterpret_cast<ovms::MultiModelOptionsImpl*>(options);
    return nullptr;
}

OVMS_Status* OVMS_ServerNew(OVMS_Server** server) {
    // Create new server once multi server configuration becomes possible.
    if (server == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SERVER));
    }
    *server = reinterpret_cast<OVMS_Server*>(&ovms::Server::instance());
    return nullptr;
}

OVMS_Status* OVMS_ServerDelete(OVMS_Server* server) {
    if (server == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SERVER));
    }
    ovms::Server* srv = reinterpret_cast<ovms::Server*>(server);
    srv->shutdownModules();
    // delete passed in ptr once multi server configuration is done
    return nullptr;
}

OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerMultiModelOptions* multi_model_specific_options) {
    if (server == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SERVER));
    }
    if (general_options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (multi_model_specific_options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::Server* srv = reinterpret_cast<ovms::Server*>(server);
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(general_options);
    ovms::MultiModelOptionsImpl* mmo = reinterpret_cast<ovms::MultiModelOptionsImpl*>(multi_model_specific_options);
    std::int64_t res = srv->start(go, mmo);
    return (OVMS_Status*)res;  // TODO: Return proper OVMS_Status instead of a raw status code in error handling PR
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcPort(OVMS_ServerGeneralOptions* options,
    uint32_t grpcPort) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcPort = grpcPort;
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestPort(OVMS_ServerGeneralOptions* options,
    uint32_t restPort) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->restPort = restPort;
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcWorkers(OVMS_ServerGeneralOptions* options,
    uint32_t grpc_workers) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcWorkers = grpc_workers;
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcBindAddress(OVMS_ServerGeneralOptions* options,
    const char* grpc_bind_address) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (grpc_bind_address == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));  // TODO name->string
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcBindAddress.assign(grpc_bind_address);
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestWorkers(OVMS_ServerGeneralOptions* options,
    uint32_t rest_workers) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->restWorkers = rest_workers;
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestBindAddress(OVMS_ServerGeneralOptions* options,
    const char* rest_bind_address) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (rest_bind_address == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));  // TODO name->string
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->restBindAddress.assign(rest_bind_address);
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcChannelArguments(OVMS_ServerGeneralOptions* options,
    const char* grpc_channel_arguments) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (grpc_channel_arguments == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));  // TODO name->string
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcChannelArguments.assign(grpc_channel_arguments);
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetFileSystemPollWaitSeconds(OVMS_ServerGeneralOptions* options,
    uint32_t seconds) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->filesystemPollWaitSeconds = seconds;
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetSequenceCleanerPollWaitMinutes(OVMS_ServerGeneralOptions* options,
    uint32_t minutes) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->sequenceCleanerPollWaitMinutes = minutes;
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetCustomNodeResourcesCleanerIntervalSeconds(OVMS_ServerGeneralOptions* options,
    uint32_t seconds) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->resourcesCleanerPollWaitSeconds = seconds;
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetCpuExtensionPath(OVMS_ServerGeneralOptions* options,
    const char* cpu_extension_path) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (cpu_extension_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));  // TODO name->string
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->cpuExtensionLibraryPath.assign(cpu_extension_path);
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetCacheDir(OVMS_ServerGeneralOptions* options,
    const char* cache_dir) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (cache_dir == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));  // TODO name->string
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->cacheDir.assign(cache_dir);
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetLogLevel(OVMS_ServerGeneralOptions* options,
    OVMS_LogLevel log_level) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    switch (log_level) {
    case OVMS_LOG_INFO:
        go->logLevel = "INFO";
        break;
    case OVMS_LOG_ERROR:
        go->logLevel = "ERROR";
        break;
    case OVMS_LOG_DEBUG:
        go->logLevel = "DEBUG";
        break;
    case OVMS_LOG_TRACE:
        go->logLevel = "TRACE";
        break;
    case OVMS_LOG_WARNING:
        go->logLevel = "WARNING";
        break;
    default:
        // TODO: Return error in error handling PR
        break;
    }
    return nullptr;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetLogPath(OVMS_ServerGeneralOptions* options,
    const char* log_path) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (log_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));  // TODO name->string
    }
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->logPath.assign(log_path);
    return nullptr;
}

OVMS_Status* OVMS_ServerMultiModelOptionsSetConfigPath(OVMS_ServerMultiModelOptions* options,
    const char* config_path) {
    if (options == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_OPTIONS));
    }
    if (config_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));  // TODO name->string
    }
    ovms::MultiModelOptionsImpl* mmo = reinterpret_cast<ovms::MultiModelOptionsImpl*>(options);
    mmo->configPath.assign(config_path);
    return nullptr;
}
// inference API
OVMS_Status* OVMS_InferenceRequestNew(OVMS_InferenceRequest** request, const char* servableName, uint32_t servableVersion) {
    // TODO should we allow to create requests to not yet loaded models?
    if (request == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (servableName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    *request = reinterpret_cast<OVMS_InferenceRequest*>(new InferenceRequest(servableName, servableVersion));
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestDelete(OVMS_InferenceRequest* request) {
    if (request == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    delete reinterpret_cast<InferenceRequest*>(request);
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestAddInput(OVMS_InferenceRequest* req, const char* inputName, OVMS_DataType datatype, const uint64_t* shape, uint32_t dimCount) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    if (shape == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_TABLE));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->addInput(inputName, datatype, shape, dimCount);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* req, const char* inputName, void* data, size_t bufferSize, BufferType bufferType, uint32_t deviceId) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_DATA));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->setInputBuffer(inputName, data, bufferSize, bufferType, deviceId);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestAddParameter(OVMS_InferenceRequest* req, const char* parameterName, OVMS_DataType datatype, const void* data, size_t byteSize) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (parameterName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_DATA));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->addParameter(parameterName, datatype, data);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestRemoveParameter(OVMS_InferenceRequest* req, const char* parameterName) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (parameterName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->removeParameter(parameterName);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestRemoveInput(OVMS_InferenceRequest* req, const char* inputName) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->removeInput(inputName);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestInputRemoveData(OVMS_InferenceRequest* req, const char* inputName) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->removeInputBuffer(inputName);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetOutput(OVMS_InferenceResponse* res, uint32_t id, const char** name, OVMS_DataType* datatype, const uint64_t** shape, uint32_t* dimCount, const void** data, size_t* bytesize, BufferType* bufferType, uint32_t* deviceId) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_RESPONSE));
    }
    if (name == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    if (datatype == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    if (shape == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_TABLE));
    }
    if (dimCount == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_DATA));
    }
    if (bytesize == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    if (bufferType == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    if (deviceId == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    const InferenceTensor* tensor = nullptr;
    const std::string* cppName;
    auto status = response->getOutput(id, &cppName, &tensor);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    if ((tensor == nullptr) ||
        (cppName == nullptr)) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::INTERNAL_ERROR, "InferenceResponse returned nullptr tensor or name"));
    }
    const Buffer* buffer = tensor->getBuffer();
    if (nullptr == buffer) {
        return reinterpret_cast<OVMS_Status*>(new Status(ovms::StatusCode::INTERNAL_ERROR, "InferenceResponse has tensor without buffer"));
    }
    *name = cppName->c_str();
    *datatype = tensor->getDataType();
    *shape = tensor->getShape().data();
    *dimCount = tensor->getShape().size();
    *bufferType = buffer->getBufferType();
    *deviceId = buffer->getDeviceId().value_or(0);  // TODO how discriminate betwen undefined & actual device 0
    // possibly it is not neccessary to discriminate
    *data = buffer->data();
    *bytesize = buffer->getByteSize();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetOutputCount(OVMS_InferenceResponse* res, uint32_t* count) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_RESPONSE));
    }
    if (count == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    *count = response->getOutputCount();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetParameterCount(OVMS_InferenceResponse* res, uint32_t* count) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_RESPONSE));
    }
    if (count == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    *count = response->getParameterCount();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetParameter(OVMS_InferenceResponse* res, uint32_t id, OVMS_DataType* datatype, const void** data) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_RESPONSE));
    }
    if (datatype == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_DATA));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    const InferenceParameter* parameter = response->getParameter(id);
    if (nullptr == parameter) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PARAMETER_FOR_REMOVAL));
    }
    *datatype = parameter->getDataType();
    *data = parameter->getData();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseDelete(OVMS_InferenceResponse* res) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_RESPONSE));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    delete response;
    return nullptr;
}

namespace {
enum : unsigned int {
    TOTAL,
    TIMER_END
};

static Status getModelInstance(ovms::Server& server, const InferenceRequest* request, std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    auto& modelManager = dynamic_cast<const ServableManagerModule*>(server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME))->getServableManager();
    return modelManager.getModelInstance(request->getServableName(), request->getServableVersion(), modelInstance, modelInstanceUnloadGuardPtr);
}
}  // namespace
OVMS_Status* OVMS_Inference(OVMS_Server* serverPtr, OVMS_InferenceRequest* request, OVMS_InferenceResponse** response) {
    OVMS_PROFILE_FUNCTION();
    using std::chrono::microseconds;
    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    if (serverPtr == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SERVER));
    }
    if (request == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (response == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_RESPONSE));
    }
    auto req = reinterpret_cast<ovms::InferenceRequest*>(request);
    ovms::Server& server = *reinterpret_cast<ovms::Server*>(serverPtr);
    std::unique_ptr<ovms::InferenceResponse> res(new ovms::InferenceResponse(req->getServableName(), req->getServableVersion()));

    SPDLOG_DEBUG("Processing C-API request for model: {}; version: {}",
        req->getServableName(),
        req->getServableVersion());

    std::shared_ptr<ovms::ModelInstance> modelInstance;
    //   std::unique_ptr<ovms::Pipeline> pipelinePtr;

    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(server, req, modelInstance, modelInstanceUnloadGuard);

    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", req->getServableName());
        // status = getPipeline(req, response, pipelinePtr);
        status = Status(StatusCode::NOT_IMPLEMENTED, "Inference with DAG not supported with C-API in preview");
    }
    if (!status.ok()) {
        if (modelInstance) {
            //    INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().reqFailGrpcPredict);
        }
        SPDLOG_INFO("Getting modelInstance or pipeline failed. {}", status.string());
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }

    // ExecutionContext executionContext{
    //   ExecutionContext::Interface::CAPI,
    //  ExecutionContext::Method::Inference};

    // if (pipelinePtr) {
    //       status = pipelinePtr->execute(executionContext);
    // INCREMENT_IF_ENABLED(pipelinePtr->getMetricReporter().getInferRequestMetric(executionContext, status.ok()));
    //   } else {
    status = modelInstance->infer(req, res.get(), modelInstanceUnloadGuard);
    //   INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().getInferRequestMetric(executionContext, status.ok()));
    //}

    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }

    timer.stop(TOTAL);
    double reqTotal = timer.elapsed<microseconds>(TOTAL);
    // if (pipelinePtr) {
    //  OBSERVE_IF_ENABLED(pipelinePtr->getMetricReporter().reqTimeGrpc, reqTotal);
    //  } else {
    //   OBSERVE_IF_ENABLED(modelInstance->getMetricReporter().reqTimeGrpc, reqTotal);
    // }
    SPDLOG_DEBUG("Total C-API req processing time: {} ms", reqTotal / 1000);
    *response = reinterpret_cast<OVMS_InferenceResponse*>(res.release());
    return nullptr;
    // return grpc::Status::OK;
}
