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
#include <cstdint>
#include <memory>
#include <string>

#include "../buffer.hpp"
#include "../dags/pipeline.hpp"
#include "../execution_context.hpp"
#include "../inferenceparameter.hpp"
#include "../inferencerequest.hpp"
#include "../inferenceresponse.hpp"
#include "../inferencetensor.hpp"
#include "../model_service.hpp"
#include "../modelinstance.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../modelmanager.hpp"
#include "../ovms.h"  // NOLINT
#include "../prediction_service.hpp"
#include "../profiler.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../server_settings.hpp"
#include "../status.hpp"
#include "../timer.hpp"

using ovms::Buffer;
using ovms::ExecutionContext;
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

#ifdef __cplusplus
extern "C" {
#endif

OVMS_Status* OVMS_ApiVersion(uint32_t* major, uint32_t* minor) {
    if (major == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    if (minor == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    *major = OVMS_API_VERSION_MAJOR;
    *minor = OVMS_API_VERSION_MINOR;
    return nullptr;
}

void OVMS_StatusDelete(OVMS_Status* status) {
    if (status == nullptr)
        return;
    delete reinterpret_cast<ovms::Status*>(status);
}

OVMS_Status* OVMS_StatusGetCode(OVMS_Status* status,
    uint32_t* code) {
    if (status == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STATUS));
    if (code == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_NUMBER));
    ovms::Status* sts = reinterpret_cast<ovms::Status*>(status);
    *code = static_cast<uint32_t>(sts->getCode());
    return nullptr;
}

OVMS_Status* OVMS_StatusGetDetails(OVMS_Status* status,
    const char** details) {
    if (status == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STATUS));
    if (details == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    ovms::Status* sts = reinterpret_cast<ovms::Status*>(status);
    *details = sts->string().c_str();
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsNew(OVMS_ServerSettings** settings) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    *settings = reinterpret_cast<OVMS_ServerSettings*>(new ovms::ServerSettingsImpl);
    return nullptr;
}

void OVMS_ServerSettingsDelete(OVMS_ServerSettings* settings) {
    if (settings == nullptr)
        return;
    delete reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
}

OVMS_Status* OVMS_ModelsSettingsNew(OVMS_ModelsSettings** settings) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    *settings = reinterpret_cast<OVMS_ModelsSettings*>(new ovms::ModelsSettingsImpl);
    return nullptr;
}

void OVMS_ModelsSettingsDelete(OVMS_ModelsSettings* settings) {
    if (settings == nullptr)
        return;
    delete reinterpret_cast<ovms::ModelsSettingsImpl*>(settings);
}

OVMS_Status* OVMS_ServerNew(OVMS_Server** server) {
    // Create new server once multi server configuration becomes possible.
    if (server == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SERVER));
    }
    *server = reinterpret_cast<OVMS_Server*>(&ovms::Server::instance());
    return nullptr;
}

void OVMS_ServerDelete(OVMS_Server* server) {
    if (server == nullptr)
        return;
    ovms::Server* srv = reinterpret_cast<ovms::Server*>(server);
    srv->shutdownModules();
    // delete passed in ptr once multi server configuration is done
}

OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerSettings* server_settings,
    OVMS_ModelsSettings* models_settings) {
    if (server == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SERVER));
    }
    if (server_settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (models_settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::Server* srv = reinterpret_cast<ovms::Server*>(server);
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(server_settings);
    ovms::ModelsSettingsImpl* modelsSettings = reinterpret_cast<ovms::ModelsSettingsImpl*>(models_settings);
    auto res = srv->start(serverSettings, modelsSettings);
    if (res.ok())
        return nullptr;
    return reinterpret_cast<OVMS_Status*>(new ovms::Status(res));
}

OVMS_Status* OVMS_ServerSettingsSetGrpcPort(OVMS_ServerSettings* settings,
    uint32_t grpcPort) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcPort = grpcPort;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetRestPort(OVMS_ServerSettings* settings,
    uint32_t restPort) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->restPort = restPort;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetGrpcWorkers(OVMS_ServerSettings* settings,
    uint32_t grpc_workers) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcWorkers = grpc_workers;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetGrpcBindAddress(OVMS_ServerSettings* settings,
    const char* grpc_bind_address) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (grpc_bind_address == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcBindAddress.assign(grpc_bind_address);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetRestWorkers(OVMS_ServerSettings* settings,
    uint32_t rest_workers) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->restWorkers = rest_workers;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetRestBindAddress(OVMS_ServerSettings* settings,
    const char* rest_bind_address) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (rest_bind_address == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->restBindAddress.assign(rest_bind_address);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetGrpcChannelArguments(OVMS_ServerSettings* settings,
    const char* grpc_channel_arguments) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (grpc_channel_arguments == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcChannelArguments.assign(grpc_channel_arguments);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetFileSystemPollWaitSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->filesystemPollWaitSeconds = seconds;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(OVMS_ServerSettings* settings,
    uint32_t minutes) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->sequenceCleanerPollWaitMinutes = minutes;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->resourcesCleanerPollWaitSeconds = seconds;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetCpuExtensionPath(OVMS_ServerSettings* settings,
    const char* cpu_extension_path) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (cpu_extension_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->cpuExtensionLibraryPath.assign(cpu_extension_path);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetCacheDir(OVMS_ServerSettings* settings,
    const char* cache_dir) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (cache_dir == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->cacheDir.assign(cache_dir);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetLogLevel(OVMS_ServerSettings* settings,
    OVMS_LogLevel log_level) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    switch (log_level) {
    case OVMS_LOG_INFO:
        serverSettings->logLevel = "INFO";
        break;
    case OVMS_LOG_ERROR:
        serverSettings->logLevel = "ERROR";
        break;
    case OVMS_LOG_DEBUG:
        serverSettings->logLevel = "DEBUG";
        break;
    case OVMS_LOG_TRACE:
        serverSettings->logLevel = "TRACE";
        break;
    case OVMS_LOG_WARNING:
        serverSettings->logLevel = "WARNING";
        break;
    default:
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_LOG_LEVEL));
    }
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetLogPath(OVMS_ServerSettings* settings,
    const char* log_path) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (log_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->logPath.assign(log_path);
    return nullptr;
}

OVMS_Status* OVMS_ModelsSettingsSetConfigPath(OVMS_ModelsSettings* settings,
    const char* config_path) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SETTINGS));
    }
    if (config_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    ovms::ModelsSettingsImpl* modelsSettings = reinterpret_cast<ovms::ModelsSettingsImpl*>(settings);
    modelsSettings->configPath.assign(config_path);
    return nullptr;
}
// inference API
OVMS_Status* OVMS_InferenceRequestNew(OVMS_InferenceRequest** request, OVMS_Server* server, const char* servableName, int64_t servableVersion) {
    if (request == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_REQUEST));
    }
    if (server == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_SERVER));
    }
    if (servableName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_STRING));
    }
    *request = reinterpret_cast<OVMS_InferenceRequest*>(new InferenceRequest(servableName, servableVersion));
    return nullptr;
}

void OVMS_InferenceRequestDelete(OVMS_InferenceRequest* request) {
    if (request == nullptr)
        return;
    delete reinterpret_cast<InferenceRequest*>(request);
}

OVMS_Status* OVMS_InferenceRequestAddInput(OVMS_InferenceRequest* req, const char* inputName, OVMS_DataType datatype, const int64_t* shape, size_t dimCount) {
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

OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* req, const char* inputName, const void* data, size_t bufferSize, OVMS_BufferType bufferType, uint32_t deviceId) {
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

OVMS_Status* OVMS_InferenceResponseGetOutput(OVMS_InferenceResponse* res, uint32_t id, const char** name, OVMS_DataType* datatype, const int64_t** shape, size_t* dimCount, const void** data, size_t* bytesize, OVMS_BufferType* bufferType, uint32_t* deviceId) {
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
    *deviceId = buffer->getDeviceId().value_or(0);
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

void OVMS_InferenceResponseDelete(OVMS_InferenceResponse* res) {
    if (res == nullptr)
        return;
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    delete response;
}

namespace {
enum : unsigned int {
    TOTAL,
    TIMER_END
};

static Status getModelInstance(ovms::Server& server, const InferenceRequest* request, std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    if (!server.isLive()) {
        return ovms::Status(ovms::StatusCode::SERVER_NOT_READY_FOR_INFERENCE, "not live");
    }
    if (!server.isReady()) {
        return ovms::Status(ovms::StatusCode::SERVER_NOT_READY_FOR_INFERENCE, "not ready");
    }
    const ovms::Module* servableModule = server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    if (!servableModule) {
        return ovms::Status(ovms::StatusCode::INTERNAL_ERROR, "missing servable manager");
    }
    auto& modelManager = dynamic_cast<const ServableManagerModule*>(servableModule)->getServableManager();
    return modelManager.getModelInstance(request->getServableName(), request->getServableVersion(), modelInstance, modelInstanceUnloadGuardPtr);
}

Status getPipeline(ovms::Server& server, const InferenceRequest* request,
    InferenceResponse* response,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr) {
    OVMS_PROFILE_FUNCTION();
    if (!server.isLive()) {
        return ovms::Status(ovms::StatusCode::SERVER_NOT_READY_FOR_INFERENCE, "not live");
    }
    if (!server.isReady()) {
        return ovms::Status(ovms::StatusCode::SERVER_NOT_READY_FOR_INFERENCE, "not ready");
    }
    const ovms::Module* servableModule = server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    if (!servableModule) {
        return ovms::Status(ovms::StatusCode::INTERNAL_ERROR, "missing servable manager");
    }
    auto& modelManager = dynamic_cast<const ServableManagerModule*>(servableModule)->getServableManager();
    return modelManager.createPipeline(pipelinePtr, request->getServableName(), request, response);
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
    std::unique_ptr<ovms::Pipeline> pipelinePtr;

    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(server, req, modelInstance, modelInstanceUnloadGuard);

    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", req->getServableName());
        status = getPipeline(server, req, res.get(), pipelinePtr);
    }
    if (!status.ok()) {
        if (modelInstance) {
            //    INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().reqFailGrpcPredict);
        }
        SPDLOG_INFO("Getting modelInstance or pipeline failed. {}", status.string());
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    // fix execution context and metrics
    ExecutionContext executionContext{
        ExecutionContext::Interface::GRPC,
        ExecutionContext::Method::ModelInfer};

    if (pipelinePtr) {
        status = pipelinePtr->execute(executionContext);
        // INCREMENT_IF_ENABLED(pipelinePtr->getMetricReporter().getInferRequestMetric(executionContext, status.ok()));
    } else {
        status = modelInstance->infer(req, res.get(), modelInstanceUnloadGuard);
        //   INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().getInferRequestMetric(executionContext, status.ok()));
    }

    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }

    timer.stop(TOTAL);
    double reqTotal = timer.elapsed<microseconds>(TOTAL);
    if (pipelinePtr) {
        //  OBSERVE_IF_ENABLED(pipelinePtr->getMetricReporter().reqTimeGrpc, reqTotal);
    } else {
        //   OBSERVE_IF_ENABLED(modelInstance->getMetricReporter().reqTimeGrpc, reqTotal);
    }
    SPDLOG_DEBUG("Total C-API req processing time: {} ms", reqTotal / 1000);
    *response = reinterpret_cast<OVMS_InferenceResponse*>(res.release());
    return nullptr;
}

#ifdef __cplusplus
}
#endif
