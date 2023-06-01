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
#include <iterator>
#include <memory>
#include <string>

#include "../dags/pipeline.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../dags/pipelinedefinitionunloadguard.hpp"
#include "../execution_context.hpp"
#include "../model_service.hpp"
#include "../modelinstance.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../modelmanager.hpp"
#include "../ovms.h"  // NOLINT
#include "../prediction_service.hpp"
#include "../profiler.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "../timer.hpp"
#include "buffer.hpp"
#include "capi_utils.hpp"
#include "inferenceparameter.hpp"
#include "inferencerequest.hpp"
#include "inferenceresponse.hpp"
#include "inferencetensor.hpp"
#include "servablemetadata.hpp"
#include "server_settings.hpp"

using ovms::Buffer;
using ovms::ExecutionContext;
using ovms::InferenceParameter;
using ovms::InferenceRequest;
using ovms::InferenceResponse;
using ovms::InferenceTensor;
using ovms::ModelInstanceUnloadGuard;
using ovms::ModelManager;
using ovms::Pipeline;
using ovms::PipelineDefinition;
using ovms::PipelineDefinitionUnloadGuard;
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "major version"));
    if (minor == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "minor version"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "status"));
    if (code == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "code"));
    ovms::Status* sts = reinterpret_cast<ovms::Status*>(status);
    *code = static_cast<uint32_t>(sts->getCode());
    return nullptr;
}

OVMS_Status* OVMS_StatusGetDetails(OVMS_Status* status,
    const char** details) {
    if (status == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "status"));
    if (details == nullptr)
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "details"));
    ovms::Status* sts = reinterpret_cast<ovms::Status*>(status);
    *details = sts->string().c_str();
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsNew(OVMS_ServerSettings** settings) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "settings"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "model settings"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server"));
    }
    // Hack to force spdlog singleton to initialize before ovms::Server singleton
    if (spdlog::get("notUsedLogger")) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::INTERNAL_ERROR, "unexpected error during spdlog configuration"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server"));
    }
    if (server_settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    if (models_settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "model settings"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcPort = grpcPort;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetRestPort(OVMS_ServerSettings* settings,
    uint32_t restPort) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->restPort = restPort;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetGrpcWorkers(OVMS_ServerSettings* settings,
    uint32_t grpc_workers) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcWorkers = grpc_workers;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetGrpcBindAddress(OVMS_ServerSettings* settings,
    const char* grpc_bind_address) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    if (grpc_bind_address == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "grpc bind address"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcBindAddress.assign(grpc_bind_address);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetRestWorkers(OVMS_ServerSettings* settings,
    uint32_t rest_workers) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->restWorkers = rest_workers;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetRestBindAddress(OVMS_ServerSettings* settings,
    const char* rest_bind_address) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    if (rest_bind_address == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "rest bind address"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->restBindAddress.assign(rest_bind_address);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetGrpcChannelArguments(OVMS_ServerSettings* settings,
    const char* grpc_channel_arguments) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    if (grpc_channel_arguments == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "grpc channel arguments"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->grpcChannelArguments.assign(grpc_channel_arguments);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetFileSystemPollWaitSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->filesystemPollWaitSeconds = seconds;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(OVMS_ServerSettings* settings,
    uint32_t minutes) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->sequenceCleanerPollWaitMinutes = minutes;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->resourcesCleanerPollWaitSeconds = seconds;
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetCpuExtensionPath(OVMS_ServerSettings* settings,
    const char* cpu_extension_path) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    if (cpu_extension_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "cpu extension path"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->cpuExtensionLibraryPath.assign(cpu_extension_path);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetCacheDir(OVMS_ServerSettings* settings,
    const char* cache_dir) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    if (cache_dir == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "cache dir"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->cacheDir.assign(cache_dir);
    return nullptr;
}

OVMS_Status* OVMS_ServerSettingsSetLogLevel(OVMS_ServerSettings* settings,
    OVMS_LogLevel log_level) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server settings"));
    }
    if (log_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "log path"));
    }
    ovms::ServerSettingsImpl* serverSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(settings);
    serverSettings->logPath.assign(log_path);
    return nullptr;
}

OVMS_Status* OVMS_ModelsSettingsSetConfigPath(OVMS_ModelsSettings* settings,
    const char* config_path) {
    if (settings == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "models settings"));
    }
    if (config_path == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "config path"));
    }
    ovms::ModelsSettingsImpl* modelsSettings = reinterpret_cast<ovms::ModelsSettingsImpl*>(settings);
    modelsSettings->configPath.assign(config_path);
    return nullptr;
}
// inference API
OVMS_Status* OVMS_InferenceRequestNew(OVMS_InferenceRequest** request, OVMS_Server* server, const char* servableName, int64_t servableVersion) {
    if (request == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (server == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server"));
    }
    if (servableName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable name"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "input name"));
    }
    if (shape == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "shape"));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->addInput(inputName, datatype, shape, dimCount);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    if (spdlog::default_logger_raw()->level() == spdlog::level::trace) {
        std::stringstream ss;
        ss << "C-API adding request input for servable: " << request->getServableName()
           << " version: " << request->getServableVersion()
           << " name: " << inputName
           << " datatype: " << toString(ovms::getOVMSDataTypeAsPrecision(datatype))
           << " shape: [";
        size_t i = 0;
        for (i = 0; i < dimCount - 1; ++i) {
            ss << shape[i] << ", ";
        }
        ss << shape[i] << "]";
        SPDLOG_TRACE(ss.str());
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* req, const char* inputName, const void* data, size_t bufferSize, OVMS_BufferType bufferType, uint32_t deviceId) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "input name"));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data"));
    }
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->setInputBuffer(inputName, data, bufferSize, bufferType, deviceId);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    if (spdlog::default_logger_raw()->level() == spdlog::level::trace) {
        std::stringstream ss;
        ss << "C-API setting request input data for servable: " << request->getServableName()
           << " version: " << request->getServableVersion()
           << " name: " << inputName
           << " bufferType: " << bufferType
           << " deviceId: " << deviceId;
        SPDLOG_TRACE(ss.str());
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestAddParameter(OVMS_InferenceRequest* req, const char* parameterName, OVMS_DataType datatype, const void* data, size_t byteSize) {
    if (req == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (parameterName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "parameter name"));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (parameterName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "parameter name"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "input name"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (inputName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "input name"));
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
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference response"));
    }
    if (name == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "output name"));
    }
    if (datatype == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data type"));
    }
    if (shape == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "shape"));
    }
    if (dimCount == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "dimension count"));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data"));
    }
    if (bytesize == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "byte size"));
    }
    if (bufferType == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "buffer type"));
    }
    if (deviceId == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "device id"));
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
    if (spdlog::default_logger_raw()->level() == spdlog::level::trace) {
        std::stringstream ss;
        ss << "C-API getting response output of servable: " << response->getServableName()
           << " version: " << response->getServableVersion()
           << " output id: " << id
           << " name: " << *cppName
           << " datatype: " << toString(ovms::getOVMSDataTypeAsPrecision(*datatype))
           << " shape: [";
        size_t i = 0;
        for (i = 0; i < *dimCount - 1; ++i) {
            ss << (*shape)[i] << ", ";
        }
        ss << (*shape)[i] << "]"
           << " bufferType: " << *bufferType
           << " deviceId: " << *deviceId;
        SPDLOG_TRACE(ss.str());
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetOutputCount(OVMS_InferenceResponse* res, uint32_t* count) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference response"));
    }
    if (count == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "output count"));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    *count = response->getOutputCount();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetParameterCount(OVMS_InferenceResponse* res, uint32_t* count) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference response"));
    }
    if (count == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "parameter count"));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    *count = response->getParameterCount();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetParameter(OVMS_InferenceResponse* res, uint32_t id, OVMS_DataType* datatype, const void** data) {
    if (res == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference response"));
    }
    if (datatype == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data type"));
    }
    if (data == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data"));
    }
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    const InferenceParameter* parameter = response->getParameter(id);
    if (nullptr == parameter) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PARAMETER));
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

static Status getModelManager(Server& server, ModelManager** modelManager) {
    if (!server.isLive()) {
        return ovms::Status(ovms::StatusCode::SERVER_NOT_READY, "not live");
    }
    if (!server.isReady()) {
        return ovms::Status(ovms::StatusCode::SERVER_NOT_READY, "not ready");
    }
    const ovms::Module* servableModule = server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    if (!servableModule) {
        return ovms::Status(ovms::StatusCode::INTERNAL_ERROR, "missing servable manager");
    }
    *modelManager = &dynamic_cast<const ServableManagerModule*>(servableModule)->getServableManager();
    return StatusCode::OK;
}

static Status getModelInstance(ovms::Server& server, const std::string& modelName, int64_t modelVersion, std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    ModelManager* modelManager{nullptr};
    auto status = getModelManager(server, &modelManager);
    if (!status.ok()) {
        return status;
    }
    return modelManager->getModelInstance(modelName, modelVersion, modelInstance, modelInstanceUnloadGuardPtr);
}

static Status getPipeline(ovms::Server& server, const InferenceRequest* request,
    InferenceResponse* response,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr) {
    OVMS_PROFILE_FUNCTION();
    ModelManager* modelManager{nullptr};
    auto status = getModelManager(server, &modelManager);
    if (!status.ok()) {
        return status;
    }
    return modelManager->createPipeline(pipelinePtr, request->getServableName(), request, response);
}

static Status getPipelineDefinition(Server& server, const std::string& servableName, PipelineDefinition** pipelineDefinition, std::unique_ptr<PipelineDefinitionUnloadGuard>& unloadGuard) {
    ModelManager* modelManager{nullptr};
    Status status = getModelManager(server, &modelManager);
    if (!status.ok()) {
        return status;
    }
    *pipelineDefinition = modelManager->getPipelineFactory().findDefinitionByName(servableName);
    if (!*pipelineDefinition) {
        return Status(StatusCode::PIPELINE_DEFINITION_NAME_MISSING);
    }
    return (*pipelineDefinition)->waitForLoaded(unloadGuard, 0);
}

}  // namespace

OVMS_Status* OVMS_Inference(OVMS_Server* serverPtr, OVMS_InferenceRequest* request, OVMS_InferenceResponse** response) {
    OVMS_PROFILE_FUNCTION();
    using std::chrono::microseconds;
    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    if (serverPtr == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server"));
    }
    if (request == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference request"));
    }
    if (response == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "inference response"));
    }
    auto req = reinterpret_cast<ovms::InferenceRequest*>(request);
    ovms::Server& server = *reinterpret_cast<ovms::Server*>(serverPtr);

    SPDLOG_DEBUG("Processing C-API request for model: {}; version: {}",
        req->getServableName(),
        req->getServableVersion());

    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::Pipeline> pipelinePtr;

    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(server, req->getServableName(), req->getServableVersion(), modelInstance, modelInstanceUnloadGuard);

    std::unique_ptr<ovms::InferenceResponse> res(new ovms::InferenceResponse(req->getServableName(), req->getServableVersion()));
    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", req->getServableName());
        status = getPipeline(server, req, res.get(), pipelinePtr);
    }
    if (!status.ok()) {
        if (modelInstance) {
            //    INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().reqFailGrpcPredict);
        }
        SPDLOG_DEBUG("Getting modelInstance or pipeline failed. {}", status.string());
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

OVMS_Status* OVMS_GetServableMetadata(OVMS_Server* serverPtr, const char* servableName, int64_t servableVersion, OVMS_ServableMetadata** servableMetadata) {
    if (serverPtr == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "server"));
    }
    if (servableName == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable name"));
    }
    if (servableMetadata == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable metadata"));
    }
    // TODO check inputs
    // TODO metrics
    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::Pipeline> pipelinePtr;
    ovms::Server& server = *reinterpret_cast<ovms::Server*>(serverPtr);
    auto status = getModelInstance(server, servableName, servableVersion, modelInstance, modelInstanceUnloadGuard);

    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", servableName);
        PipelineDefinition* pipelineDefinition = nullptr;
        std::unique_ptr<PipelineDefinitionUnloadGuard> unloadGuard;
        status = getPipelineDefinition(server, servableName, &pipelineDefinition, unloadGuard);
        if (!status.ok() || !pipelineDefinition) {
            return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::MODEL_NAME_MISSING));
        }
        *servableMetadata = reinterpret_cast<OVMS_ServableMetadata*>(new ovms::ServableMetadata(servableName, servableVersion, pipelineDefinition->getInputsInfo(), pipelineDefinition->getOutputsInfo()));
        return nullptr;
    }
    if (!status.ok()) {
        if (modelInstance) {
            //    INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().reqFailGrpcPredict);
        }
        SPDLOG_INFO("Getting modelInstance or pipeline failed. {}", status.string());
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    *servableMetadata = reinterpret_cast<OVMS_ServableMetadata*>(new ovms::ServableMetadata(servableName, servableVersion, modelInstance->getInputsInfo(), modelInstance->getOutputsInfo(), modelInstance->getRTInfo()));
    return nullptr;
}

OVMS_Status* OVMS_ServableMetadataGetInputCount(OVMS_ServableMetadata* servableMetadata, uint32_t* count) {
    if (servableMetadata == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable metadata"));
    }
    if (count == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "input count"));
    }
    ovms::ServableMetadata* metadata = reinterpret_cast<ovms::ServableMetadata*>(servableMetadata);
    *count = metadata->getInputsInfo().size();
    return nullptr;
}

OVMS_Status* OVMS_ServableMetadataGetOutputCount(OVMS_ServableMetadata* servableMetadata, uint32_t* count) {
    if (servableMetadata == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable metadata"));
    }
    if (count == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "output count"));
    }
    ovms::ServableMetadata* metadata = reinterpret_cast<ovms::ServableMetadata*>(servableMetadata);
    *count = metadata->getOutputsInfo().size();
    return nullptr;
}
OVMS_Status* OVMS_ServableMetadataGetInput(OVMS_ServableMetadata* servableMetadata, uint32_t id, const char** name, OVMS_DataType* datatype, size_t* dimCount, int64_t** shapeMin, int64_t** shapeMax) {
    if (servableMetadata == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable metadata"));
    }
    if (name == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "input name"));
    }
    if (datatype == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data type"));
    }
    if (dimCount == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "dimension count"));
    }
    if (shapeMin == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "shape min array"));
    }
    if (shapeMax == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "shape max array"));
    }
    ovms::ServableMetadata* metadata = reinterpret_cast<ovms::ServableMetadata*>(servableMetadata);
    if (id >= metadata->getInputsInfo().size()) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_TENSOR));
    }
    auto it = std::next(metadata->getInputsInfo().begin(), id);
    *name = it->second->getName().c_str();
    *datatype = getPrecisionAsOVMSDataType(it->second->getPrecision());
    *dimCount = metadata->getInputDimsMin().at(*name).size();
    *shapeMin = const_cast<int64_t*>(metadata->getInputDimsMin().at(*name).data());
    *shapeMax = const_cast<int64_t*>(metadata->getInputDimsMax().at(*name).data());
    if (spdlog::default_logger_raw()->level() == spdlog::level::trace) {
        std::stringstream ss;
        ss << "C-API request input metadata for servable: " << metadata->getName()
           << " version: " << metadata->getVersion()
           << " name: " << *name
           << " datatype: " << toString(ovms::getOVMSDataTypeAsPrecision(*datatype))
           << " shape min: [";
        size_t i = 0;
        for (i = 0; i < *dimCount - 1; ++i) {
            ss << (*shapeMin)[i] << ", ";
        }
        ss << (*shapeMin)[i] << "]"
           << " shape max: [";
        i = 0;
        for (i = 0; i < *dimCount - 1; ++i) {
            ss << (*shapeMax)[i] << ", ";
        }
        ss << (*shapeMax)[i] << "]";
        SPDLOG_TRACE(ss.str());
    }
    return nullptr;
}

OVMS_Status* OVMS_ServableMetadataGetOutput(OVMS_ServableMetadata* servableMetadata, uint32_t id, const char** name, OVMS_DataType* datatype, size_t* dimCount, int64_t** shapeMin, int64_t** shapeMax) {
    if (servableMetadata == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable metadata"));
    }
    if (name == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "output name"));
    }
    if (datatype == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "data type"));
    }
    if (dimCount == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "dimension count"));
    }
    if (shapeMin == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "shape min array"));
    }
    if (shapeMax == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "shape max array"));
    }
    ovms::ServableMetadata* metadata = reinterpret_cast<ovms::ServableMetadata*>(servableMetadata);
    if (id >= metadata->getOutputsInfo().size()) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_TENSOR));
    }
    auto it = std::next(metadata->getOutputsInfo().begin(), id);
    *name = it->second->getName().c_str();
    *datatype = getPrecisionAsOVMSDataType(it->second->getPrecision());
    *dimCount = metadata->getOutputDimsMin().at(*name).size();
    *shapeMin = const_cast<int64_t*>(metadata->getOutputDimsMin().at(*name).data());
    *shapeMax = const_cast<int64_t*>(metadata->getOutputDimsMax().at(*name).data());
    if (spdlog::default_logger_raw()->level() == spdlog::level::trace) {
        std::stringstream ss;
        ss << "C-API request output metadata for servable: " << metadata->getName()
           << " version: " << metadata->getVersion()
           << " name: " << *name
           << " datatype: " << toString(ovms::getOVMSDataTypeAsPrecision(*datatype))
           << " shape min: [";
        size_t i = 0;
        for (i = 0; i < *dimCount - 1; ++i) {
            ss << (*shapeMin)[i] << ", ";
        }
        ss << (*shapeMin)[i] << "]"
           << " shape max: [";
        i = 0;
        for (i = 0; i < *dimCount - 1; ++i) {
            ss << (*shapeMax)[i] << ", ";
        }
        ss << (*shapeMax)[i] << "]";
        SPDLOG_TRACE(ss.str());
    }
    return nullptr;
}

OVMS_Status* OVMS_ServableMetadataGetInfo(OVMS_ServableMetadata* servableMetadata, const void** info) {
    if (servableMetadata == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "servable metadata"));
    }
    if (info == nullptr) {
        return reinterpret_cast<OVMS_Status*>(new Status(StatusCode::NONEXISTENT_PTR, "info"));
    }
    ovms::ServableMetadata* metadata = reinterpret_cast<ovms::ServableMetadata*>(servableMetadata);
    *info = const_cast<void*>(reinterpret_cast<const void*>(&(metadata->getInfo())));
    return nullptr;
}

void OVMS_ServableMetadataDelete(OVMS_ServableMetadata* metadata) {
    if (metadata == nullptr)
        return;
    delete reinterpret_cast<ovms::ServableMetadata*>(metadata);
}

#ifdef __cplusplus
}
#endif
