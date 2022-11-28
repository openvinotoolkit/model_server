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
#include <string>

#include "buffer.hpp"
#include "inferenceparameter.hpp"
#include "inferencerequest.hpp"
#include "inferenceresponse.hpp"
#include "inferencetensor.hpp"
#include "status.hpp"
#include "api_options.hpp"
#include "server.hpp"

using ovms::Buffer;
using ovms::InferenceParameter;
using ovms::InferenceRequest;
using ovms::InferenceResponse;
using ovms::InferenceTensor;
using ovms::Status;
using ovms::StatusCode;

OVMS_Status* OVMS_ServerGeneralOptionsNew(OVMS_ServerGeneralOptions** options) {
    *options = reinterpret_cast<OVMS_ServerGeneralOptions*>(new ovms::GeneralOptionsImpl);
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsDelete(OVMS_ServerGeneralOptions* options) {
    delete reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsNew(OVMS_ServerMultiModelOptions** options) {
    *options = reinterpret_cast<OVMS_ServerMultiModelOptions*>(new ovms::MultiModelOptionsImpl);
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsDelete(OVMS_ServerMultiModelOptions* options) {
    delete reinterpret_cast<ovms::MultiModelOptionsImpl*>(options);
    return 0;
}

OVMS_Status* OVMS_ServerNew(OVMS_Server** server) {
    // Create new server once multi server configuration becomes possible.
    *server = reinterpret_cast<OVMS_Server*>(&ovms::Server::instance());
    return 0;
}

OVMS_Status* OVMS_ServerDelete(OVMS_Server* server) {
    // Make use of the server pointer instead of singleton once multi server configuration becomes possible.
    ovms::Server::instance().shutdownModules();
    return 0;
}

OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerMultiModelOptions* multi_model_specific_options) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(general_options);
    ovms::MultiModelOptionsImpl* mmo = reinterpret_cast<ovms::MultiModelOptionsImpl*>(multi_model_specific_options);
    std::int64_t res = ovms::Server::instance().start(go, mmo);
    return (OVMS_Status*)res;  // TODO: Return proper OVMS_Status instead of a raw status code in error handling PR
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcPort(OVMS_ServerGeneralOptions* options,
    uint32_t grpcPort) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcPort = grpcPort;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestPort(OVMS_ServerGeneralOptions* options,
    uint32_t restPort) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->restPort = restPort;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcWorkers(OVMS_ServerGeneralOptions* options,
    uint32_t grpc_workers) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcWorkers = grpc_workers;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcBindAddress(OVMS_ServerGeneralOptions* options,
    const char* grpc_bind_address) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcBindAddress = std::string(grpc_bind_address);
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestWorkers(OVMS_ServerGeneralOptions* options,
    uint32_t rest_workers) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->restWorkers = rest_workers;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestBindAddress(OVMS_ServerGeneralOptions* options,
    const char* rest_bind_address) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->restBindAddress = std::string(rest_bind_address);
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcChannelArguments(OVMS_ServerGeneralOptions* options,
    const char* grpc_channel_arguments) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->grpcChannelArguments = std::string(grpc_channel_arguments);
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetFileSystemPollWaitSeconds(OVMS_ServerGeneralOptions* options,
    uint32_t file_system_poll_wait_seconds) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->filesystemPollWaitSeconds = file_system_poll_wait_seconds;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetSequenceCleanerPollWaitMinutes(OVMS_ServerGeneralOptions* options,
    uint32_t sequence_cleaner_poll_wait_minutes) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->sequenceCleanerPollWaitMinutes = sequence_cleaner_poll_wait_minutes;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetCustomNodeResourcesCleanerInterval(OVMS_ServerGeneralOptions* options,
    uint32_t custom_node_resources_cleaner_interval) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->resourcesCleanerPollWaitSeconds = custom_node_resources_cleaner_interval;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetCpuExtensionPath(OVMS_ServerGeneralOptions* options,
    const char* cpu_extension_path) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->cpuExtensionLibraryPath = std::string(cpu_extension_path);
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetCacheDir(OVMS_ServerGeneralOptions* options,
    const char* cache_dir) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->cacheDir = std::string(cache_dir);
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetLogLevel(OVMS_ServerGeneralOptions* options,
    OVMS_LogLevel log_level) {
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
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetLogPath(OVMS_ServerGeneralOptions* options,
    const char* log_path) {
    ovms::GeneralOptionsImpl* go = reinterpret_cast<ovms::GeneralOptionsImpl*>(options);
    go->logPath = std::string(log_path);
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsSetConfigPath(OVMS_ServerMultiModelOptions* options,
    const char* config_path) {
    ovms::MultiModelOptionsImpl* mmo = reinterpret_cast<ovms::MultiModelOptionsImpl*>(options);
    mmo->configPath = std::string(config_path);
    return 0;
}
// inference API
OVMS_Status* OVMS_InferenceRequestNew(OVMS_InferenceRequest** request, const char* servableName, uint32_t servableVersion) {
    // TODO should we allow to create requests to not yet loaded models?
    *request = reinterpret_cast<OVMS_InferenceRequest*>(new InferenceRequest(servableName, servableVersion));
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestDelete(OVMS_InferenceRequest* request) {
    delete reinterpret_cast<InferenceRequest*>(request);
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestAddInput(OVMS_InferenceRequest* req, const char* inputName, OVMS_DataType datatype, const uint64_t* shape, uint32_t dimCount) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->addInput(inputName, datatype, shape, dimCount);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* req, const char* inputName, void* data, size_t bufferSize, BufferType bufferType, uint32_t deviceId) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->setInputBuffer(inputName, data, bufferSize, bufferType, deviceId);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestAddParameter(OVMS_InferenceRequest* req, const char* parameterName, OVMS_DataType datatype, const void* data, size_t byteSize) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->addParameter(parameterName, datatype, data);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestRemoveParameter(OVMS_InferenceRequest* req, const char* parameterName) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->removeParameter(parameterName);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestRemoveInput(OVMS_InferenceRequest* req, const char* inputName) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->removeInput(inputName);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceRequestInputRemoveData(OVMS_InferenceRequest* req, const char* inputName) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceRequest* request = reinterpret_cast<InferenceRequest*>(req);
    auto status = request->removeInputBuffer(inputName);
    if (!status.ok()) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetOutput(OVMS_InferenceResponse* res, uint32_t id, const char** name, OVMS_DataType* datatype, const uint64_t** shape, uint32_t* dimCount, void** data, size_t* bytesize, BufferType* bufferType, uint32_t* deviceId) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    InferenceTensor* tensor = nullptr;
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
    *data = const_cast<void*>(buffer->data());  // should data return const ptr?
    *bytesize = buffer->getByteSize();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetOutputCount(OVMS_InferenceResponse* res, uint32_t* count) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    *count = response->getOutputCount();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetParameterCount(OVMS_InferenceResponse* res, uint32_t* count) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    *count = response->getParameterCount();
    return nullptr;
}

OVMS_Status* OVMS_InferenceResponseGetParameter(OVMS_InferenceResponse* res, uint32_t id, OVMS_DataType* datatype, const void** data) {
    // TODO error handling if null
    // if (nullptr == req)
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
    // TODO error handling if null
    // if (nullptr == req)
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    delete response;
    return nullptr;
}
