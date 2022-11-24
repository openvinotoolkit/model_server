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
#include "poc_api_impl.hpp"
#include "status.hpp"

using ovms::Buffer;
using ovms::InferenceParameter;
using ovms::InferenceRequest;
using ovms::InferenceResponse;
using ovms::InferenceTensor;
using ovms::Status;
using ovms::StatusCode;

OVMS_Status* OVMS_ServerGeneralOptionsNew(OVMS_ServerGeneralOptions** options) {
    *options = (OVMS_ServerGeneralOptions*)new ovms::GeneralOptionsImpl;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsDelete(OVMS_ServerGeneralOptions* options) {
    delete (ovms::GeneralOptionsImpl*)options;
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsNew(OVMS_ServerMultiModelOptions** options) {
    *options = (OVMS_ServerMultiModelOptions*)new ovms::MultiModelOptionsImpl;
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsDelete(OVMS_ServerMultiModelOptions* options) {
    delete (ovms::MultiModelOptionsImpl*)options;
    return 0;
}

OVMS_Status* OVMS_ServerNew(OVMS_Server** server) {
    *server = (OVMS_Server*)new ovms::ServerImpl;
    return 0;
}

OVMS_Status* OVMS_ServerDelete(OVMS_Server* server) {
    delete (ovms::ServerImpl*)server;
    return 0;
}

OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerMultiModelOptions* multi_model_specific_options) {
    ovms::ServerImpl* srv = (ovms::ServerImpl*)server;
    ovms::GeneralOptionsImpl* go = (ovms::GeneralOptionsImpl*)general_options;
    ovms::MultiModelOptionsImpl* mmo = (ovms::MultiModelOptionsImpl*)multi_model_specific_options;
    std::int64_t res = srv->start(go, mmo);
    return (OVMS_Status*)res;  // TODO: Return proper OVMS_Status instead of a raw status code
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcPort(OVMS_ServerGeneralOptions* options,
    uint64_t grpcPort) {
    ovms::GeneralOptionsImpl* go = (ovms::GeneralOptionsImpl*)options;
    go->grpcPort = grpcPort;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestPort(OVMS_ServerGeneralOptions* options,
    uint64_t restPort) {
    ovms::GeneralOptionsImpl* go = (ovms::GeneralOptionsImpl*)options;
    go->restPort = restPort;
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsSetConfigPath(OVMS_ServerMultiModelOptions* options,
    const char* config_path) {
    ovms::MultiModelOptionsImpl* mmo = (ovms::MultiModelOptionsImpl*)options;
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
OVMS_Status* OVMS_InferenceResponseGetOutput(OVMS_InferenceResponse* res, uint32_t id, const char** name, OVMS_DataType* datatype, const uint64_t** shape, uint32_t* dimCount, BufferType* bufferType, uint32_t* deviceId, void** data) {
    // TODO error handling if null
    // if (nullptr == req)
    InferenceResponse* response = reinterpret_cast<InferenceResponse*>(res);
    InferenceTensor* tensor = nullptr;
    const std::string* cppName;
    auto status = response->getOutput(id, &cppName, &tensor);
    if (!status.ok() ||
        (tensor == nullptr) ||
        (cppName == nullptr)) {
        return reinterpret_cast<OVMS_Status*>(new Status(status));
    }
    const Buffer* buffer = tensor->getBuffer();
    if (nullptr == buffer) {
        return reinterpret_cast<OVMS_Status*>(new Status(ovms::StatusCode::NOT_IMPLEMENTED));  // TODO retcode
    }
    *name = cppName->c_str();
    *datatype = tensor->getDataType();
    *shape = tensor->getShape().data();
    *dimCount = tensor->getShape().size();
    *bufferType = buffer->getBufferType();
    *deviceId = buffer->getDeviceId().value_or(0);  // TODO how discriminate betwen undefined & actual device 0
    // possibly it is not neccessary to discriminate
    *data = const_cast<void*>(buffer->data());  // should data return const ptr?
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
