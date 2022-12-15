#pragma once
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
#include <stddef.h>
#include <stdint.h>  //  For precise data types

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OVMS_Server_ OVMS_Server;
typedef struct OVMS_Status_ OVMS_Status;

typedef struct OVMS_ServerSettings_ OVMS_ServerSettings;
typedef struct OVMS_ModelsSettings_ OVMS_ModelsSettings;

typedef enum OVMS_DataType_enum {
    OVMS_DATATYPE_BF16,
    OVMS_DATATYPE_FP64,
    OVMS_DATATYPE_FP32,
    OVMS_DATATYPE_FP16,
    OVMS_DATATYPE_I64,
    OVMS_DATATYPE_I32,
    OVMS_DATATYPE_I16,
    OVMS_DATATYPE_I8,
    OVMS_DATATYPE_I4,
    OVMS_DATATYPE_U64,
    OVMS_DATATYPE_U32,
    OVMS_DATATYPE_U16,
    OVMS_DATATYPE_U8,
    OVMS_DATATYPE_U4,
    OVMS_DATATYPE_U1,
    OVMS_DATATYPE_BOOL,
    OVMS_DATATYPE_CUSTOM,
    OVMS_DATATYPE_UNDEFINED,
    OVMS_DATATYPE_DYNAMIC,
    OVMS_DATATYPE_MIXED,
    OVMS_DATATYPE_Q78,
    OVMS_DATATYPE_BIN,
    OVMS_DATATYPE_END
} OVMS_DataType;

typedef enum OVMS_BufferType_enum {
    OVMS_BUFFERTYPE_CPU,
    OVMS_BUFFERTYPE_CPU_PINNED,
    OVMS_BUFFERTYPE_GPU,
    OVMS_BUFFERTYPE_HDDL,
} OVMS_BufferType;

typedef struct OVMS_InferenceRequest_ OVMS_InferenceRequest;
typedef struct OVMS_InferenceResponse_ OVMS_InferenceResponse;

typedef enum OVMS_LogLevel_enum {
    OVMS_LOG_TRACE,
    OVMS_LOG_DEBUG,
    OVMS_LOG_INFO,
    OVMS_LOG_WARNING,
    OVMS_LOG_ERROR
} OVMS_LogLevel;

////
//// OVMS_Status
//// Structure for status management.
//// Whenever C-API call returns non null pointer it should be treated as error with code and detailed message.
//// The status should be deallocated with OVMS_StatusDelete afterwards.
////
// Deallocates status memory for given ptr
void OVMS_StatusDelete(OVMS_Status* status);

OVMS_Status* OVMS_StatusGetCode(OVMS_Status* status,
    uint32_t* code);

OVMS_Status* OVMS_StatusGetDetails(OVMS_Status* status,
    const char** details);

////
//// OVMS_ServerSettings
//// Structure for server settings for both: single and multi (with config.json) management
////
// Allocates memory for server settings and returns ptr
OVMS_Status* OVMS_ServerSettingsNew(OVMS_ServerSettings** settings);
// Deallocates server settings memory for given ptr
void OVMS_ServerSettingsDelete(OVMS_ServerSettings* settings);

// --port
OVMS_Status* OVMS_ServerSettingsSetGrpcPort(OVMS_ServerSettings* settings,
    uint32_t grpc_port);

// --rest_port
OVMS_Status* OVMS_ServerSettingsSetRestPort(OVMS_ServerSettings* settings,
    uint32_t rest_port);

// --grpc_workers
OVMS_Status* OVMS_ServerSettingsSetGrpcWorkers(OVMS_ServerSettings* settings,
    uint32_t grpc_workers);

// --grpc_bind_address
OVMS_Status* OVMS_ServerSettingsSetGrpcBindAddress(OVMS_ServerSettings* settings,
    const char* grpc_bind_address);

// --rest_workers
OVMS_Status* OVMS_ServerSettingsSetRestWorkers(OVMS_ServerSettings* settings,
    uint32_t rest_workers);

// --rest_bind_address
OVMS_Status* OVMS_ServerSettingsSetRestBindAddress(OVMS_ServerSettings* settings,
    const char* rest_bind_address);

// --grpc_channel_arguments
OVMS_Status* OVMS_ServerSettingsSetGrpcChannelArguments(OVMS_ServerSettings* settings,
    const char* grpc_channel_arguments);

// --file_system_poll_wait_seconds
OVMS_Status* OVMS_ServerSettingsSetFileSystemPollWaitSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds);

// --sequence_cleaner_poll_wait_minutes
OVMS_Status* OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(OVMS_ServerSettings* settings,
    uint32_t minutes);

// --custom_node_resources_cleaner_interval_seconds
OVMS_Status* OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds);

// --cpu_extension
OVMS_Status* OVMS_ServerSettingsSetCpuExtensionPath(OVMS_ServerSettings* settings,
    const char* cpu_extension_path);

// --cache_dir
OVMS_Status* OVMS_ServerSettingsSetCacheDir(OVMS_ServerSettings* settings,
    const char* cache_dir);

// --log_level
OVMS_Status* OVMS_ServerSettingsSetLogLevel(OVMS_ServerSettings* settings,
    OVMS_LogLevel log_level);

// --log_path
OVMS_Status* OVMS_ServerSettingsSetLogPath(OVMS_ServerSettings* settings,
    const char* log_path);

////
//// OVMS_ModelsSettings
//// Options for starting multi model server controlled by config.json file
//// Models management settings for starting OVMS. Right now only using config.json file
//// is supported
////
// Allocates memory for models settings and returns ptr
OVMS_Status* OVMS_ModelsSettingsNew(OVMS_ModelsSettings** settings);
// Deallocates models settings memory for given ptr
void OVMS_ModelsSettingsDelete(OVMS_ModelsSettings* settings);

// --config_path
OVMS_Status* OVMS_ModelsSettingsSetConfigPath(OVMS_ModelsSettings* settings,
    const char* config_path);

////
//// OVMS_Server
//// Handler for all management activities
////
// Allocates memory for server and returns ptr
OVMS_Status* OVMS_ServerNew(OVMS_Server** server);
// Deallocates server memory for given ptr
void OVMS_ServerDelete(OVMS_Server* server);

// Start with configuration file config.json
// Return error if already started
OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerSettings* server_settings,
    OVMS_ModelsSettings* models_settings);  // in fact only --config_path
// Unload all and cleanup

// OVMS_InferenceRequest
OVMS_Status* OVMS_InferenceRequestNew(OVMS_InferenceRequest** request, OVMS_Server* server, const char* servableName, uint32_t servableVersion);
void OVMS_InferenceRequestDelete(OVMS_InferenceRequest* response);

OVMS_Status* OVMS_InferenceRequestAddInput(OVMS_InferenceRequest* request, const char* inputName, OVMS_DataType datatype, const uint64_t* shape, uint32_t dimCount);

// ownership of data needs to be maintained during inference
OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* request, const char* inputName, const void* data, size_t bufferSize, OVMS_BufferType bufferType, uint32_t deviceId);
OVMS_Status* OVMS_InferenceRequestInputRemoveData(OVMS_InferenceRequest* request, const char* inputName);
OVMS_Status* OVMS_InferenceRequestRemoveInput(OVMS_InferenceRequest* request, const char* inputName);  // this will allow for reuse of request but with different input data
OVMS_Status* OVMS_InferenceRequestAddParameter(OVMS_InferenceRequest* request, const char* parameterName, OVMS_DataType datatype, const void* data, size_t byteSize);
OVMS_Status* OVMS_InferenceRequestRemoveParameter(OVMS_InferenceRequest* request, const char* parameterName);

// OVMS_Inference Response
OVMS_Status* OVMS_InferenceResponseGetOutputCount(OVMS_InferenceResponse* response, uint32_t* count);
OVMS_Status* OVMS_InferenceResponseGetOutput(OVMS_InferenceResponse* response, uint32_t id, const char** name, OVMS_DataType* datatype, const uint64_t** shape, uint32_t* dimCount, const void** data, size_t* bytesize, OVMS_BufferType* bufferType, uint32_t* deviceId);
OVMS_Status* OVMS_InferenceResponseGetParameterCount(OVMS_InferenceResponse* response, uint32_t* count);
OVMS_Status* OVMS_InferenceResponseGetParameter(OVMS_InferenceResponse* response, uint32_t id, OVMS_DataType* datatype, const void** data);
void OVMS_InferenceResponseDelete(OVMS_InferenceResponse* response);

OVMS_Status* OVMS_Inference(OVMS_Server* server, OVMS_InferenceRequest* request, OVMS_InferenceResponse** response);

#ifdef __cplusplus
}
#endif
