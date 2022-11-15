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

struct OVMS_Server;
struct OVMS_Status;

struct OVMS_ServerGeneralOptions;
struct OVMS_ServerMultiModelOptions;

// TODO reuse this in precision.hpp
enum DataType {
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
};

enum BufferType {
    OVMS_BUFFERTYPE_CPU,
    OVMS_BUFFERTYPE_CPU_PINNED,
    OVMS_BUFFERTYPE_GPU,
    OVMS_BUFFERTYPE_HDDL,
};

struct OVMS_Status;
struct OVMS_InferenceRequest;
struct OVMS_InferenceResponse;

typedef enum OVMSSERVER_loglevel_enum {
    OVMSSERVER_LOG_TRACE,
    OVMSSERVER_LOG_DEBUG,
    OVMSSERVER_LOG_INFO,
    OVMSSERVER_LOG_WARNING,
    OVMSSERVER_LOG_ERROR
} OVMSSERVER_LogLevel;

////
//// OVMS_ServerGeneralOptions
//// Structure for general options for both: single and multi (with config.json) management
////
// Allocates memory for server general options and returns ptr
OVMS_Status* OVMS_ServerGeneralOptionsNew(OVMS_ServerGeneralOptions** options);
// Deallocates server general options memory for given ptr
OVMS_Status* OVMS_ServerGeneralOptionsDelete(OVMS_ServerGeneralOptions* options);

// --port
OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcPort(OVMS_ServerGeneralOptions* options,
    uint64_t grpcPort);

// --rest_port
OVMS_Status* OVMS_ServerGeneralOptionsSetRestPort(OVMS_ServerGeneralOptions* options,
    uint64_t restPort);

// --log_level
OVMS_Status* OVMS_ServerGeneralOptionsSetLogLevel(OVMS_ServerGeneralOptions* options,
    OVMSSERVER_LogLevel log_level);

// --log_path
OVMS_Status* OVMS_ServerGeneralOptionsSetLogPath(OVMS_ServerGeneralOptions* options,
    const char* log_path);

// --file_system_poll_wait_seconds
OVMS_Status* OVMS_ServerGeneralOptionsSetFileSystemPollWaitSeconds(OVMS_ServerGeneralOptions* options,
    uint64_t file_system_poll_wait_seconds);

// --sequence_cleaner_poll_wait_minutes
OVMS_Status* OVMS_ServerGeneralOptionsSetSequenceCleanerPollWaitMinutes(OVMS_ServerGeneralOptions* options,
    uint64_t sequence_cleaner_poll_wait_minutes);

// --custom_node_resources_cleaner_interval
OVMS_Status* OVMS_ServerGeneralOptionsSetCustomNodeResourcesCleanerInterval(OVMS_ServerGeneralOptions* options,
    uint64_t custom_node_resources_cleaner_interval);  // TODO: Should include seconds or minutes in the name

// --cache_dir
void OVMS_ServerGeneralOptionsSetCacheDir(OVMS_ServerGeneralOptions* options,
    const char* cache_dir);

// --cpu_extension
OVMS_Status* OVMS_ServerGeneralOptionsSetCpuExtensionPath(OVMS_ServerGeneralOptions* options,
    const char* cpu_extension_path);

////
//// OVMS_ServerMultiModelOptions
//// Options for starting multi model server controlled by config.json file
////
// Allocates memory for multi model server options and returns ptr
OVMS_Status* OVMS_ServerMultiModelOptionsNew(OVMS_ServerMultiModelOptions** options);
// Deallocates options memory for given ptr
OVMS_Status* OVMS_ServerMultiModelOptionsDelete(OVMS_ServerMultiModelOptions* options);

// --config_path
OVMS_Status* OVMS_ServerMultiModelOptionsSetConfigPath(OVMS_ServerMultiModelOptions* options,
    const char* config_path);

////
//// OVMS_Server
//// Handler for all management activities
////
// Allocates memory for server and returns ptr
OVMS_Status* OVMS_ServerNew(OVMS_Server** server);
// Deallocates server memory for given ptr
OVMS_Status* OVMS_ServerDelete(OVMS_Server* server);

// Start with configuration file config.json
// Return error if already started
OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerMultiModelOptions* multi_model_specific_options);  // in fact only --config_path
// Unload all and cleanup
// TODO: Should not be possible to re-start?
OVMS_Status* OVMS_ServerStop(OVMS_Server* server);

// OVMS_InferenceRequest
OVMS_Status* OVMS_InferenceRequestNew(char* modelName, uint32_t servableVersion);
OVMS_Status* OVMS_InferenceRequestDelete(OVMS_InferenceRequest* response);
OVMS_Status* OVMS_InferenceRequestAddInput(OVMS_InferenceRequest* request, char* inputName, DataType datatype, uint64_t* shape, uint32_t dimCount);
OVMS_Status* OVMS_InferenceRequestAddInputRaw(OVMS_InferenceRequest* request, char* inputName, DataType datatype);  // TODO consider no datatype & handle the parameters
// ownership of data needs to be maintained during inference
OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* request, char* inputName, void* data, size_t bufferSize, BufferType bufferType, uint32_t deviceId);
OVMS_Status* OVMS_InferenceRequestInputRemoveData(OVMS_InferenceRequest* request, char* inputName);
OVMS_Status* OVMS_InferenceRequestRemoveInput(OVMS_InferenceRequest* request, char* inputName);  // this will allow for reuse of request but with different input data
OVMS_Status* OVMS_InferenceRequestRemoveAllInputs(OVMS_InferenceRequest* request);
OVMS_Status* OVMS_InferenceRequestAddRequestedOutput(OVMS_InferenceRequest* request, char* inputName);  // TODO consider the other way around - add not usefull outputs
OVMS_Status* OVMS_InferenceRequestAddParameter(OVMS_InferenceRequest* request, char* paramaterName, DataType datatype, void* data, size_t byteSize);

// OVMS_Inference Response
OVMS_Status* OVMS_InferenceResponseGetOutputCount(OVMS_InferenceResponse* response, uint32_t* count);
OVMS_Status* OVMS_InferenceResponseOutput(OVMS_InferenceResponse* response, uint32_t id, char* name, DataType* datatype, uint64_t* shape, uint32_t dimCount, BufferType* bufferType, uint32_t* deviceId, void** data);
OVMS_Status* OVMS_InferenceResponseDelete(OVMS_InferenceResponse* response);
OVMS_Status* OVMS_InferenceResponseGetParameterCount(OVMS_InferenceResponse* response, uint32_t* count);
OVMS_Status* OVMS_InferenceResponseGetParameter(OVMS_InferenceResponse* response, uint32_t id, DataType* datatype, void** data);

OVMS_Status* OVMS_Inference(OVMS_InferenceRequest* request, OVMS_InferenceResponse** response);

// POCAPI to be removed
int OVMS_Start(int argc, char** argv);
void OVMS_Infer(char* name, float* data, float* output);
