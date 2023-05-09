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

typedef struct OVMS_ServableMetadata_ OVMS_ServableMetadata;

#define OVMS_API_VERSION_MAJOR 0
#define OVMS_API_VERSION_MINOR 3

// Function to retrieve OVMS API version.
//
// \param major Returns major version of OVMS API. Represents breaking, non-backward compatible API changes.
// \param minor Returns minor version of OVMS API. Represents non-breaking, backward compatible API changes.
OVMS_Status* OVMS_ApiVersion(uint32_t* major, uint32_t* minor);

// OVMS_DataType
//
// Tensor and parameter data types recognized by OVMS.
//
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

// OVMS_BufferType
//
// Types of memory used by OVMS.
//
typedef enum OVMS_BufferType_enum {
    OVMS_BUFFERTYPE_CPU,
    OVMS_BUFFERTYPE_CPU_PINNED,
    OVMS_BUFFERTYPE_GPU,
    OVMS_BUFFERTYPE_HDDL,
} OVMS_BufferType;

typedef struct OVMS_InferenceRequest_ OVMS_InferenceRequest;
typedef struct OVMS_InferenceResponse_ OVMS_InferenceResponse;

// OVMS_LogLevel
//
// Levels of OVMS logging.
//
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
//// Whenever C-API call returns non null pointer it should be treated as error with code and string message.
//// The status should be deallocated with OVMS_StatusDelete afterwards.
////
// Deallocates a status object.
//
//  \param status The status object
void OVMS_StatusDelete(OVMS_Status* status);

// Get the status code from a status.
//
// \param status The status object
// \param code Value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_StatusGetCode(OVMS_Status* status,
    uint32_t* code);

// Get the status details from a status.
//
// \param status The status object
// \param details The status details
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_StatusGetDetails(OVMS_Status* status,
    const char** details);

////
//// OVMS_ServerSettings
//// Structure for server settings for both: single and multi (with config.json) management.
////
// Allocates memory for server settings and returns ptr.
//
// \param settings The server settings object to be created
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsNew(OVMS_ServerSettings** settings);

// Deallocates server settings object for given ptr.
//
// \param settings The settings object to be removed
void OVMS_ServerSettingsDelete(OVMS_ServerSettings* settings);

// Set the gRPC port of starting OVMS. Equivalent of using --port parameter from OVMS CLI.
// If not set server will start with gRPC port set to 9178.
//
// \param settings The server settings object to be set
// \param grpc_port The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetGrpcPort(OVMS_ServerSettings* settings,
    uint32_t grpc_port);

// Set rest port for starting server. If not set the http server will not start
// Equivalent of starting server with
// --rest_port.
//
// \param settings The server settings object to be set
// \param rest_port The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetRestPort(OVMS_ServerSettings* settings,
    uint32_t rest_port);

// Set gRPC workers server setting.
// Equivalent of starting server with
// --grpc_workers.
//
// \param settings The server settings object to be set
// \param grpc_workers The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetGrpcWorkers(OVMS_ServerSettings* settings,
    uint32_t grpc_workers);

// Set gRPC bind address for starting server
// Equivalent of starting server with
// --grpc_bind_address.
//
// \param settings The server settings object to be set
// \param grpc_bind_address The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetGrpcBindAddress(OVMS_ServerSettings* settings,
    const char* grpc_bind_address);

// Set REST workers server setting.
// Equivalent of starting server with
// --rest_workers.
//
// \param settings The server settings object to be set
// \param rest_workers The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetRestWorkers(OVMS_ServerSettings* settings,
    uint32_t rest_workers);

// Set REST bind address server setting.
// Equivalent of starting server with
// --rest_bind_address.
//
// \param settings The server settings object to be set
// \param rest_bind_address The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetRestBindAddress(OVMS_ServerSettings* settings,
    const char* rest_bind_address);

// Set the gRPC channel arguments server setting.
// Equivalent of starting server with
// --grpc_channel_arguments.
//
// \param settings The server settings object to be set
// \param grpc_channel_arguments The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetGrpcChannelArguments(OVMS_ServerSettings* settings,
    const char* grpc_channel_arguments);

// Set config check interval server setting.
// Equivalent of starting server with
// --file_system_poll_wait_seconds.
//
// \param settings The server settings object to be set
// \param seconds The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetFileSystemPollWaitSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds);

// Set sequence cleaner interval server setting.
// Equivalent of starting server with
// --sequence_cleaner_poll_wait_minutes.
//
// \param settings The server settings object to be set
// \param minutes The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(OVMS_ServerSettings* settings,
    uint32_t minutes);

// Set custom node resource cleaner interval server setting.
// Equivalent of starting server with
// --custom_node_resources_cleaner_interval_seconds.
//
// \param settings The server settings object to be set
// \param seconds The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(OVMS_ServerSettings* settings,
    uint32_t seconds);

// Set cpu extension path server setting. Equivalent of starting server with
// --cpu_extension.
//
// \param settings The server settings object to be set
// \param cpu_extension_path The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetCpuExtensionPath(OVMS_ServerSettings* settings,
    const char* cpu_extension_path);

// Set cache dir server setting. Equivalent of starting server with
// --cache_dir.
//
// \param settings The server settings object to be set
// \param cache_dir The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetCacheDir(OVMS_ServerSettings* settings,
    const char* cache_dir);

// Set log level server setting. Equivalent of starting server with
// --log_level.
//
// \param settings The server settings object to be set
// \param log_level The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetLogLevel(OVMS_ServerSettings* settings,
    OVMS_LogLevel log_level);

// Set the server log_path setting. Equivalent of starting server with
// --log_path.
//
// \param settings The server settings object to be set
// \param log_path The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerSettingsSetLogPath(OVMS_ServerSettings* settings,
    const char* log_path);

////
//// OVMS_ModelsSettings
//// Options for starting multi model server controlled by config.json file
//// Models management settings for starting OVMS. Right now only using config.json file
//// is supported.
////
// Allocates memory for models settings and returns ptr.
//
// \param settings The models settings object to be created
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ModelsSettingsNew(OVMS_ModelsSettings** settings);

// Deallocates models settings memory for given ptr.
//
// \param settings The models settings object to be removed
void OVMS_ModelsSettingsDelete(OVMS_ModelsSettings* settings);

// Set the server configuration file path. Equivalent of starting server with
// --config_path.
//
// \param settings The models settings object to be set
// \param config_path The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ModelsSettingsSetConfigPath(OVMS_ModelsSettings* settings,
    const char* config_path);

////
//// OVMS_Server
//// Handler for all management activities.
////
// Allocates memory for server and returns ptr
//
// \param server The server object to be created and set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerNew(OVMS_Server** server);

// Deallocates server memory for given ptr.
//
// \param server The server object to be removed
void OVMS_ServerDelete(OVMS_Server* server);

// Start server with configuration file config.json.
// Return error if already started or any other loading, configuration error occured.
// In preview only using config file is supported, providing model name and model path is not.
//
// \param server The server object to be started
// \param server_settings The server settings to be used
// \param models_settings The models settings to be used
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerSettings* server_settings,
    OVMS_ModelsSettings* models_settings);

// OVMS_InferenceRequest
//
// Create new inference request object. In case of servable version set to 0 server will choose
// the default servable version.
//
// \param request The request object to be created
// \param server The server object
// \param servableName The name of the servable to be used
// \param servableVersion The version of the servable to be used
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceRequestNew(OVMS_InferenceRequest** request, OVMS_Server* server, const char* servableName, int64_t servableVersion);
void OVMS_InferenceRequestDelete(OVMS_InferenceRequest* response);

// Set the data of the input buffer. Ownership of data needs to be maintained during inference.
//
// \param request The request object
// \param inputName The name of the input
// \param datatype The data type of the input
// \param shape The shape of the input
// \param dimCount The number of dimensions of the shape
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceRequestAddInput(OVMS_InferenceRequest* request, const char* inputName, OVMS_DataType datatype, const int64_t* shape, size_t dimCount);

// Set the data of the input buffer. Ownership of data needs to be maintained during inference.
//
// \param request The request object
// \param inputName The name of the input with data to be set
// \param data The data of the input
// \param byteSize The byte size of the data
// \param bufferType The buffer type of the data
// \param deviceId The device id of the data memory buffer
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* request, const char* inputName, const void* data, size_t byteSize, OVMS_BufferType bufferType, uint32_t deviceId);

// Remove the data of the input.
//
// \param request The request object
// \param inputName The name of the input with data to be removed
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceRequestInputRemoveData(OVMS_InferenceRequest* request, const char* inputName);

// Remove input from the request.
//
// \param request The request object
// \param inputName The name of the input to be removed
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceRequestRemoveInput(OVMS_InferenceRequest* request, const char* inputName);

// Add parameter to the request.
//
// \param request The request object
// \param parameterName The name of the parameter to be added
// \param datatype The request object
// \param data The data representing parameter value
// \param byteSize The byte size of the added parameter value
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceRequestAddParameter(OVMS_InferenceRequest* request, const char* parameterName, OVMS_DataType datatype, const void* data, size_t byteSize);

// Remove parameter from the inference request.
//
// \param request The request object
// \param parameterName The name of the parameter to be removed
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceRequestRemoveParameter(OVMS_InferenceRequest* request, const char* parameterName);

// OVMS_InferenceResponse
//
// Get the number of outputs in the response.
//
// \param response The response object
// \param count The value to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceResponseGetOutputCount(OVMS_InferenceResponse* response, uint32_t* count);

// Get all information about an output from the response by providing output id.
//
// \param response The response object
// \param id The id of the output
// \param name The name of the output
// \param datatype The data type of the output
// \param shape The shape of the output
// \param dimCount The number of dimensions of the shape
// \param data The data of the output
// \param byteSize The buffer size of the data
// \param bufferType The buffer type of the data
// \param deviceId The device id of the data memory buffer
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceResponseGetOutput(OVMS_InferenceResponse* response, uint32_t id, const char** name, OVMS_DataType* datatype, const int64_t** shape, size_t* dimCount, const void** data, size_t* byteSize, OVMS_BufferType* bufferType, uint32_t* deviceId);

// Get the number of parameters in response.
//
// \param response The response object
// \param count The parameter count to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceResponseGetParameterCount(OVMS_InferenceResponse* response, uint32_t* count);

// Extract information about parameter by providing its id.
//
// \param response The response object
// \param id The id of the parameter
// \param datatype The data type of the parameter
// \param data The parameter content
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_InferenceResponseGetParameter(OVMS_InferenceResponse* response, uint32_t id, OVMS_DataType* datatype, const void** data);

// Delete OVMS_InferenceResponse object.
//
// \param response The response object to be removed
void OVMS_InferenceResponseDelete(OVMS_InferenceResponse* response);

// Execute synchronous inference.
//
// \param server The server object
// \param request The request object
// \param response The respons object. In case of success, caller takes the ownership of the response
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_Inference(OVMS_Server* server, OVMS_InferenceRequest* request, OVMS_InferenceResponse** response);

// Get OVMS_ServableMetadata object
//
// Creates OVMS_ServableMetadata object describing inputs and outputs.
// Returned object needs to be deleted after use with OVMS_ServableMetadataDelete
// if call succeeded.
//
// \param server The server object
// \param servableName The name of the servable to be used
// \param servableVersion The version of the servable to be used
// \param metadata The metadata object to be created
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_GetServableMetadata(OVMS_Server* server, const char* servableName, int64_t servableVersion, OVMS_ServableMetadata** metadata);

// Get the number of inputs of servable.
//
// \param metadata The metadata object
// \param count The parameter count to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServableMetadataGetInputCount(OVMS_ServableMetadata* metadata, uint32_t* count);

// Get the number of outputs of servable.
//
// \param metadata The metadata object
// \param count The parameter count to be set
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServableMetadataGetOutputCount(OVMS_ServableMetadata* metadata, uint32_t* count);

// Get the metadata of servable input given the index
//
// The received shapeMin and shapeMax indicate whether the underlying servable accepts
// a shape range or fully dynamic shape. A value of -1 for both shapeMin and shapeMax
// for a specific dimension means that the servable accepts any value on that dimension.
//
// \param metadata The metadata object
// \param id The id of the input
// \param name The name of the input
// \param datatype The data type of the input
// \param dimCount The number of dimensions of the shape
// \param shapeMin The shape lower bounds of the input
// \param shapeMax The shape upper bounds of the input
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServableMetadataGetInput(OVMS_ServableMetadata* metadata, uint32_t id, const char** name, OVMS_DataType* datatype, size_t* dimCount, int64_t** shapeMinArray, int64_t** shapeMaxArray);

// Get the metadata of servable output given the index
//
// The received shapeMin and shapeMax indicate whether the underlying servable accepts
// a shape range or fully dynamic shape. A value of -1 for both shapeMin and shapeMax
// for a specific dimension means that the servable accepts any value on that dimension.
//
// \param metadata The metadata object
// \param id The id of the output
// \param name The name of the output
// \param datatype The data type of the output
// \param dimCount The number of dimensions of the shape
// \param shapeMin The shape of the output
// \param shapeMax The shape of the output
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServableMetadataGetOutput(OVMS_ServableMetadata* metadata, uint32_t id, const char** name, OVMS_DataType* datatype, size_t* dimCount, int64_t** shapeMinArray, int64_t** shapeMaxArray);


// EXPERIMENTAL // TODO if declare specific type for underlying ov::AnyMap
// Get the additional info about servable.
//
// \param metadata The metadata object
// \param info The ptr to the ov::AnyMap*
// \return OVMS_Status object in case of failure
OVMS_Status* OVMS_ServableMetadataGetInfo(OVMS_ServableMetadata* metadata, const void** info);

// Deallocates a status object.
//
//  \param metadata The metadata object
void OVMS_ServableMetadataDelete(OVMS_ServableMetadata* metadata);

#ifdef __cplusplus
}
#endif
