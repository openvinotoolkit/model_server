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
#include "stdint.h"  //  For precise data types

struct OVMS_Server;

struct OVMS_ServerGeneralOptions;

typedef enum OVMSSERVER_loglevel_enum {
  OVMSSERVER_LOG_TRACE,
  OVMSSERVER_LOG_DEBUG,
  OVMSSERVER_LOG_INFO,
  OVMSSERVER_LOG_WARNING,
  OVMSSERVER_LOG_ERROR
} OVMSSERVER_LogLevel;

// TODO: The functions should return OVMS_Status or OVMS_Error

////
//// OVMS_ServerGeneralOptions
//// Structure for general options for both: single and multi (with config.json) management
////
// Allocates memory for server general options and returns ptr
void OVMS_ServerGeneralOptionsNew(OVMS_ServerGeneralOptions** options);
// Deallocates server general options memory for given ptr
void OVMS_ServerGeneralOptionsDelete(OVMS_ServerGeneralOptions* options);

// --log_level
void OVMS_ServerGeneralOptionsSetLogLevel(OVMS_ServerGeneralOptions* options,
    OVMSSERVER_LogLevel log_level);

// --log_path
void OVMS_ServerGeneralOptionsSetLogPath(OVMS_ServerGeneralOptions* options,
    const char* log_path);

// --file_system_poll_wait_seconds
void OVMS_ServerGeneralOptionsSetFileSystemPollWaitSeconds(OVMS_ServerGeneralOptions* options,
    uint64_t file_system_poll_wait_seconds);

// --sequence_cleaner_poll_wait_minutes
void OVMS_ServerGeneralOptionsSetSequenceCleanerPollWaitMinutes(OVMS_ServerGeneralOptions* options,
    uint64_t sequence_cleaner_poll_wait_minutes);

// --custom_node_resources_cleaner_interval
void OVMS_ServerGeneralOptionsSetCustomNodeResourcesCleanerInterval(OVMS_ServerGeneralOptions* options,
    uint64_t custom_node_resources_cleaner_interval);  // TODO: Should include seconds or minutes in the name

// --cache_dir
void OVMS_ServerGeneralOptionsSetCacheDir(OVMS_ServerGeneralOptions* options,
    const char* cache_dir);

// --cpu_extension
void OVMS_ServerGeneralOptionsSetCpuExtensionPath(OVMS_ServerGeneralOptions* options,
    const char* cpu_extension_path);


////
//// OVMS_ServerMultiModelOptions
//// Options for starting multi model server controlled by config.json file
////
// Allocates memory for multi model server options and returns ptr
void OVMS_ServerMultiModelOptionsNew(OVMS_ServerMultiModelOptions** options);
// Deallocates options memory for given ptr
void OVMS_ServerMultiModelOptionsDelete(OVMS_ServerMultiModelOptions* options);

// --config_path
void OVMS_ServerMultiModelOptionsSetConfigPath(OVMS_ServerMultiModelOptions* options,
    const char* config_path);


////
//// OVMS_Server
//// Handler for all managerment activities
////
// Allocates memory for server and returns ptr
void OVMS_ServerNew(OVMS_Server** server);
// Deallocates server memory for given ptr
void OVMS_ServerDelete(OVMS_Server* server);

// Start with configuration file config.json
// Return error if already started
void OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerMultiModelOptions* multi_model_specific_options); // in fact only --config_path
// Unload all and cleanup
// TODO: Should not be possible to re-start?
void OVMS_ServerStop(OVMS_Server* server);
