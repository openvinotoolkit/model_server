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
struct OVMS_ServerMultiModelOptions;    

struct OVMS_ServerMetricOption;         // TODO: This will might be out of scope for POC, multi model would be used instead.
struct OVMS_ServerPluginConfigOption;   // TODO: This will might be out of scope for POC, multi model would be used instead.
struct OVMS_ServerSingleModelOptions;   // TODO: This will might be out of scope for POC, multi model would be used instead.

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
//// OVMS_ServerPluginConfigOption (Optional for POC)
//// Structure for plugin config map
////
// Allocates memory for plugin config container and returns ptr
void OVMS_ServerPluginConfigOptionNew(OVMS_ServerPluginConfigOption** options);
// Deallocates plugin config container for given ptr
void OVMS_ServerPluginConfigOptionDelete(OVMS_ServerPluginConfigOption* options);
// Adds new option with key and value
void OVMS_ServerPluginConfigOptionAdd(OVMS_ServerPluginConfigOption* option,
    const char* key,
    const char* value);


////
//// OVMS_ServerMetricOption (Optional for POC)
//// Structure for metric configuration
////
// Allocates memory for metric options and returns ptr
void OVMS_ServerMetricOptionNew(OVMS_ServerMetricOption** option);
// Deallocates metric options for given ptr
void OVMS_ServerMetricOptionDelete(OVMS_ServerMetricOption* option);
// Enables or disables the metrics
// By default disabled
void OVMS_ServerMetricOptionEnable(OVMS_ServerMetricOption* option,
    bool enable);
// When enabled, allows switching family name on and off by adding desired names
// When not specified only default group is enabled
void OVMS_ServerMetricOptionAddFamily(OVMS_ServerMetricOption* option,
    const char* metric_family_name);  // TODO: Possibly use enum?


////
//// OVMS_ServerSingleModelOptions (Optional for POC)
//// Options for single model server without config.json file
////
// Allocates memory for single model server options and returns ptr
void OVMS_ServerSingleModelOptionsNew(OVMS_ServerSingleModelOptions** options);
// Deallocates options memory for given ptr
void OVMS_ServerSingleModelOptionsDelete(OVMS_ServerSingleModelOptions* options);

// --model_name
void OVMS_ServerSingleModelOptionsSetModelName(OVMS_ServerSingleModelOptions* options,
    const char* model_name);
// --model_path
void OVMS_ServerSingleModelOptionsSetModelPath(OVMS_ServerSingleModelOptions* options,
    const char* model_path);

// --batch_size
// accepting only range positive numbers
void OVMS_ServerSingleModelOptionsSetStaticBatchSize(OVMS_ServerSingleModelOptions* options,
    uint64_t batch_size);
// Allows accepting range of batch size (only positive numbers)
// min must be lower than max
void OVMS_ServerSingleModelOptionsSetBatchSizeRange(OVMS_ServerSingleModelOptions* options,
    uint64_t min,
    uint64_t max);
// Allows accepting dynamic batch size (-1 in OVMS)
void OVMS_ServerSingleModelOptionsSetDynamicBatchSize(OVMS_ServerSingleModelOptions* options);
// Accepts automatic reloading to meet requests batch size (auto)
void OVMS_ServerSingleModelOptionsSetAutoBatchSize(OVMS_ServerSingleModelOptions* options);

// --layout
// Allows transforming the accepted input layout (alter the model) or override deduced one 
void OVMS_ServerSingleModelOptionsSetLayout(OVMS_ServerSingleModelOptions* options,
    const char* layout);

// --model_version_policy
// Sets model version policy to latest N models
void OVMS_ServerSingleModelOptionsSetModelVersionPolicyLatestVersionsNum(OVMS_ServerSingleModelOptions* options,
    uint64_t num);
// Sets model version policy to all
void OVMS_ServerSingleModelOptionsSetModelVersionPolicyAll(OVMS_ServerSingleModelOptions* options);
// Sets model version policy to specific and adds given specific versions to the list
void OVMS_ServerSingleModelOptionsSetModelVersionPolicyLatestSpecificAndAdd(OVMS_ServerSingleModelOptions* options,
    uint64_t specific_version);

// --nireq
void OVMS_ServerSingleModelOptionsSetNireq(OVMS_ServerSingleModelOptions* options,
    uint64_t nireq);
// --target_device
void OVMS_ServerSingleModelOptionsSetTargetDevice(OVMS_ServerSingleModelOptions* options,
    const char* target_device);
// --plugin_config
void OVMS_ServerSingleModelOptionsSetPluginConfig(OVMS_ServerSingleModelOptions* options,
    OVMS_ServerPluginConfigOption* plugin_config);
// --stateful
void OVMS_ServerSingleModelOptionsSetStateful(OVMS_ServerSingleModelOptions* options,
    bool is_stateful);
// --metrics_enable/--metrics_list
void OVMS_ServerSingleModelOptionsSetMetricConfig(OVMS_ServerSingleModelOptions* options,
    OVMS_ServerMetricOption* metric_config);
// --idle_sequence_cleanup
void OVMS_ServerSingleModelOptionsEnableIdleSequenceCleanup(OVMS_ServerSingleModelOptions* options,
    bool enable);
// --low_latency_transformation
void OVMS_ServerSingleModelOptionsEnableLowLatencyTransformation(OVMS_ServerSingleModelOptions* options,
    bool enable);
// --max_sequence_number
void OVMS_ServerSingleModelOptionsSetMaxSequenceNumber(OVMS_ServerSingleModelOptions* options,
    int64_t max_sequence_number);


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

// Start with single model directory (Optional for POC)
// Return error if already started
void OVMS_ServerStartSingleModel(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerSingleModelOptions* single_model_specific_options);
// Start with configuration file config.json
// Return error if already started
void OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerMultiModelOptions* multi_model_specific_options); // in fact only --config_path
// Unload all and cleanup
// TODO: Should not be possible to re-start?
void OVMS_ServerStop(OVMS_Server* server);
