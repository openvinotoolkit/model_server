//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "config.hpp"

#include <filesystem>
#include <limits>
#include <regex>
#include <thread>
#include <vector>

#include "logging.hpp"
#include "ovms_exit_codes.hpp"

#include "capi_frontend/server_settings.hpp"
#include "cli_parser.hpp"
#include "modelconfig.hpp"
#include "stringutils.hpp"
#include "systeminfo.hpp"

namespace ovms {

const uint32_t AVAILABLE_CORES = getCoreCount();
const uint32_t WIN_MAX_GRPC_WORKERS = 1;
const uint32_t MAX_PORT_NUMBER = std::numeric_limits<uint16_t>::max();

// For drogon, we need to minimize the number of default workers since this value is set for both: unary and streaming (making it always double)
#if (USE_DROGON == 0)
const uint64_t DEFAULT_REST_WORKERS = AVAILABLE_CORES * 4.0;
#else
const uint64_t DEFAULT_REST_WORKERS = AVAILABLE_CORES;
#endif
const uint32_t DEFAULT_GRPC_MAX_THREADS = AVAILABLE_CORES * 8.0;
const size_t DEFAULT_GRPC_MEMORY_QUOTA = (size_t)2 * 1024 * 1024 * 1024;  // 2GB
const uint64_t MAX_REST_WORKERS = 10'000;

Config& Config::parse(int argc, char** argv) {
    ovms::CLIParser p;
    ovms::ServerSettingsImpl serverSettings;
    ovms::ModelsSettingsImpl modelsSettings;
    p.parse(argc, argv);
    p.prepare(&serverSettings, &modelsSettings);
    if (!this->parse(&serverSettings, &modelsSettings))
        exit(OVMS_EX_USAGE);
    return *this;
}

bool Config::parse(ServerSettingsImpl* serverSettings, ModelsSettingsImpl* modelsSettings) {
    this->serverSettings = *serverSettings;
    this->modelsSettings = *modelsSettings;
    return validate();
}

bool Config::check_hostname_or_ip(const std::string& input) {
    if (input.size() > 255) {
        return false;
    }
    bool all_numeric = true;
    for (char c : input) {
        if (c == '.' || c == ':') {
            continue;
        }
        if (!::isxdigit(c)) {
            all_numeric = false;
        }
    }
    if (all_numeric) {
        static const std::regex valid_ipv4_regex("^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$");
        static const std::regex valid_ipv6_regex(R"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))");
        return std::regex_match(input, valid_ipv4_regex) ||
               std::regex_match(input, valid_ipv6_regex);
    } else {
        std::regex valid_hostname_regex("^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\\-]*[a-zA-Z0-9])\\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\\-]*[A-Za-z0-9])$");
        return std::regex_match(input, valid_hostname_regex);
    }
}

bool Config::validateUserSettingsInConfigAddRemoveModel(const ModelsSettingsImpl& modelsSettings) {
    static const std::vector<std::string> allowedUserSettings = {"model_name", "model_path"};
    std::vector<std::string> usedButDisallowedUserSettings;
    for (const std::string& userSetting : modelsSettings.userSetSingleModelArguments) {
        bool isAllowed = false;
        for (const std::string& allowedSetting : allowedUserSettings) {
            if (userSetting == allowedSetting)
                isAllowed = true;
        }

        if (!isAllowed)
            usedButDisallowedUserSettings.push_back(userSetting);
    }

    if (!usedButDisallowedUserSettings.empty()) {
        std::string arguments = "";
        for (const std::string& userSetting : usedButDisallowedUserSettings) {
            arguments += userSetting + ", ";
        }
        std::cerr << "Adding or removing models from the configuration file, allows passing only model_name and model_path parameters. Invalid parameters passed: " << arguments << std::endl;

        return false;
    }

    return true;
}

bool Config::validate() {
    if (this->serverSettings.serverMode == HF_PULL_MODE || this->serverSettings.serverMode == HF_PULL_AND_START_MODE) {
        if (!serverSettings.hfSettings.sourceModel.size()) {
            std::cerr << "source_model parameter is required for pull mode";
            return false;
        }
        if (!serverSettings.hfSettings.downloadPath.size()) {
            std::cerr << "model_repository_path parameter is required for pull mode";
            return false;
        }
        if (this->serverSettings.hfSettings.task == UNKNOWN_GRAPH) {
            std::cerr << "Error: --task parameter not set." << std::endl;
            return false;
        }
        if (this->serverSettings.hfSettings.task == TEXT_GENERATION_GRAPH) {
            if (!std::holds_alternative<TextGenGraphSettingsImpl>(this->serverSettings.hfSettings.graphSettings)) {
                std::cerr << "Graph options not initialized for text generation.";
                return false;
            }
            auto settings = std::get<TextGenGraphSettingsImpl>(this->serverSettings.hfSettings.graphSettings);
            std::vector allowedPipelineTypes = {"LM", "LM_CB", "VLM", "VLM_CB", "AUTO"};
            if (settings.pipelineType.has_value() && std::find(allowedPipelineTypes.begin(), allowedPipelineTypes.end(), settings.pipelineType) == allowedPipelineTypes.end()) {
                std::cerr << "pipeline_type: " << settings.pipelineType.value() << " is not allowed. Supported types: LM, LM_CB, VLM, VLM_CB, AUTO" << std::endl;
                return false;
            }

            std::vector allowedTargetDevices = {"CPU", "GPU", "NPU", "AUTO"};
            bool validDeviceSelected = false;
            if (settings.targetDevice.rfind("GPU.", 0) == 0) {
                // Accept GPU.x where x is a number to select specific GPU card
                std::string indexPart = settings.targetDevice.substr(4);
                validDeviceSelected = !indexPart.empty() && std::all_of(indexPart.begin(), indexPart.end(), ::isdigit);
            } else if (settings.targetDevice.rfind("HETERO", 0) == 0) {
                // Accept HETERO:<device1>,<device2>,... to select specific devices in the list
                validDeviceSelected = true;
            } else if (std::find(allowedTargetDevices.begin(), allowedTargetDevices.end(), settings.targetDevice) != allowedTargetDevices.end()) {
                // Accept CPU, GPU, NPU, AUTO as valid devices
                validDeviceSelected = true;
            }

            if (!validDeviceSelected) {
                std::cerr << "target_device: " << settings.targetDevice << " is not allowed. Supported devices: CPU, GPU, NPU, HETERO, AUTO" << std::endl;
                return false;
            }

            std::vector allowedBoolValues = {"false", "true"};
            if (std::find(allowedBoolValues.begin(), allowedBoolValues.end(), settings.enablePrefixCaching) == allowedBoolValues.end()) {
                std::cerr << "enable_prefix_caching: " << settings.enablePrefixCaching << " is not allowed. Supported values: true, false" << std::endl;
                return false;
            }

            if (std::find(allowedBoolValues.begin(), allowedBoolValues.end(), settings.dynamicSplitFuse) == allowedBoolValues.end()) {
                std::cerr << "dynamic_split_fuse: " << settings.dynamicSplitFuse << " is not allowed. Supported values: true, false" << std::endl;
                return false;
            }
        }

        if (this->serverSettings.hfSettings.task == EMBEDDINGS_GRAPH) {
            if (!std::holds_alternative<EmbeddingsGraphSettingsImpl>(this->serverSettings.hfSettings.graphSettings)) {
                std::cerr << "Graph options not initialized for embeddings.";
                return false;
            }
            auto settings = std::get<EmbeddingsGraphSettingsImpl>(this->serverSettings.hfSettings.graphSettings);

            std::vector allowedBoolValues = {"false", "true"};
            if (std::find(allowedBoolValues.begin(), allowedBoolValues.end(), settings.normalize) == allowedBoolValues.end()) {
                std::cerr << "normalize: " << settings.normalize << " is not allowed. Supported values: true, false" << std::endl;
                return false;
            }

            if (std::find(allowedBoolValues.begin(), allowedBoolValues.end(), settings.truncate) == allowedBoolValues.end()) {
                std::cerr << "truncate: " << settings.truncate << " is not allowed. Supported values: true, false" << std::endl;
                return false;
            }
        }
        // No more validation needed
        if (this->serverSettings.serverMode == HF_PULL_MODE) {
            return true;
        }
    }
    if (this->serverSettings.serverMode == LIST_MODELS_MODE) {
        if (this->serverSettings.hfSettings.downloadPath.empty()) {
            std::cerr << "Use --list_models with --model_repository_path" << std::endl;
            return false;
        }
        return true;
    }

    if (this->serverSettings.serverMode != MODIFY_CONFIG_MODE) {
        if (!configPath().empty() && (!modelName().empty() || !modelPath().empty())) {
            std::cerr << "Use either config_path or model_path with model_name" << std::endl;
            return false;
        }
        if (configPath().empty() && !(!modelName().empty() && !modelPath().empty())) {
            std::cerr << "Use config_path or model_path with model_name" << std::endl;
            return false;
        }
        if (!configPath().empty() && (!this->modelsSettings.batchSize.empty() || !shape().empty() ||
                                         nireq() != 0 || !modelVersionPolicy().empty() || !this->modelsSettings.targetDevice.empty() ||
                                         !pluginConfig().empty())) {
            std::cerr << "Model parameters in CLI are exclusive with the config file" << std::endl;
            return false;
        }
        // check grpc_workers value
        if (((grpcWorkers() > AVAILABLE_CORES) || (grpcWorkers() < 1))) {
            std::cerr << "grpc_workers count should be from 1 to CPU core count : " << AVAILABLE_CORES << std::endl;
            return false;
        }
        // metrics on rest port
        if (metricsEnabled() && restPort() == 0) {
            std::cerr << "rest_port setting is missing, metrics are enabled on rest port" << std::endl;
            return false;
        }
        // metrics_list without metrics_enable
        if (!metricsEnabled() && !metricsList().empty()) {
            std::cerr << "metrics_enable setting is missing, required when metrics_list is provided" << std::endl;
            return false;
        }
        // both ports cannot be unset
        if (startedFromCLI() && ((restPort() == 0) && (port() == 0))) {
            std::cerr << "port and rest_port cannot both be unset" << std::endl;
            return false;
        }
    } else {
        if (configPath().empty()) {
            std::cerr << "Set config_path with add_to_config/remove_from_config" << std::endl;
            return false;
        }
        if (modelName().empty()) {
            std::cerr << "Set model_name with add_to_config/remove_from_config" << std::endl
                      << "Usage: " << std::endl
                      << "  ovms --model_name <model_name> --model_repository_path <repo_path> --add_to_config <config_path>" << std::endl
                      << "  ovms --model_name <model_name> --model_path <model_path> --add_to_config <config_path>" << std::endl
                      << "  ovms --model_name <model_name> --remove_from_config <config_path>" << std::endl;
            return false;
        }
        if (modelPath().empty() && this->serverSettings.exportConfigType == ENABLE_MODEL) {
            std::cerr << "Set model_name either with model_path or model_repository_path with add_to_config" << std::endl
                      << "Usage: " << std::endl
                      << "  ovms --model_name <model_name> --model_repository_path <repo_path> --add_to_config <config_path>" << std::endl
                      << "  ovms --model_name <model_name> --model_path <model_path> --add_to_config <config_path>" << std::endl;
            return false;
        }

        if (!Config::validateUserSettingsInConfigAddRemoveModel(this->modelsSettings))
            return false;
    }

    // check rest_workers value
    if (((restWorkers() > MAX_REST_WORKERS) || (restWorkers() < 2))) {
        std::cerr << "rest_workers count should be from 2 to " << MAX_REST_WORKERS << std::endl;
        return false;
    }

    if (this->serverSettings.restWorkers.has_value() && restPort() == 0) {
        std::cerr << "rest_workers is set but rest_port is not set. rest_port is required to start rest servers" << std::endl;
        return false;
    }

#ifdef _WIN32
    if (grpcWorkers() > WIN_MAX_GRPC_WORKERS) {
        std::cerr << "grpcWorkers count can only be set to 1 on Windows. Set " << grpcWorkers() << std::endl;
        return false;
    }
#endif

    if (port() && (port() > MAX_PORT_NUMBER)) {
        std::cerr << "port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
        return false;
    }

    if (restPort() && (restPort() > MAX_PORT_NUMBER)) {
        std::cerr << "rest_port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
        return false;
    }

    // check bind addresses:
    if (!restBindAddress().empty() && check_hostname_or_ip(restBindAddress()) == false) {
        std::cerr << "rest_bind_address has invalid format: proper hostname or IP address expected." << std::endl;
        return false;
    }
    if (!grpcBindAddress().empty() && check_hostname_or_ip(grpcBindAddress()) == false) {
        std::cerr << "grpc_bind_address has invalid format: proper hostname or IP address expected." << std::endl;
        return false;
    }
    // port and rest_port cannot be the same
    if ((port() == restPort()) && (port() != 0)) {
        std::cerr << "port and rest_port cannot have the same values" << std::endl;
        return false;
    }

    // check cpu_extension path:
    if (!cpuExtensionLibraryPath().empty() && !std::filesystem::exists(cpuExtensionLibraryPath())) {
        std::cerr << "File path provided as an --cpu_extension parameter does not exist in the filesystem: " << this->cpuExtensionLibraryPath() << std::endl;
        return false;
    }

    // check log_level values
    std::vector v({"TRACE", "DEBUG", "INFO", "WARNING", "ERROR"});
    if (std::find(v.begin(), v.end(), logLevel()) == v.end()) {
        std::cerr << "log_level should be one of: TRACE, DEBUG, INFO, WARNING, ERROR" << std::endl;
        return false;
    }
    // check stateful flags:
    if ((this->modelsSettings.lowLatencyTransformation.has_value() || this->modelsSettings.maxSequenceNumber.has_value() || this->modelsSettings.idleSequenceCleanup.has_value()) && !stateful()) {
        std::cerr << "Setting low_latency_transformation, max_sequence_number and idle_sequence_cleanup require setting stateful flag for the model." << std::endl;
        return false;
    }
    return true;
}

const std::string& Config::configPath() const { return this->modelsSettings.configPath; }
uint32_t Config::port() const { return this->serverSettings.grpcPort; }
const std::string Config::cpuExtensionLibraryPath() const { return this->serverSettings.cpuExtensionLibraryPath; }
const std::string Config::grpcBindAddress() const { return this->serverSettings.grpcBindAddress; }
uint32_t Config::restPort() const { return this->serverSettings.restPort; }
const std::string Config::restBindAddress() const { return this->serverSettings.restBindAddress; }
uint32_t Config::grpcWorkers() const { return this->serverSettings.grpcWorkers; }
uint32_t Config::grpcMaxThreads() const { return this->serverSettings.grpcMaxThreads.value_or(DEFAULT_GRPC_MAX_THREADS); }
size_t Config::grpcMemoryQuota() const { return this->serverSettings.grpcMemoryQuota.value_or(DEFAULT_GRPC_MEMORY_QUOTA); }
uint32_t Config::restWorkers() const { return this->serverSettings.restWorkers.value_or(DEFAULT_REST_WORKERS); }
const std::string& Config::modelName() const { return this->modelsSettings.modelName; }
const std::string& Config::modelPath() const { return this->modelsSettings.modelPath; }
const std::string& Config::batchSize() const {
    static const std::string defaultBatch = "";
    return this->modelsSettings.batchSize.empty() ? defaultBatch : this->modelsSettings.batchSize;
}
const std::string& Config::Config::shape() const { return this->modelsSettings.shape; }
const std::string& Config::layout() const { return this->modelsSettings.layout; }
const std::string& Config::modelVersionPolicy() const { return this->modelsSettings.modelVersionPolicy; }
uint32_t Config::nireq() const { return this->modelsSettings.nireq; }
const std::string& Config::targetDevice() const {
    static const std::string defaultTargetDevice = "CPU";
    return this->modelsSettings.targetDevice.empty() ? defaultTargetDevice : this->modelsSettings.targetDevice;
}
const std::string& Config::Config::pluginConfig() const { return this->modelsSettings.pluginConfig; }
bool Config::stateful() const { return this->modelsSettings.stateful.value_or(false); }
bool Config::metricsEnabled() const { return this->serverSettings.metricsEnabled; }
std::string Config::metricsList() const { return this->serverSettings.metricsList; }
bool Config::idleSequenceCleanup() const { return this->modelsSettings.idleSequenceCleanup.value_or(true); }
uint32_t Config::maxSequenceNumber() const { return this->modelsSettings.maxSequenceNumber.value_or(DEFAULT_MAX_SEQUENCE_NUMBER); }
bool Config::lowLatencyTransformation() const { return this->modelsSettings.lowLatencyTransformation.value_or(false); }
const std::string& Config::logLevel() const { return this->serverSettings.logLevel; }
const std::string& Config::logPath() const { return this->serverSettings.logPath; }
#ifdef MTR_ENABLED
const std::string& Config::tracePath() const { return this->serverSettings.tracePath; }
#endif
const std::string& Config::grpcChannelArguments() const { return this->serverSettings.grpcChannelArguments; }
uint32_t Config::filesystemPollWaitMilliseconds() const { return this->serverSettings.filesystemPollWaitMilliseconds; }
uint32_t Config::sequenceCleanerPollWaitMinutes() const { return this->serverSettings.sequenceCleanerPollWaitMinutes; }
uint32_t Config::resourcesCleanerPollWaitSeconds() const { return this->serverSettings.resourcesCleanerPollWaitSeconds; }
bool Config::allowCredentials() const { return this->serverSettings.allowCredentials; }
const std::string& Config::allowedOrigins() const { return this->serverSettings.allowedOrigins; }
const std::string& Config::allowedMethods() const { return this->serverSettings.allowedMethods; }
const std::string& Config::allowedHeaders() const { return this->serverSettings.allowedHeaders; }
const std::string Config::cacheDir() const { return this->serverSettings.cacheDir; }

}  // namespace ovms
