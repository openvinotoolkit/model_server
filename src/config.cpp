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

#include <spdlog/spdlog.h>
#include <sysexits.h>

#include "capi_frontend/server_settings.hpp"
#include "cli_parser.hpp"
#include "modelconfig.hpp"
#include "systeminfo.hpp"

namespace ovms {

const uint AVAILABLE_CORES = getCoreCount();
const uint MAX_PORT_NUMBER = std::numeric_limits<ushort>::max();

const uint64_t DEFAULT_REST_WORKERS = AVAILABLE_CORES * 4.0;
const uint64_t MAX_REST_WORKERS = 10'000;

Config& Config::parse(int argc, char** argv) {
    ovms::CLIParser p;
    ovms::ServerSettingsImpl serverSettings;
    ovms::ModelsSettingsImpl modelsSettings;
    p.parse(argc, argv);
    p.prepare(&serverSettings, &modelsSettings);
    if (!this->parse(&serverSettings, &modelsSettings))
        exit(EX_USAGE);
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
        if (c == '.') {
            continue;
        }
        if (!::isdigit(c)) {
            all_numeric = false;
        }
    }
    if (all_numeric) {
        std::regex valid_ip_regex("^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$");
        return std::regex_match(input, valid_ip_regex);
    } else {
        std::regex valid_hostname_regex("^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\\-]*[a-zA-Z0-9])\\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\\-]*[A-Za-z0-9])$");
        return std::regex_match(input, valid_hostname_regex);
    }
}

bool Config::validate() {
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

    // check rest_workers value
    if (((restWorkers() > MAX_REST_WORKERS) || (restWorkers() < 2))) {
        std::cerr << "rest_workers count should be from 2 to " << MAX_REST_WORKERS << std::endl;
        return false;
    }

    if (this->serverSettings.restWorkers.has_value() && restPort() == 0) {
        std::cerr << "rest_workers is set but rest_port is not set. rest_port is required to start rest servers" << std::endl;
        return false;
    }

    if (port() && (port() > MAX_PORT_NUMBER)) {
        std::cerr << "port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
        return false;
    }

    if (restPort() && (restPort() > MAX_PORT_NUMBER)) {
        std::cerr << "rest_port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
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
    if (port() == restPort()) {
        std::cerr << "port and rest_port cannot have the same values" << std::endl;
        return false;
    }

    // check cpu_extension path:
    if (!cpuExtensionLibraryPath().empty() && !std::filesystem::exists(cpuExtensionLibraryPath())) {
        std::cerr << "File path provided as an --cpu_extension parameter does not exists in the filesystem: " << this->cpuExtensionLibraryPath() << std::endl;
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
uint32_t Config::restWorkers() const { return this->serverSettings.restWorkers.value_or(DEFAULT_REST_WORKERS); }
const std::string& Config::modelName() const { return this->modelsSettings.modelName; }
const std::string& Config::modelPath() const { return this->modelsSettings.modelPath; }
const std::string& Config::batchSize() const {
    static const std::string defaultBatch = "0";
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
uint32_t Config::filesystemPollWaitSeconds() const { return this->serverSettings.filesystemPollWaitSeconds; }
uint32_t Config::sequenceCleanerPollWaitMinutes() const { return this->serverSettings.sequenceCleanerPollWaitMinutes; }
uint32_t Config::resourcesCleanerPollWaitSeconds() const { return this->serverSettings.resourcesCleanerPollWaitSeconds; }
const std::string Config::cacheDir() const { return this->serverSettings.cacheDir; }

}  // namespace ovms
