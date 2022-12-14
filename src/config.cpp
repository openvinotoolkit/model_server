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

#include <sysexits.h>

#include "cli_parser.hpp"
#include "modelconfig.hpp"
#include "server_options.hpp"

namespace ovms {

const uint AVAILABLE_CORES = std::thread::hardware_concurrency();
const uint MAX_PORT_NUMBER = std::numeric_limits<ushort>::max();

const uint64_t DEFAULT_REST_WORKERS = AVAILABLE_CORES * 4.0;
const uint64_t MAX_REST_WORKERS = 10'000;

// TODO: Not used in OVMS - get rid of. Used only in tests.
Config& Config::parse(int argc, char** argv) {
    ovms::CLIParser p;
    ovms::GeneralOptionsImpl go;
    ovms::MultiModelOptionsImpl mmo;
    p.parse(argc, argv);
    p.prepare(&go, &mmo);
    if (!this->parse(&go, &mmo))
        exit(EX_USAGE);
    return *this;
}

bool Config::parse(GeneralOptionsImpl* go, MultiModelOptionsImpl* mmo) {
    this->go = *go;
    this->mmo = *mmo;
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

    if (!configPath().empty() && (!this->mmo.batchSize.empty() || !shape().empty() ||
                                     nireq() != 0 || !modelVersionPolicy().empty() || !this->mmo.targetDevice.empty() ||
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

    if (this->go.restWorkers.has_value() && restPort() == 0) {
        std::cerr << "rest_workers is set but rest_port is not set. rest_port is required to start rest servers" << std::endl;
        return false;
    }

    if (port() && ((port() > MAX_PORT_NUMBER) || (port() < 0))) {
        std::cerr << "port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
        return false;
    }

    if (restPort() != 0 && ((restPort() > MAX_PORT_NUMBER) || (restPort() < 0))) {
        std::cerr << "rest_port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
        return false;
    }

    // metrics on rest port
    if (metricsEnabled() && restPort() == 0) {
        std::cerr << "rest_port setting is missing, metrics are enabled on rest port" << std::endl;
        return false;
    }

    // metrics on rest port
    if ((metricsEnabled() || !metricsList().empty()) && !configPath().empty()) {
        std::cerr << "metrics_enable or metrics_list and config_path cant be used together. Use json config file to enable metrics when using config_path." << std::endl;
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
    if ((this->mmo.lowLatencyTransformation.has_value() || this->mmo.maxSequenceNumber.has_value() || this->mmo.idleSequenceCleanup.has_value()) && !stateful()) {
        std::cerr << "Setting low_latency_transformation, max_sequence_number and idle_sequence_cleanup require setting stateful flag for the model." << std::endl;
        return false;
    }
    return true;
}

const std::string& Config::configPath() const { return this->mmo.configPath; }
uint32_t Config::port() const { return this->go.grpcPort; }
const std::string Config::cpuExtensionLibraryPath() const { return this->go.cpuExtensionLibraryPath; }
const std::string Config::grpcBindAddress() const { return this->go.grpcBindAddress; }
uint32_t Config::restPort() const { return this->go.restPort; }
const std::string Config::restBindAddress() const { return this->go.restBindAddress; }
uint32_t Config::grpcWorkers() const { return this->go.grpcWorkers; }
uint32_t Config::restWorkers() const { return this->go.restWorkers.value_or(DEFAULT_REST_WORKERS); }
const std::string& Config::modelName() const { return this->mmo.modelName; }
const std::string& Config::modelPath() const { return this->mmo.modelPath; }
const std::string& Config::batchSize() const {
    static const std::string defaultBatch = "0";
    return this->mmo.batchSize.empty() ? defaultBatch : this->mmo.batchSize;
}
const std::string& Config::Config::shape() const { return this->mmo.shape; }
const std::string& Config::layout() const { return this->mmo.layout; }
const std::string& Config::modelVersionPolicy() const { return this->mmo.modelVersionPolicy; }
uint32_t Config::nireq() const { return this->mmo.nireq; }
const std::string& Config::targetDevice() const {
    static const std::string defaultTargetDevice = "CPU";
    return this->mmo.targetDevice.empty() ? defaultTargetDevice : this->mmo.targetDevice;
}
const std::string& Config::Config::pluginConfig() const { return this->mmo.pluginConfig; }
bool Config::stateful() const { return this->mmo.stateful.value_or(false); }
bool Config::metricsEnabled() const { return this->go.metricsEnabled; }
std::string Config::metricsList() const { return this->go.metricsList; }
bool Config::idleSequenceCleanup() const { return this->mmo.idleSequenceCleanup.value_or(true); }
uint32_t Config::maxSequenceNumber() const { return this->mmo.maxSequenceNumber.value_or(DEFAULT_MAX_SEQUENCE_NUMBER); }
bool Config::lowLatencyTransformation() const { return this->mmo.lowLatencyTransformation.value_or(false); }
const std::string& Config::logLevel() const { return this->go.logLevel; }
const std::string& Config::logPath() const { return this->go.logPath; }
#ifdef MTR_ENABLED
const std::string& Config::tracePath() const { return this->go.tracePath; }
#endif
const std::string& Config::grpcChannelArguments() const { return this->go.grpcChannelArguments; }
uint32_t Config::filesystemPollWaitSeconds() const { return this->go.filesystemPollWaitSeconds; }
uint32_t Config::sequenceCleanerPollWaitMinutes() const { return this->go.sequenceCleanerPollWaitMinutes; }
uint32_t Config::resourcesCleanerPollWaitSeconds() const { return this->go.resourcesCleanerPollWaitSeconds; }
const std::string Config::cacheDir() const { return this->go.cacheDir; }

}  // namespace ovms
