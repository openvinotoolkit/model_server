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

#include <algorithm>
#include <filesystem>
#include <limits>
#include <regex>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <sysexits.h>

#include "logging.hpp"
#include "modelconfig.hpp"
#include "poc_api_impl.hpp"
#include "version.hpp"

namespace ovms {

const uint AVAILABLE_CORES = std::thread::hardware_concurrency();
const uint MAX_PORT_NUMBER = std::numeric_limits<ushort>::max();

const uint64_t DEFAULT_REST_WORKERS = AVAILABLE_CORES * 4.0;
const std::string DEFAULT_REST_WORKERS_STRING{std::to_string(DEFAULT_REST_WORKERS)};
const uint64_t MAX_REST_WORKERS = 10'000;

uint32_t Config::__maxSequenceNumber() const {
    if (!result->count("max_sequence_number")) {
        return DEFAULT_MAX_SEQUENCE_NUMBER;
    }
    return result->operator[]("max_sequence_number").as<uint32_t>();
}
uint32_t Config::maxSequenceNumber() const { return _maxSequenceNumber; }

Config& Config::parse(int argc, char** argv) {
    try {
        options = std::make_unique<cxxopts::Options>(argv[0], "OpenVINO Model Server");

        // clang-format off
        options->add_options()
            ("h, help",
                "Show this help message and exit")
            ("version",
                "Show binary version")
            ("port",
                "gRPC server port",
                cxxopts::value<uint64_t>()->default_value("9178"),
                "PORT")
            ("grpc_bind_address",
                "Network interface address to bind to for the gRPC API",
                cxxopts::value<std::string>()->default_value("0.0.0.0"),
                "GRPC_BIND_ADDRESS")
            ("rest_port",
                "REST server port, the REST server will not be started if rest_port is blank or set to 0",
                cxxopts::value<uint64_t>()->default_value("0"),
                "REST_PORT")
            ("rest_bind_address",
                "Network interface address to bind to for the REST API",
                cxxopts::value<std::string>()->default_value("0.0.0.0"),
                "REST_BIND_ADDRESS")
            ("grpc_workers",
                "Number of gRPC servers. Default 1. Increase for multi client, high throughput scenarios",
                cxxopts::value<uint>()->default_value("1"),
                "GRPC_WORKERS")
            ("rest_workers",
                "Number of worker threads in REST server - has no effect if rest_port is not set. Default value depends on number of CPUs. ",
                cxxopts::value<uint>()->default_value(DEFAULT_REST_WORKERS_STRING.c_str()),
                "REST_WORKERS")
            ("log_level",
                "serving log level - one of TRACE, DEBUG, INFO, WARNING, ERROR",
                cxxopts::value<std::string>()->default_value("INFO"), "LOG_LEVEL")
            ("log_path",
                "Optional path to the log file",
                cxxopts::value<std::string>(), "LOG_PATH")
#ifdef MTR_ENABLED
            ("trace_path",
                "Path to the trace file",
                cxxopts::value<std::string>(), "TRACE_PATH")
#endif
            ("grpc_channel_arguments",
                "A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000)",
                cxxopts::value<std::string>(), "GRPC_CHANNEL_ARGUMENTS")
            ("file_system_poll_wait_seconds",
                "Time interval between config and model versions changes detection. Default is 1. Zero or negative value disables changes monitoring.",
                cxxopts::value<uint>()->default_value("1"),
                "FILE_SYSTEM_POLL_WAIT_SECONDS")
            ("sequence_cleaner_poll_wait_minutes",
                "Time interval between two consecutive sequence cleanup scans. Default is 5. Zero value disables sequence cleaner.",
                cxxopts::value<uint32_t>()->default_value("5"),
                "SEQUENCE_CLEANER_POLL_WAIT_MINUTES")
            ("custom_node_resources_cleaner_interval",
                "Time interval between two consecutive resources cleanup scans. Default is 1. Must be greater than 0.",
                cxxopts::value<uint32_t>()->default_value("1"),
                "CUSTOM_NODE_RESOURCES_CLEANER_INTERVAL")
            ("cache_dir",
                "Overrides model cache directory. By default cache files are saved into /opt/cache if the directory is present. When enabled, first model load will produce cache files.",
                cxxopts::value<std::string>(),
                "CACHE_DIR")
            ("cpu_extension",
                "A path to shared library containing custom CPU layer implementation. Default: empty.",
                cxxopts::value<std::string>()->default_value(""),
                "CPU_EXTENSION");
        options->add_options("multi model")
            ("config_path",
                "Absolute path to json configuration file",
                cxxopts::value<std::string>(), "CONFIG_PATH");

        options->add_options("single model")
            ("model_name",
                "Name of the model",
                cxxopts::value<std::string>(),
                "MODEL_NAME")
            ("model_path",
                "Absolute path to model, as in tf serving",
                cxxopts::value<std::string>(),
                "MODEL_PATH")
            ("batch_size",
                "Resets models batchsize, int value or auto. This parameter will be ignored if shape is set",
                cxxopts::value<std::string>(),
                "BATCH_SIZE")
            ("shape",
                "Resets models shape (model must support reshaping). If set, batch_size parameter is ignored",
                cxxopts::value<std::string>(),
                "SHAPE")
            ("layout",
                "Resets model layout.",
                cxxopts::value<std::string>(),
                "LAYOUT")
            ("model_version_policy",
                "Model version policy",
                cxxopts::value<std::string>(),
                "MODEL_VERSION_POLICY")
            ("nireq",
                "Size of inference request queue for model executions. Recommended to be >= parallel executions. Default value calculated by OpenVINO based on available resources. Request for 0 is treated as request for default value",
                cxxopts::value<uint32_t>(),
                "NIREQ")
            ("target_device",
                "Target device to run the inference",
                cxxopts::value<std::string>()->default_value("CPU"),
                "TARGET_DEVICE")
            ("plugin_config",
                "A dictionary of plugin configuration keys and their values, eg \"{\\\"PERFORMANCE_HINT\\\": \\\"LATENCY\\\"}\". Default is PERFORMANCE_HINT set to THROUGHPUT mode. ",
                cxxopts::value<std::string>(),
                "PLUGIN_CONFIG")
            ("stateful",
                "Flag indicating model is stateful",
                cxxopts::value<bool>()->default_value("false"),
                "STATEFUL")
            ("metrics_enable",
                "Flag enabling metrics endpoint on rest_port.",
                cxxopts::value<bool>()->default_value("false"),
                "METRICS")
            ("metrics_list",
                "Comma separated list of metrics. If unset, only default metrics will be enabled. Default metrics: ovms_requests_success, ovms_requests_fail, ovms_request_time_us, ovms_streams, ovms_inference_time_us, ovms_wait_for_infer_req_time_us. When set, only the listed metrics will be enabled. Optional metrics: ovms_infer_req_queue_size, ovms_infer_req_active.",
                cxxopts::value<std::string>()->default_value(""),
                "METRICS_LIST")
            ("idle_sequence_cleanup",
                "Flag indicating if model is subject to sequence cleaner scans",
                cxxopts::value<bool>()->default_value("true"),
                "IDLE_SEQUENCE_CLEANUP")
            ("low_latency_transformation",
                "Flag indicating that Model Server should perform low latency transformation on that model",
                cxxopts::value<bool>()->default_value("false"),
                "LOW_LATENCY_TRANSFORMATION")
            ("max_sequence_number",
                "Determines how many sequences can be processed concurrently by one model instance. When that value is reached, attempt to start a new sequence will result in error.",
                cxxopts::value<uint32_t>(),
                "MAX_SEQUENCE_NUMBER");

        // clang-format on

        result = std::make_unique<cxxopts::ParseResult>(options->parse(argc, argv));

        if (result->count("version")) {
            std::string project_name(PROJECT_NAME);
            std::string project_version(PROJECT_VERSION);
            std::cout << project_name + " " + project_version << std::endl;
            std::cout << "OpenVINO backend " << OPENVINO_NAME << std::endl;
            exit(EX_OK);
        }

        if (result->count("help") || result->arguments().size() == 0) {
            std::cout << options->help({"", "multi model", "single model"}) << std::endl;
            exit(EX_OK);
        }

        this->_configPath = __configPath();
        this->_port = __port();
        this->_cpuExtensionLibraryPath = __cpuExtensionLibraryPath();
        this->_grpcBindAddress = __grpcBindAddress();
        this->_restPort = __restPort();
        this->_restBindAddress = __restBindAddress();
        this->_grpcWorkers = __grpcWorkers();
        this->_restWorkers = __restWorkers();
        this->_modelName = __modelName();
        this->_modelPath = __modelPath();
        this->_batchSize = __batchSize();
        this->_shape = __shape();
        this->_layout = __layout();
        this->_modelVersionPolicy = __modelVersionPolicy();
        this->_nireq = __nireq();
        this->_targetDevice = __targetDevice();
        this->_pluginConfig = __pluginConfig();
        this->_stateful = __stateful();
        this->_metricsEnabled = __metricsEnabled();
        this->_metricsList = __metricsList();
        this->_idleSequenceCleanup = __idleSequenceCleanup();
        this->_lowLatencyTransformation = __lowLatencyTransformation();
        this->_maxSequenceNumber = __maxSequenceNumber();
        this->_logLevel = __logLevel();
        this->_logPath = __logPath();
#ifdef MTR_ENABLED
        this->_tracePath = __tracePath();
#endif
        this->_grpcChannelArguments = __grpcChannelArguments();
        this->_filesystemPollWaitSeconds = __filesystemPollWaitSeconds();
        this->_sequenceCleanerPollWaitMinutes = __sequenceCleanerPollWaitMinutes();
        this->_resourcesCleanerPollWaitSeconds = __resourcesCleanerPollWaitSeconds();
        this->_cacheDir = __cacheDir();

        validate();
    } catch (const cxxopts::OptionException& e) {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(EX_USAGE);
    }

    return instance();
}

Config& Config::parse(GeneralOptionsImpl* go, MultiModelOptionsImpl* mmo) {
    // TODO: Implement
    this->_port = go->grpcPort;
    this->_restPort = go->restPort;
    this->_configPath = mmo->configPath;
    return instance();
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

void Config::validate() {
    // cannot set both config path & model_name/model_path
    if (result->count("config_path") && (result->count("model_name") || result->count("model_path"))) {
        std::cerr << "Use either config_path or model_path with model_name" << std::endl;
        exit(EX_USAGE);
    }

    if (!result->count("config_path") && !(result->count("model_name") && result->count("model_path"))) {
        std::cerr << "Use config_path or model_path with model_name" << std::endl;
        exit(EX_USAGE);
    }

    if (result->count("config_path") && (result->count("batch_size") || result->count("shape") ||
                                            result->count("nireq") || result->count("model_version_policy") || result->count("target_device") ||
                                            result->count("plugin_config"))) {
        std::cerr << "Model parameters in CLI are exclusive with the config file" << std::endl;
        exit(EX_USAGE);
    }

    // check grpc_workers value
    if (result->count("grpc_workers") && ((this->grpcWorkers() > AVAILABLE_CORES) || (this->grpcWorkers() < 1))) {
        std::cerr << "grpc_workers count should be from 1 to CPU core count : " << AVAILABLE_CORES << std::endl;
        exit(EX_USAGE);
    }

    // check rest_workers value
    if (result->count("rest_workers") && ((this->restWorkers() > MAX_REST_WORKERS) || (this->restWorkers() < 2))) {
        std::cerr << "rest_workers count should be from 2 to " << MAX_REST_WORKERS << std::endl;
        exit(EX_USAGE);
    }

    if (result->count("rest_workers") && (this->restWorkers() != DEFAULT_REST_WORKERS) && this->restPort() == 0) {
        std::cerr << "rest_workers is set but rest_port is not set. rest_port is required to start rest servers" << std::endl;
        exit(EX_USAGE);
    }

    // check docker ports
    if (result->count("port") && ((this->port() > MAX_PORT_NUMBER) || (this->port() < 0))) {
        std::cerr << "port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
        exit(EX_USAGE);
    }
    if (result->count("rest_port") && ((this->restPort() > MAX_PORT_NUMBER) || (this->restPort() < 0))) {
        std::cerr << "rest_port number out of range from 0 to " << MAX_PORT_NUMBER << std::endl;
        exit(EX_USAGE);
    }

    // metrics on rest port
    if (result->count("metrics_enable") && !result->count("rest_port")) {
        std::cerr << "rest_port setting is missing, metrics are enabled on rest port" << std::endl;
        exit(EX_USAGE);
    }

    // metrics on rest port
    if ((result->count("metrics_enable") || result->count("metrics_list")) && result->count("config_path")) {
        std::cerr << "metrics_enable or metrics_list and config_path cant be used together. Use json config file to enable metrics when using config_path." << std::endl;
        exit(EX_USAGE);
    }

    // metrics_list without metrics_enable
    if (!result->count("metrics_enable") && result->count("metrics_list")) {
        std::cerr << "metrics_enable setting is missing, required when metrics_list is provided" << std::endl;
        exit(EX_USAGE);
    }

    // check bind addresses:
    if (result->count("rest_bind_address") && check_hostname_or_ip(this->restBindAddress()) == false) {
        std::cerr << "rest_bind_address has invalid format: proper hostname or IP address expected." << std::endl;
        exit(EX_USAGE);
    }
    if (result->count("grpc_bind_address") && check_hostname_or_ip(this->grpcBindAddress()) == false) {
        std::cerr << "grpc_bind_address has invalid format: proper hostname or IP address expected." << std::endl;
        exit(EX_USAGE);
    }

    // port and rest_port cannot be the same
    if (this->port() == this->restPort()) {
        std::cerr << "port and rest_port cannot have the same values" << std::endl;
        exit(EX_USAGE);
    }

    // check cpu_extension path:
    if (result->count("cpu_extension") && !std::filesystem::exists(this->cpuExtensionLibraryPath())) {
        std::cerr << "File path provided as an --cpu_extension parameter does not exists in the filesystem: " << this->cpuExtensionLibraryPath() << std::endl;
        exit(EX_USAGE);
    }

    // check log_level values
    if (result->count("log_level")) {
        std::vector v({"TRACE", "DEBUG", "INFO", "WARNING", "ERROR"});
        if (std::find(v.begin(), v.end(), this->logLevel()) == v.end()) {
            std::cerr << "log_level should be one of: TRACE, DEBUG, INFO, WARNING, ERROR" << std::endl;
            exit(EX_USAGE);
        }
    }

    // check stateful flags:
    if ((result->count("low_latency_transformation") || result->count("max_sequence_number") || result->count("idle_sequence_cleanup")) && !result->count("stateful")) {
        std::cerr << "Setting low_latency_transformation, max_sequence_number and idle_sequence_cleanup require setting stateful flag for the model." << std::endl;
        exit(EX_USAGE);
    }
    return;
}

const std::string& Config::__configPath() const {
    if (result->count("config_path"))
        return result->operator[]("config_path").as<std::string>();
    return empty;
}

const std::string& Config::configPath() const { return _configPath; }

uint64_t Config::__port() const {
    return result->operator[]("port").as<uint64_t>();
}

uint64_t Config::port() const { return _port; }

const std::string Config::__cpuExtensionLibraryPath() const {
    if (result != nullptr && result->count("cpu_extension")) {
        return result->operator[]("cpu_extension").as<std::string>();
    }
    return "";
}

const std::string Config::cpuExtensionLibraryPath() const { return _cpuExtensionLibraryPath; }

const std::string Config::__grpcBindAddress() const {
    if (result->count("grpc_bind_address"))
        return result->operator[]("grpc_bind_address").as<std::string>();
    return "0.0.0.0";
}

const std::string Config::grpcBindAddress() const { return _grpcBindAddress; }

uint64_t Config::__restPort() const {
    return result->operator[]("rest_port").as<uint64_t>();
}

uint64_t Config::restPort() const { return _restPort; }

const std::string Config::__restBindAddress() const {
    if (result->count("rest_bind_address"))
        return result->operator[]("rest_bind_address").as<std::string>();
    return "0.0.0.0";
}

const std::string Config::restBindAddress() const { return _restBindAddress; }

uint Config::__grpcWorkers() const {
    return result->operator[]("grpc_workers").as<uint>();
}

uint Config::grpcWorkers() const { return _grpcWorkers; }

uint Config::__restWorkers() const {
    return result->operator[]("rest_workers").as<uint>();
}

uint Config::restWorkers() const { return _restWorkers; }

const std::string& Config::__modelName() const {
    if (result->count("model_name"))
        return result->operator[]("model_name").as<std::string>();
    return empty;
}

const std::string& Config::modelName() const { return _modelName; }

const std::string& Config::__modelPath() const {
    if (result->count("model_path"))
        return result->operator[]("model_path").as<std::string>();
    return empty;
}

const std::string& Config::modelPath() const { return _modelPath; }

const std::string& Config::__batchSize() const {
    if (!result->count("batch_size")) {
        static const std::string d = "0";
        return d;
    }
    return result->operator[]("batch_size").as<std::string>();
}

const std::string& Config::batchSize() const { return _batchSize; }

const std::string& Config::__shape() const {
    if (result->count("shape"))
        return result->operator[]("shape").as<std::string>();
    return empty;
}

const std::string& Config::Config::shape() const { return _shape; }

const std::string& Config::__layout() const {
    if (result->count("layout"))
        return result->operator[]("layout").as<std::string>();
    return empty;
}

const std::string& Config::layout() const { return _layout; }

const std::string& Config::__modelVersionPolicy() const {
    if (result->count("model_version_policy"))
        return result->operator[]("model_version_policy").as<std::string>();
    return empty;
}

const std::string& Config::modelVersionPolicy() const { return _modelVersionPolicy; }

uint32_t Config::__nireq() const {
    if (!result->count("nireq")) {
        return 0;
    }
    return result->operator[]("nireq").as<uint32_t>();
}

uint32_t Config::nireq() const { return _nireq; }

const std::string& Config::__targetDevice() const {
    return result->operator[]("target_device").as<std::string>();
}

const std::string& Config::targetDevice() const { return _targetDevice; }

const std::string& Config::__pluginConfig() const {
    if (result->count("plugin_config"))
        return result->operator[]("plugin_config").as<std::string>();
    return empty;
}
const std::string& Config::Config::pluginConfig() const { return _pluginConfig; }

bool Config::__stateful() const {
    return result->operator[]("stateful").as<bool>();
}

bool Config::stateful() const { return _stateful; }

bool Config::__metricsEnabled() const {
    if (!result->count("metrics_enable")) {
        return false;
    }
    return result->operator[]("metrics_enable").as<bool>();
}

bool Config::metricsEnabled() const { return _metricsEnabled; }

std::string Config::__metricsList() const {
    if (!result->count("metrics_list")) {
        return std::string("");
    }
    return result->operator[]("metrics_list").as<std::string>();
}

std::string Config::metricsList() const { return _metricsList; }

bool Config::__idleSequenceCleanup() const {
    return result->operator[]("idle_sequence_cleanup").as<bool>();
}

bool Config::idleSequenceCleanup() const { return _idleSequenceCleanup; }

bool Config::__lowLatencyTransformation() const {
    return result->operator[]("low_latency_transformation").as<bool>();
}

bool Config::lowLatencyTransformation() const { return _lowLatencyTransformation; }

const std::string& Config::__logLevel() const {
    if (result->count("log_level"))
        return result->operator[]("log_level").as<std::string>();
    return empty;
}

const std::string& Config::logLevel() const { return _logLevel; }

const std::string& Config::__logPath() const {
    if (result->count("log_path"))
        return result->operator[]("log_path").as<std::string>();
    return empty;
}

const std::string& Config::logPath() const { return _logPath; }

#ifdef MTR_ENABLED
const std::string& Config::__tracePath() const {
    if (result->count("trace_path"))
        return result->operator[]("trace_path").as<std::string>();
    return empty;
}

const std::string& Config::tracePath() const { return _tracePath; }
#endif

const std::string& Config::__grpcChannelArguments() const {
    if (result->count("grpc_channel_arguments"))
        return result->operator[]("grpc_channel_arguments").as<std::string>();
    return empty;
}

const std::string& Config::grpcChannelArguments() const { return _grpcChannelArguments; }

uint Config::__filesystemPollWaitSeconds() const {
    return result->operator[]("file_system_poll_wait_seconds").as<uint>();
}

uint Config::filesystemPollWaitSeconds() const { return _filesystemPollWaitSeconds; }

uint32_t Config::__sequenceCleanerPollWaitMinutes() const {
    return result->operator[]("sequence_cleaner_poll_wait_minutes").as<uint32_t>();
}

uint32_t Config::sequenceCleanerPollWaitMinutes() const { return _sequenceCleanerPollWaitMinutes; }

uint32_t Config::__resourcesCleanerPollWaitSeconds() const {
    return result->operator[]("custom_node_resources_cleaner_interval").as<uint32_t>();
}

uint32_t Config::resourcesCleanerPollWaitSeconds() const { return _resourcesCleanerPollWaitSeconds; }

const std::string Config::__cacheDir() const {
    if (result != nullptr && result->count("cache_dir")) {
        return result->operator[]("cache_dir").as<std::string>();
    }
    return empty;
}

const std::string Config::cacheDir() const { return _cacheDir; }

}  // namespace ovms
