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
#include <limits>
#include <thread>

#include <sysexits.h>

#include "version.hpp"

namespace ovms {

const uint AVAILABLE_CORES = std::thread::hardware_concurrency();
const uint MAX_PORT_NUMBER = std::numeric_limits<ushort>::max();

const std::string DEFAULT_REST_WORKERS_STRING{"24"};
const uint64_t DEFAULT_REST_WORKERS = std::stoul(DEFAULT_REST_WORKERS_STRING);
const uint64_t MAX_REST_WORKERS = 10'000;

Config& Config::parse(int argc, char** argv) {
    try {
        options = std::make_unique<cxxopts::Options>(argv[0], "OpenVINO Model Server");

        // clang-format off
        options->add_options()
            ("h, help",
                "show this help message and exit")
            ("version",
                "show binary version")
            ("port",
                "gRPC server port",
                cxxopts::value<uint64_t>()->default_value("9178"),
                "PORT")
            ("rest_port",
                "REST server port, the REST server will not be started if rest_port is blank or set to 0",
                cxxopts::value<uint64_t>()->default_value("0"),
                "REST_PORT")
            ("grpc_workers",
                "number of gRPC servers. Default 1. Increase for multi client, high throughput scenarios",
                cxxopts::value<uint>()->default_value("1"),
                "GRPC_WORKERS")
            ("rest_workers",
                "number of workers in REST server - has no effect if rest_port is not set",
                cxxopts::value<uint>()->default_value(DEFAULT_REST_WORKERS_STRING.c_str()),
                "REST_WORKERS")
            ("log_level",
                "serving log level - one of DEBUG, INFO, ERROR",
                cxxopts::value<std::string>()->default_value("INFO"), "LOG_LEVEL")
            ("log_path",
                "optional path to the log file",
                cxxopts::value<std::string>(), "LOG_PATH")
            ("grpc_channel_arguments",
                "A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000)",
                cxxopts::value<std::string>(), "GRPC_CHANNEL_ARGUMENTS")
            ("file_system_poll_wait_seconds",
                "Time interval between config and model versions changes detection. Default is 1. Zero or negative value disables changes monitoring.",
                cxxopts::value<uint>()->default_value("1"),
                "SECONDS");
        options->add_options("multi model")
            ("config_path",
                "absolute path to json configuration file",
                cxxopts::value<std::string>(), "CONFIG_PATH");

        options->add_options("single model")
            ("model_name",
                "name of the model",
                cxxopts::value<std::string>(),
                "MODEL_NAME")
            ("model_path",
                "absolute path to model, as in tf serving",
                cxxopts::value<std::string>(),
                "MODEL_PATH")
            ("batch_size",
                "resets models batchsize, int value or auto. This parameter will be ignored if shape is set",
                cxxopts::value<std::string>(),
                "BATCH_SIZE")
            ("shape",
                "resets models shape (model must support reshaping). If set, batch_size parameter is ignored",
                cxxopts::value<std::string>(),
                "SHAPE")
            ("model_version_policy",
                "model version policy",
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
                "a dictionary of plugin configuration keys and their values, eg \"{\\\"CPU_THROUGHPUT_STREAMS\\\": \\\"1\\\"}\". Default throughput streams for CPU and GPU are calculated by OpenVINO",
                cxxopts::value<std::string>(),
                "PLUGIN_CONFIG");

        // clang-format on

        result = std::make_unique<cxxopts::ParseResult>(options->parse(argc, argv));

        if (result->count("version")) {
            std::cout << PROJECT_NAME << PROJECT_VER_PATCH << std::endl;
            std::cout << GetOpenVinoVersionFromPackageUrl() << std::endl;
            exit(EX_OK);
        }

        if (result->count("help") || result->arguments().size() == 0) {
            std::cout << options->help({"", "multi model", "single model"}) << std::endl;
            exit(EX_OK);
        }

        validate();
    } catch (const cxxopts::OptionException& e) {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(EX_USAGE);
    }

    return instance();
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
    if (result->count("rest_workers") && ((this->restWorkers() > MAX_REST_WORKERS) || (this->restWorkers() < 1))) {
        std::cerr << "rest_workers count should be from 1 to " << MAX_REST_WORKERS << std::endl;
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

    // port and rest_port cannot be the same
    if (this->port() == this->restPort()) {
        std::cerr << "port and rest_port cannot have the same values" << std::endl;
        exit(EX_USAGE);
    }

    return;
}

}  // namespace ovms
