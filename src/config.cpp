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

namespace ovms {

const uint AVAILABLE_CORES = std::thread::hardware_concurrency();
const std::string DEFAULT_NIREQ = std::to_string(AVAILABLE_CORES / 8 + 2);
const std::string DEFAULT_GRPC_SERVERS = std::to_string(AVAILABLE_CORES / 8 + 4);
const uint MAX_PORT_NUMBER = std::numeric_limits<ushort>::max();

Config& Config::parse(int argc, char** argv) {
    try {
        options = std::make_unique<cxxopts::Options>(argv[0], "OpenVINO Model Server");

        // clang-format off
        options->add_options()
            ("h, help",
                "show this help message and exit")
            ("port",
                "gRPC server port",
                cxxopts::value<uint64_t>()->default_value("9178"),
                "PORT")
            ("rest_port",
                "REST server port, the REST server will not be started if rest_port is blank or set to 0",
                cxxopts::value<uint64_t>()->default_value("0"),
                "REST_PORT")
            ("grpc_workers",
                "number of gRPC servers. Recommended to be >= NIREQ. Default value calculated at runtime: NIREQ + 2",
                cxxopts::value<uint>()->default_value(DEFAULT_GRPC_SERVERS.c_str()),
                "GRPC_WORKERS")
            ("rest_workers",
                "number of workers in REST server - has no effect if rest_port is not set",
                cxxopts::value<uint>()->default_value("24"),
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
                "sets models batchsize, int value or auto. This parameter will be ignored if shape is set",
                cxxopts::value<std::string>()->default_value("0"),
                "BATCH_SIZE")
            ("shape",
                "sets models shape (model must support reshaping). If set, batch_size parameter is ignored",
                cxxopts::value<std::string>(),
                "SHAPE")
            ("model_version_policy",
                "model version policy",
                cxxopts::value<std::string>(),
                "MODEL_VERSION_POLICY")
            ("nireq",
                "Number of parallel inference request executions for model. Recommended to be >= CPU_THROUGHPUT_STREAMS. Default value calculated at runtime: CPU cores / 8",
                cxxopts::value<uint>()->default_value(DEFAULT_NIREQ.c_str()),
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
