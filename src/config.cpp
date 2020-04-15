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
#include "config.h"

namespace ovms {

Config& Config::parse(int argc, char** argv) {
    try
    {
        options = std::make_unique<cxxopts::Options>(argv[0], "OpenVINO Model Server");

        options->add_options()
            ("h, help",
                "show this help message and exit");
        options->add_options("config")
            ("config_path",
                "absolute path to json configuration file",
                cxxopts::value<std::string>(), "CONFIG_PATH")
            ("port",
                "gRPC server port",
                cxxopts::value<uint16_t>()->default_value("9178"),
                "PORT")
            ("rest_port",
                "REST server port, the REST server will not be started if rest_port is blank or set to 0",
                 cxxopts::value<uint16_t>()->default_value("0"),
                 "REST_PORT")
            ("grpc_workers",
                "number of workers in gRPC server",
                cxxopts::value<uint>()->default_value("24"),
                "GRPC_WORKERS")
            ("rest_workers",
                "number of workers in REST server - has no effect if rest_port is not set",
                cxxopts::value<uint>()->default_value("24"),
                "REST_WORKERS");
        options->add_options("model")
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
                cxxopts::value<size_t>()->default_value("0"),
                "BATCH_SIZE")
            ("shape",
                "sets models shape (model must support reshaping). If set, batch_size parameter is ignored",
                 cxxopts::value<std::vector<size_t>>(),
                  "SHAPE")
            ("model_version_policy",
                "model version policy",
                cxxopts::value<std::string>(),
                "MODEL_VERSION_POLICY")
            ("nireq",
                "Number of parallel inference request executions for model. Default: 1",
                cxxopts::value<uint>()->default_value("1"),
                "NIREQ")
            ("target_device",
                "name of the model",
                cxxopts::value<std::string>()->default_value("CPU"),
                "TARGET_DEVICE")
            ("plugin_config",
                "a dictionary of plugin configuration keys and their values",
                cxxopts::value<std::string>(),
                "PLUGIN_CONFIG");

        result = std::make_unique<cxxopts::ParseResult>(options->parse(argc, argv));

        if (result->count("help"))
        {
            std::cout << options->help({"", "config", "model"}) << std::endl;       
            exit(1);
        }

        validate();
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    return instance();
}

bool Config::validate() {
    // cannot set both config path & model_name/model_path
    if (result->count("config_path") && (result->count("model_name") || result->count("model_path"))) {
        std::cerr << "Use either config_path or model_path with model_name" << std::endl;
        exit(2);
    }

    // port and rest_port cannot be the same
    if (port() == restPort()) {
        std::cerr << "port and rest_port cannot have the same values" << std::endl;
        exit(3);
    }

    return true;
}

} // namespace ovms 
