//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "cli_parser.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#include "capi_frontend/server_settings.hpp"
#include "graph_export/graph_cli_parser.hpp"
#include "ovms_exit_codes.hpp"
#include "version.hpp"

namespace ovms {

void CLIParser::parse(int argc, char** argv) {
    try {
        options = std::make_unique<cxxopts::Options>(argv[0], "OpenVINO Model Server");
        // Adding this option to parse unrecognised options in another parser
        options->allow_unrecognised_options();

        // clang-format off
        options->add_options()
            ("h, help",
                "Show this help message and exit")
            ("version",
                "Show binary version")
            ("port",
                "gRPC server port",
                cxxopts::value<uint32_t>()->default_value("0"),
                "PORT")
            ("grpc_bind_address",
                "Network interface address to bind to for the gRPC API",
                cxxopts::value<std::string>()->default_value("0.0.0.0"),
                "GRPC_BIND_ADDRESS")
            ("rest_port",
                "REST server port, the REST server will not be started if rest_port is blank or set to 0",
                cxxopts::value<uint32_t>()->default_value("0"),
                "REST_PORT")
            ("rest_bind_address",
                "Network interface address to bind to for the REST API",
                cxxopts::value<std::string>()->default_value("0.0.0.0"),
                "REST_BIND_ADDRESS")
            ("grpc_workers",
                "Number of gRPC servers. Default 1. Increase for multi client, high throughput scenarios",
                cxxopts::value<uint32_t>()->default_value("1"),
                "GRPC_WORKERS")
            ("grpc_max_threads",
                "Maximum number of threads which can be used by the gRPC server. Default value depends on number of CPUs.",
                cxxopts::value<uint32_t>(),
                "GRPC_MAX_THREADS")
            ("grpc_memory_quota",
                "GRPC server buffer memory quota. Default value set to 2147483648 (2GB).",
                cxxopts::value<size_t>(),
                "GRPC_MEMORY_QUOTA")
            ("rest_workers",
                "Number of worker threads in REST server - has no effect if rest_port is not set. Default value depends on number of CPUs. ",
                cxxopts::value<uint32_t>(),
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
                "A comma separated list of arguments to be passed to the gRPC server. (e.g. grpc.max_connection_age_ms=2000)",
                cxxopts::value<std::string>(), "GRPC_CHANNEL_ARGUMENTS")
            ("file_system_poll_wait_seconds",
                "Time interval between config and model versions changes detection. Default is 1. Zero or negative value disables changes monitoring.",
                cxxopts::value<uint32_t>()->default_value("1"),
                "FILE_SYSTEM_POLL_WAIT_SECONDS")
            ("sequence_cleaner_poll_wait_minutes",
                "Time interval between two consecutive sequence cleanup scans. Default is 5. Zero value disables sequence cleaner. It also sets the schedule for releasing free memory from the heap.",
                cxxopts::value<uint32_t>()->default_value("5"),
                "SEQUENCE_CLEANER_POLL_WAIT_MINUTES")
            ("custom_node_resources_cleaner_interval_seconds",
                "Time interval between two consecutive resources cleanup scans. Default is 1. Must be greater than 0.",
                cxxopts::value<uint32_t>()->default_value("1"),
                "CUSTOM_NODE_RESOURCES_CLEANER_INTERVAL_SECONDS")
            ("cache_dir",
                "Overrides model cache directory. By default cache files are saved into"
#ifdef __linux__
                 "/opt/cache"
#elif _WIN32
                 "c:\\Intel\\openvino_cache"
#endif
                 " if the directory is present. When enabled, first model load will produce cache files.",
                cxxopts::value<std::string>(),
                "CACHE_DIR")
            ("metrics_enable",
                "Flag enabling metrics endpoint on rest_port.",
                cxxopts::value<bool>()->default_value("false"),
                "METRICS")
            ("metrics_list",
                "Comma separated list of metrics. If unset, only default metrics will be enabled. Default metrics: ovms_requests_success, ovms_requests_fail, ovms_request_time_us, ovms_streams, ovms_inference_time_us, ovms_wait_for_infer_req_time_us. When set, only the listed metrics will be enabled. Optional metrics: ovms_infer_req_queue_size, ovms_infer_req_active.",
                cxxopts::value<std::string>()->default_value(""),
                "METRICS_LIST")
            ("cpu_extension",
                "A path to shared library containing custom CPU layer implementation. Default: empty.",
                cxxopts::value<std::string>()->default_value(""),
                "CPU_EXTENSION");

        options->add_options("multi model")
            ("config_path",
                "Absolute path to json configuration file",
                cxxopts::value<std::string>(), "CONFIG_PATH");

        options->add_options("pull hf model")
            ("pull",
                "Pull model from HF",
                cxxopts::value<bool>()->default_value("false"),
                "PULL_HF")
            ("source_model",
                "HF source model path",
                cxxopts::value<std::string>(),
                "HF_SOURCE")
            ("model_repository_path",
                "HF model destination download path",
                cxxopts::value<std::string>(),
                "MODEL_REPOSITORY_PATH")
            ("task",
                "Choose type of model export: text_generation - chat and completion endpoints, embeddings - embeddings endpoint, rerank - rerank endpoint.",
                cxxopts::value<std::string>()->default_value("text_generation"),
                "TASK");

        options->add_options("single model")
            ("model_name",
                "Name of the model",
                cxxopts::value<std::string>(),
                "MODEL_NAME")
            ("model_path",
                "Folder with AI model versions or with mediapipe graph",
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
                "A dictionary of plugin configuration keys and their values, eg \"{\\\"NUM_STREAMS\\\": \\\"1\\\"}\". Default number of streams is optimized to optimal latency with low concurrency.",
                cxxopts::value<std::string>(),
                "PLUGIN_CONFIG")
            ("stateful",
                "Flag indicating model is stateful",
                cxxopts::value<bool>()->default_value("false"),
                "STATEFUL")
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

        result = std::make_unique<cxxopts::ParseResult>(options->parse(argc, argv));

        if (result->unmatched().size()) {
            this->graphOptionsParser.parse(result->unmatched());
        }
#pragma warning(push)
#pragma warning(disable : 4129)
        if (result->count("version")) {
            std::string project_name(PROJECT_NAME);
            std::string project_version(PROJECT_VERSION);
            std::cout << project_name + " " + project_version << std::endl;
            std::cout << "OpenVINO backend " << OPENVINO_NAME << std::endl;
            std::cout << "Bazel build flags: " << BAZEL_BUILD_FLAGS << std::endl;
#pragma warning(pop)
            exit(OVMS_EX_OK);
        }

        if (result->count("help") || result->arguments().size() == 0) {
            std::cout << options->help({"", "multi model", "single model", "pull hf model"}) << std::endl;
            this->graphOptionsParser.printHelp();
            exit(OVMS_EX_OK);
        }
    } catch (const std::exception& e) {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(OVMS_EX_USAGE);
    }
}

void CLIParser::prepare(ServerSettingsImpl* serverSettings, ModelsSettingsImpl* modelsSettings) {
    if (nullptr == result) {
        throw std::logic_error("Tried to prepare server and model settings without parse result");
    }

    // Server settings
    serverSettings->startedWithCLI = true;
    serverSettings->grpcPort = result->operator[]("port").as<uint32_t>();
    serverSettings->restPort = result->operator[]("rest_port").as<uint32_t>();
    serverSettings->metricsEnabled = result->operator[]("metrics_enable").as<bool>();
    serverSettings->metricsList = result->operator[]("metrics_list").as<std::string>();
    serverSettings->filesystemPollWaitMilliseconds = result->operator[]("file_system_poll_wait_seconds").as<uint32_t>() * 1000;
    serverSettings->sequenceCleanerPollWaitMinutes = result->operator[]("sequence_cleaner_poll_wait_minutes").as<uint32_t>();
    serverSettings->resourcesCleanerPollWaitSeconds = result->operator[]("custom_node_resources_cleaner_interval_seconds").as<uint32_t>();
    serverSettings->grpcWorkers = result->operator[]("grpc_workers").as<uint32_t>();

    if (result->count("log_level"))
        serverSettings->logLevel = result->operator[]("log_level").as<std::string>();
    if (result->count("log_path"))
        serverSettings->logPath = result->operator[]("log_path").as<std::string>();

    if (result->count("grpc_channel_arguments"))
        serverSettings->grpcChannelArguments = result->operator[]("grpc_channel_arguments").as<std::string>();

    if (result != nullptr && result->count("cache_dir")) {
        serverSettings->cacheDir = result->operator[]("cache_dir").as<std::string>();
    }
    if (result->count("cpu_extension")) {
        serverSettings->cpuExtensionLibraryPath = result->operator[]("cpu_extension").as<std::string>();
    }

    if (result->count("grpc_bind_address"))
        serverSettings->grpcBindAddress = result->operator[]("grpc_bind_address").as<std::string>();

    if (result->count("rest_bind_address"))
        serverSettings->restBindAddress = result->operator[]("rest_bind_address").as<std::string>();

    if (result->count("grpc_max_threads"))
        serverSettings->grpcMaxThreads = result->operator[]("grpc_max_threads").as<uint32_t>();

    if (result->count("grpc_memory_quota"))
        serverSettings->grpcMemoryQuota = result->operator[]("grpc_memory_quota").as<size_t>();

    if (result->count("rest_workers"))
        serverSettings->restWorkers = result->operator[]("rest_workers").as<uint32_t>();

#if (PYTHON_DISABLE == 0)
        serverSettings->withPython = true;
#endif

#ifdef MTR_ENABLED
    if (result->count("trace_path"))
        serverSettings->tracePath = result->operator[]("trace_path").as<std::string>();
#endif
    // Ovms Pull models mode
    if (result->count("pull")) {
        serverSettings->hfSettings.pullHfModelMode = result->operator[]("pull").as<bool>();
        if (result->count("source_model"))
            serverSettings->hfSettings.sourceModel = result->operator[]("source_model").as<std::string>();
        if (result->count("model_repository_path"))
            serverSettings->hfSettings.downloadPath = result->operator[]("model_repository_path").as<std::string>();
        if (result->count("task")) {
            std::string task = result->operator[]("task").as<std::string>();
            if (task != "text_generation") {
                if (task != "embeddings") {
                    if (task != "rerank") {
                        throw std::logic_error("Error: --task parameter unsupported value: " + task);
                    }
                }
            }
            serverSettings->hfSettings.task = task;
        }
        this->graphOptionsParser.prepare(serverSettings, modelsSettings);
    } else {
        serverSettings->hfSettings.pullHfModelMode = false;
    }

    // Model settings
    if (result->count("model_name")) {
        modelsSettings->modelName = result->operator[]("model_name").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("model_name");
    }
    if (result->count("model_path")) {
        modelsSettings->modelPath = result->operator[]("model_path").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("model_name");
    }

    if (result->count("max_sequence_number")) {
        modelsSettings->maxSequenceNumber = result->operator[]("max_sequence_number").as<uint32_t>();
        modelsSettings->userSetSingleModelArguments.push_back("max_sequence_number");
    }

    if (result->count("batch_size")) {
        modelsSettings->batchSize = result->operator[]("batch_size").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("batch_size");
    }

    if (result->count("shape")) {
        modelsSettings->shape = result->operator[]("shape").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("shape");
    }

    if (result->count("layout")) {
        modelsSettings->layout = result->operator[]("layout").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("layout");
    }

    if (result->count("model_version_policy")) {
        modelsSettings->modelVersionPolicy = result->operator[]("model_version_policy").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("model_version_policy");
    }

    if (result->count("nireq")) {
        modelsSettings->nireq = result->operator[]("nireq").as<uint32_t>();
        modelsSettings->userSetSingleModelArguments.push_back("nireq");
    }

    if (result->count("target_device")) {
        modelsSettings->targetDevice = result->operator[]("target_device").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("target_device");
    }

    if (result->count("plugin_config")) {
        modelsSettings->pluginConfig = result->operator[]("plugin_config").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("plugin_config");
    }

    if (result->count("stateful")) {
        modelsSettings->stateful = result->operator[]("stateful").as<bool>();
        modelsSettings->userSetSingleModelArguments.push_back("stateful");
    }

    if (result->count("idle_sequence_cleanup")) {
        modelsSettings->idleSequenceCleanup = result->operator[]("idle_sequence_cleanup").as<bool>();
        modelsSettings->userSetSingleModelArguments.push_back("idle_sequence_cleanup");
    }

    if (result->count("low_latency_transformation")) {
        modelsSettings->lowLatencyTransformation = result->operator[]("low_latency_transformation").as<bool>();
        modelsSettings->userSetSingleModelArguments.push_back("low_latency_transformation");
    }

    if (result->count("config_path")) {
        modelsSettings->configPath = result->operator[]("config_path").as<std::string>();
        modelsSettings->userSetSingleModelArguments.push_back("config_path");
    }
}

}  // namespace ovms
