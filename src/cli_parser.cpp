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

#include <fstream>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <map>
#include <utility>
#include <vector>
#include <variant>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

#include "capi_frontend/server_settings.hpp"
#include "logging.hpp"
#include "graph_export/graph_cli_parser.hpp"
#include "graph_export/rerank_graph_cli_parser.hpp"
#include "graph_export/embeddings_graph_cli_parser.hpp"
#include "graph_export/t2s_graph_cli_parser.hpp"
#include "graph_export/s2t_graph_cli_parser.hpp"
#include "graph_export/image_generation_graph_cli_parser.hpp"
#include "ovms_exit_codes.hpp"
#include "filesystem/filesystem.hpp"
#include "filesystem/localfilesystem.hpp"
#include "pull_module/hf_env_vars.hpp"
#include "pull_module/curl_downloader.hpp"
#include "stringutils.hpp"
#include "version.hpp"

namespace ovms {

constexpr const char* CONFIG_MANAGEMENT_HELP_GROUP{"config management"};
constexpr const char* API_KEY_ENV_VAR{"API_KEY"};
constexpr const char* MODEL_CONFIG_FILENAME{"config.json"};

namespace {

const std::map<std::string, std::string> architectureToTask = {
    {"BertForSequenceClassification", "rerank"},
    {"BertModel", "embeddings"},
    {"CLIPTextModel", "image_generation"},
    {"FluxTransformer2DModel", "image_generation"},
    {"GemmaForCausalLM", "text_generation"},
    {"Gemma2ForCausalLM", "text_generation"},
    {"InternVLChatModel", "text_generation"},
    {"JinaBertModel", "embeddings"},
    {"LlamaForCausalLM", "text_generation"},
    {"MPNetModel", "embeddings"},
    {"MPTForCausalLM", "text_generation"},
    {"ParlerTTSForConditionalGeneration", "text2speech"},
    {"Phi3ForCausalLM", "text_generation"},
    {"Phi3VForCausalLM", "text_generation"},
    {"Qwen2ForCausalLM", "text_generation"},
    {"Qwen2ForSequenceClassification", "rerank"},
    {"Qwen2Model", "embeddings"},
    {"Qwen2VLForConditionalGeneration", "text_generation"},
    {"Qwen3ForCausalLM", "text_generation"},
    {"RobertaForSequenceClassification", "rerank"},
    {"RobertaModel", "embeddings"},
    {"SD3Transformer2DModel", "image_generation"},
    {"SeamlessM4TModel", "speech2text"},
    {"SeamlessM4Tv2Model", "speech2text"},
    {"SpeechT5ForTextToSpeech", "text2speech"},
    {"T5EncoderModel", "embeddings"},
    {"UNet2DConditionModel", "image_generation"},
    {"WhisperForConditionalGeneration", "speech2text"},
    {"XLMRobertaForSequenceClassification", "rerank"},
    {"XLMRobertaModel", "embeddings"},
};

std::string getEnvOrDefault(const char* envName, const std::string& defaultValue = "") {
    const char* envValue = std::getenv(envName);
    if (envValue == nullptr) {
        return defaultValue;
    }
    return envValue;
}

std::string ensureTrailingSlash(std::string path) {
    if (path.empty() || path.back() == '/') {
        return path;
    }
    path.push_back('/');
    return path;
}

std::string getTaskForArchitecture(const std::string& architecture) {
    const auto exactMatch = architectureToTask.find(architecture);
    if (exactMatch != architectureToTask.end()) {
        return exactMatch->second;
    }
    if (architecture == "WhisperForConditionalGeneration" || architecture.rfind("SeamlessM4T", 0) == 0) {
        return "speech2text";
    }
    if (endsWith(architecture, "ForTextToSpeech")) {
        return "text2speech";
    }
    if (endsWith(architecture, "ForSequenceClassification")) {
        return "rerank";
    }
    if (endsWith(architecture, "Transformer2DModel") || architecture == "UNet2DConditionModel" || architecture == "AutoencoderKL") {
        return "image_generation";
    }
    if (endsWith(architecture, "ForCausalLM") || endsWith(architecture, "ForConditionalGeneration")) {
        return "text_generation";
    }
    if (endsWith(architecture, "EncoderModel") || endsWith(architecture, "Model")) {
        return "embeddings";
    }
    return "";
}

std::string determineTaskFromArchitectures(const rapidjson::Value& architecturesNode) {
    if (!architecturesNode.IsArray() || architecturesNode.Empty()) {
        throw std::logic_error("config.json does not contain a non-empty architectures array");
    }
    std::optional<std::string> resolvedTask;
    for (const auto& architecture : architecturesNode.GetArray()) {
        if (!architecture.IsString()) {
            continue;
        }
        std::string task = getTaskForArchitecture(architecture.GetString());
        if (task.empty()) {
            continue;
        }
        if (!resolvedTask.has_value()) {
            resolvedTask = task;
            continue;
        }
        if (resolvedTask.value() != task) {
            throw std::logic_error("config.json architectures map to multiple default tasks");
        }
    }
    if (!resolvedTask.has_value()) {
        throw std::logic_error("config.json architectures do not map to a supported default task");
    }
    return resolvedTask.value();
}

std::string determineTaskFromConfigStream(std::istream& configStream, const std::string& configSourceDescription) {
    rapidjson::Document configJson;
    rapidjson::IStreamWrapper wrapper(configStream);
    configJson.ParseStream(wrapper);
    if (configJson.HasParseError()) {
        throw std::logic_error("failed to parse " + configSourceDescription + ": " + std::string(rapidjson::GetParseError_En(configJson.GetParseError())));
    }
    if (!configJson.HasMember("architectures")) {
        throw std::logic_error(configSourceDescription + " does not contain architectures field");
    }
    return determineTaskFromArchitectures(configJson["architectures"]);
}

std::string determineTaskFromConfigContents(const std::string& configContents, const std::string& configSourceDescription) {
    rapidjson::Document configJson;
    configJson.Parse(configContents.c_str());
    if (configJson.HasParseError()) {
        throw std::logic_error("failed to parse " + configSourceDescription + ": " + std::string(rapidjson::GetParseError_En(configJson.GetParseError())));
    }
    if (!configJson.HasMember("architectures")) {
        throw std::logic_error(configSourceDescription + " does not contain architectures field");
    }
    return determineTaskFromArchitectures(configJson["architectures"]);
}

}  // namespace

std::string getConfigPath(const std::string& configPath) {
    bool isDir = false;
    auto status = LocalFileSystem::isDir(configPath, &isDir);
    if (!status.ok()) {
        throw std::logic_error("Invalid path for the config: " + configPath);
    }
    if (isDir) {
        return FileSystem::joinPath({configPath, "config.json"});
    }
    return configPath;
}

std::string CLIParser::determineDefaultTaskParameter(const std::optional<std::string>& modelPath, const std::optional<std::string>& sourceModel, const std::optional<std::string>& modelRepositoryPath) {
    if (modelPath.has_value() && !modelPath->empty()) {
        const auto configPath = std::filesystem::path(*modelPath) / MODEL_CONFIG_FILENAME;
        std::ifstream configFile(configPath);
        if (!configFile.is_open()) {
            throw std::logic_error("failed to open model config file: " + configPath.string());
        }
        return determineTaskFromConfigStream(configFile, configPath.string());
    }

    if (!sourceModel.has_value() || sourceModel->empty()) {
        throw std::logic_error("cannot determine default --task without model_path or source_model");
    }

    if (modelRepositoryPath.has_value() && !modelRepositoryPath->empty()) {
        const auto localModelDirectory = std::filesystem::path(*modelRepositoryPath) / *sourceModel;
        if (std::filesystem::exists(localModelDirectory)) {
            const auto configPath = localModelDirectory / MODEL_CONFIG_FILENAME;
            std::ifstream configFile(configPath);
            if (!configFile.is_open()) {
                throw std::logic_error("failed to open model config file: " + configPath.string());
            }
            return determineTaskFromConfigStream(configFile, configPath.string());
        }
    }

    std::string responseBody;
    const std::string hfEndpoint = ensureTrailingSlash(getEnvOrDefault(HF_ENDPOINT_ENV_VAR, DEFAULT_HF_ENDPOINT));
    const std::string configUrl = hfEndpoint + *sourceModel + "/resolve/main/" + MODEL_CONFIG_FILENAME;
    const auto status = fetchUrlToString(configUrl, getEnvOrDefault(HF_TOKEN_ENV_VAR), responseBody);
    if (!status.ok()) {
        throw std::logic_error("failed to download model config file from: " + configUrl);
    }
    return determineTaskFromConfigContents(responseBody, configUrl);
}

std::string CLIParser::getEffectiveTaskParameter() const {
    if (result->count("task")) {
        const auto task = result->operator[]("task").as<std::string>();
        SPDLOG_DEBUG("Effective task parameter specified by user: {}", task);
        return task;
    }
    if (inferredTaskParameter.has_value()) {
        SPDLOG_DEBUG("Effective task parameter using inferred default: {}", inferredTaskParameter.value());
        return inferredTaskParameter.value();
    }
    throw std::logic_error("error parsing options - --task parameter wasn't passed");
}

std::variant<bool, std::pair<int, std::string>> CLIParser::parse(int argc, char** argv) {
    std::stringstream ss;
    try {
        options = std::make_unique<cxxopts::Options>(argv[0], "OpenVINO Model Server");
        auto configOptions = std::make_unique<cxxopts::Options>("ovms --model_name <MODEL_NAME> --add_to_config --config_path <CONFIG_PATH> --model_repository_path <MODEL_REPO_PATH> \n  ovms --model_path <MODEL_PATH> --model_name <MODEL_NAME> --add_to_config --config_path <CONFIG_PATH> \n  ovms --remove_from_config --config_path <CONFIG_PATH> --model_name <MODEL_NAME>", "config management commands:");
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
            ("verbose_response",
                "When enabled, responses include an extra "
                "\"__verbose\" object with additional debug information.",
                cxxopts::value<bool>()->default_value("false"),
                "VERBOSE_RESPONSE")
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
            ("custom_node_resources_cleaner_interval_seconds",
                "Time interval between two consecutive resources cleanup scans. Default is 300. Zero value disables resources cleaner.",
                cxxopts::value<uint32_t>()->default_value("300"),
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
                "CPU_EXTENSION")
            ("allowed_media_domains",
                "Comma separated list of media domains from which URLs can be used as input for LLMs. Set to \"all\" to disable this restriction.",
                cxxopts::value<std::vector<std::string>>(),
                "ALLOWED_MEDIA_DOMAINS")
            ("allowed_local_media_path",
                "Path to directory that contains multimedia files that can be used as input for LLMs.",
                cxxopts::value<std::string>(),
                "ALLOWED_LOCAL_MEDIA_PATH")
            ("allow_credentials",
                "Flag enabling credentials on the API.",
                cxxopts::value<bool>()->default_value("false"),
                "ALLOW_CREDENTIALS")
            ("allowed_origins",
                "Comma separated list of origins that are allowed to access the API. Default: *.",
                cxxopts::value<std::string>()->default_value("*"),
                "ALLOWED_ORIGINS")
            ("allowed_methods",
                "Comma separated list of methods that are allowed to access the API. Default: *.",
                cxxopts::value<std::string>()->default_value("*"),
                "ALLOWED_METHODS")
            ("allowed_headers",
                "Comma separated list of headers that are allowed to access the API. Default: *.",
                cxxopts::value<std::string>()->default_value("*"),
                "ALLOWED_HEADERS")
            ("api_key_file",
                "path to the text file containing API key for authentication for generative endpoints. If not set, authentication is disabled.",
                cxxopts::value<std::string>()->default_value(""),
                "API_KEY");

        options->add_options("multi model")
            ("config_path",
                "Absolute path to json configuration file",
                cxxopts::value<std::string>(), "CONFIG_PATH");

        options->add_options(CONFIG_MANAGEMENT_HELP_GROUP)
            ("list_models",
                "Directive to show available servables in models repository",
                cxxopts::value<bool>()->default_value("false"),
                "LIST_MODELS")
            ("add_to_config",
                "Directive to add a model to configuration file. This parameter should be executed with --model_name, --config_path and either with --model_path or --model_repository_path.",
                cxxopts::value<bool>()->default_value("false"),
                "ADD_TO_CONFIG")
            ("remove_from_config",
                "Directive to remove a model from configuration file. This parameter should be executed with --config_path and --model_name to specify which model to remove.",
                cxxopts::value<bool>()->default_value("false"),
                "REMOVE_FROM_CONFIG");

        // Set default value for model_repository_path from environment variable if it exists and is not empty
        std::string defaultModelRepoPath = "";
        std::string defaultConfigPath = "";
        const char* envModelRepoPath = std::getenv("OVMS_MODEL_REPOSITORY_PATH");
        if (envModelRepoPath != nullptr && std::string(envModelRepoPath).length() > 0) {
            defaultModelRepoPath = envModelRepoPath;
            defaultConfigPath = std::string(envModelRepoPath) + "/config.json";
        }

        options->add_options("pull hf model")
            ("pull",
            "Pull model from HF. Uses optional environment variables: HF_TOKEN - when set used for authentication, HF_ENDPOINT - when set replaces huggingface.co for model download.",
            cxxopts::value<bool>()->default_value("false"),
            "PULL_HF")
            ("source_model",
            "HF source model path",
            cxxopts::value<std::string>(),
            "HF_SOURCE")
            ("source_loras",
            "LoRA adapters for image generation. Format: alias1=org1/repo1,alias2=org2/repo2@file.safetensors,alias3=https://url/file.safetensors,alias4=/local/path/file.safetensors",
            cxxopts::value<std::string>(),
            "SOURCE_LORAS")
            ("gguf_filename",
            "Name of the GGUF file",
            cxxopts::value<std::string>(),
            "GGUF_FILENAME")
            ("overwrite_models",
            "Overwrite the model if it already exists in the models repository",
            cxxopts::value<bool>()->default_value("false"),
            "OVERWRITE_MODELS")
            ("model_repository_path",
            "HF model destination download path",
            cxxopts::value<std::string>()->default_value(defaultModelRepoPath),
            "MODEL_REPOSITORY_PATH")
            ("weight-format",
            "Model precision used in optimum-cli export with conversion",
            cxxopts::value<std::string>()->default_value("int8"),
            "WEIGHT_FORMAT")
            ("extra_quantization_params",
                "Model quantization parameters used in optimum-cli export with conversion for text generation models",
                cxxopts::value<std::string>(),
                "EXTRA_QUANTIZATION_PARAMS")
            ("vocoder",
                "The vocoder model to use for text2speech. For example microsoft/speecht5_hifigan",
                cxxopts::value<std::string>(),
                "VOCODER");

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
                "Resets model layout. It should be in format <TARGET_LAYOUT>:<SOURCE_LAYOUT> e.g. NCHW:NHWC",
                cxxopts::value<std::string>(),
                "LAYOUT")
            ("mean",
                "Resets model mean.",
                cxxopts::value<std::string>(),
                "MEAN")
            ("scale",
                "Resets model scale.",
                cxxopts::value<std::string>(),
                "SCALE")
            ("color_format",
                "Resets model color format. It should be in format <TARGET_COLOR_FORMAT>:<SOURCE_COLOR_FORMAT> e.g. BGR:RGB",
                cxxopts::value<std::string>(),
                "COLOR_FORMAT")
            ("precision",
                "Resets model precision.",
                cxxopts::value<std::string>(),
                "PRECISION")
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
                "PLUGIN_CONFIG");

        options->add_options("generative task (applies to: pull hf model, single model)")
            ("task",
                "Specifies the generative task for the local model. If not provided, default task value is inferred from model config.json architectures. It should be followed by task specific parameters. Supported tasks: text_generation, embeddings, rerank, image_generation, text2speech, speech2text. It creates the pipeline graph in memory based on the provided task-specific options.",
                cxxopts::value<std::string>(),
                "TASK");
        configOptions->custom_help("");
        configOptions->add_options(CONFIG_MANAGEMENT_HELP_GROUP)
            ("list_models",
                "Directive to show available servables in models repository",
                cxxopts::value<bool>()->default_value("false"),
                "LIST_MODELS")
            ("add_to_config",
                "Directive to add a model to configuration file. This parameter should be executed with --model_name, --config_path and either with --model_path or --model_repository_path.",
                cxxopts::value<bool>()->default_value("false"),
                "ADD_TO_CONFIG")
            ("remove_from_config",
                "Directive to remove a model from configuration file. This parameter should be executed with --config_path and --model_name to specify which model to remove.",
                cxxopts::value<bool>()->default_value("false"),
                "REMOVE_FROM_CONFIG")
            ("model_repository_path",
                "Absolute or relative path from the config directory to the model repository",
                cxxopts::value<std::string>()->default_value(defaultModelRepoPath),
                "MODEL_REPOSITORY_PATH")
            ("model_path",
                "Absolute or relative path from the config directory to the model. By default is a combination of the model_repository_path and model_name.",
                cxxopts::value<std::string>(),
                "MODEL_PATH")
            ("model_name",
                "Name of the model",
                cxxopts::value<std::string>(),
                "MODEL_NAME")
            ("config_path",
                "Path to json configuration file",
                cxxopts::value<std::string>()->default_value(defaultConfigPath),
                "CONFIG_PATH");

        result = std::make_unique<cxxopts::ParseResult>(options->parse(argc, argv));

        const bool isConfigManagementFlow =
            result->count("add_to_config") || result->count("remove_from_config") || result->count("list_models");
        if (!result->count("task") &&
            !result->count("pull") &&
            !result->count("source_model") &&
            result->count("model_path") &&
            !isConfigManagementFlow) {
            const std::optional<std::string> modelPath = std::make_optional(result->operator[]("model_path").as<std::string>());
            const auto configPath = std::filesystem::path(*modelPath) / MODEL_CONFIG_FILENAME;
            if (std::filesystem::exists(configPath)) {
                inferredTaskParameter = determineDefaultTaskParameter(modelPath, std::nullopt, std::nullopt);
                SPDLOG_DEBUG("Default task parameter inferred from model config: {}", inferredTaskParameter.value());
            }
        }

        // HF pull mode or pull and start mode or starting from local folder with graph created in memory
        if (isHFPullOrPullAndStart(this->result) || isInMemoryGraphMode(this->result)) {
            std::vector<std::string> unmatchedOptions;
            GraphExportType task;
            std::string taskValue;
            if (!result->count("task")) {
                const std::optional<std::string> modelPath = result->count("model_path") ? std::make_optional(result->operator[]("model_path").as<std::string>()) : std::nullopt;
                const std::optional<std::string> sourceModel = result->count("source_model") ? std::make_optional(result->operator[]("source_model").as<std::string>()) : std::nullopt;
                const std::optional<std::string> modelRepositoryPath = result->count("model_repository_path") ? std::make_optional(result->operator[]("model_repository_path").as<std::string>()) : std::nullopt;
                inferredTaskParameter = determineDefaultTaskParameter(modelPath, sourceModel, modelRepositoryPath);
                SPDLOG_DEBUG("Default task parameter inferred from model config: {}", inferredTaskParameter.value());
            }
            taskValue = getEffectiveTaskParameter();
            task = stringToEnum(taskValue);
            if (task != UNKNOWN_GRAPH) {
                switch (task) {
                    case TEXT_GENERATION_GRAPH: {
                        GraphCLIParser cliParser;
                        unmatchedOptions = cliParser.parse(result->unmatched());
                        this->graphOptionsParser = std::move(cliParser);
                        break;
                    }
                    case EMBEDDINGS_GRAPH: {
                        EmbeddingsGraphCLIParser cliParser;
                        unmatchedOptions = cliParser.parse(result->unmatched());
                        this->graphOptionsParser = std::move(cliParser);
                        break;
                    }
                    case RERANK_GRAPH: {
                        RerankGraphCLIParser cliParser;
                        unmatchedOptions = cliParser.parse(result->unmatched());
                        this->graphOptionsParser = std::move(cliParser);
                        break;
                    }
                    case IMAGE_GENERATION_GRAPH: {
                        ImageGenerationGraphCLIParser cliParser;
                        unmatchedOptions = cliParser.parse(result->unmatched());
                        this->graphOptionsParser = std::move(cliParser);
                        break;
                    }
                    case TEXT_TO_SPEECH_GRAPH: {
                        TextToSpeechGraphCLIParser cliParser;
                        unmatchedOptions = cliParser.parse(result->unmatched());
                        this->graphOptionsParser = std::move(cliParser);
                        break;
                    }
                    case SPEECH_TO_TEXT_GRAPH: {
                        SpeechToTextGraphCLIParser cliParser;
                        unmatchedOptions = cliParser.parse(result->unmatched());
                        this->graphOptionsParser = std::move(cliParser);
                        break;
                    }
                    case UNKNOWN_GRAPH: {
                        ss << "error parsing options - --task parameter unsupported value: " + taskValue;
                        return std::make_pair(OVMS_EX_USAGE, ss.str());
                    }
                }
            } else {
                ss << "error parsing options - --task parameter unsupported value: " + taskValue;
                return std::make_pair(OVMS_EX_USAGE, ss.str());
            }

            if (unmatchedOptions.size()) {
                ss << "task: " << enumToString(task) << " - error parsing options - unmatched arguments : ";
                for (auto& argument : unmatchedOptions) {
                    ss << argument << ", ";
                }
                ss << std::endl;
                return std::make_pair(OVMS_EX_USAGE, ss.str());
            }
        } else if (result->unmatched().size()){
            ss << "error parsing options - unmatched arguments: ";
            for (auto& argument : result->unmatched()) {
                ss << argument << ", ";
            }
            ss << std::endl;
            return std::make_pair(OVMS_EX_USAGE, ss.str());
        }
        if (isHFPullOrPullAndStart(this->result) && result->count("list_models")) {
            ss << "error parsing options - --list_models cannot be used with --pull or --task" << std::endl;
            return std::make_pair(OVMS_EX_USAGE, ss.str());
        }
        if (isHFPullOrPullAndStart(this->result) && result->count("remove_from_config")) {
            ss << "error parsing options - --remove_from_config cannot be used with --pull or --task" << std::endl;
            return std::make_pair(OVMS_EX_USAGE, ss.str());
        }
        if (isHFPullOrPullAndStart(this->result) && result->count("add_to_config")) {
            ss << "error parsing options - --add_to_config cannot be used with --pull or --task" << std::endl;
            return std::make_pair(OVMS_EX_USAGE, ss.str());
        }
        if (result->count("add_to_config") && result->count("list_models")) {
            ss << "error parsing options - --list_models cannot be used with --add_to_config" << std::endl;
            return std::make_pair(OVMS_EX_USAGE, ss.str());
        }
        if (result->count("remove_from_config") && result->count("list_models")) {
            ss << "error parsing options - --list_models cannot be used with --remove_from_config" << std::endl;
            return std::make_pair(OVMS_EX_USAGE, ss.str());
        }
        if (result->count("remove_from_config") && result->count("model_path")) {
            ss << "error parsing options - --model_path cannot be used with --remove_from_config" << std::endl;
            return std::make_pair(OVMS_EX_USAGE, ss.str());
        }
#pragma warning(push)
#pragma warning(disable : 4129)
        if (result->count("version")) {
            std::string project_name(PROJECT_NAME);
            std::string project_version(PROJECT_VERSION);
            ss << project_name + " " + project_version << std::endl;
            ss << "OpenVINO backend " << ovms::getOpenVINOVersion() << std::endl;
            const char* genaiVersion = ovms::getGenAIVersion();
            if (genaiVersion[0] != '\0') {
                ss << "OpenVINO GenAI backend " << genaiVersion << std::endl;
            }
            ss << "Bazel build flags: " << BAZEL_BUILD_FLAGS << std::endl;
#pragma warning(pop)
            return std::make_pair(OVMS_EX_OK, ss.str());
        }

        if (result->count("help") || result->arguments().size() == 0) {
            ss << options->help({"", "multi model", "single model", "pull hf model"}) << std::endl;
            ss << configOptions->help({CONFIG_MANAGEMENT_HELP_GROUP}) << std::endl;
            GraphCLIParser parser1;
            RerankGraphCLIParser parser2;
            EmbeddingsGraphCLIParser parser3;
            ImageGenerationGraphCLIParser imageGenParser;
            TextToSpeechGraphCLIParser ttsParser;
            SpeechToTextGraphCLIParser sttParser;
            parser1.printHelp();
            parser2.printHelp();
            parser3.printHelp();
            imageGenParser.printHelp();
            return std::make_pair(OVMS_EX_OK, ss.str());
        }

        return true;
    } catch (const std::exception& e) {
        ss << "error parsing options: " << e.what() << std::endl;
        return std::make_pair(OVMS_EX_USAGE, ss.str());
    }
}

void CLIParser::prepareServer(ServerSettingsImpl& serverSettings) {
    // Server settings
    serverSettings.startedWithCLI = true;
    // list models mode
    if (result->count("list_models")) {
        serverSettings.serverMode = LIST_MODELS_MODE;
        std::cout << "Listing models in repository..." << std::endl;
        serverSettings.hfSettings.downloadPath = result->operator[]("model_repository_path").as<std::string>();
        std::cout << "Model repository path: " << serverSettings.hfSettings.downloadPath << std::endl;
        return;
    }

    std::string defaultConfigPath = "";
    const char* envModelRepoPath = std::getenv("OVMS_MODEL_REPOSITORY_PATH");
    if (envModelRepoPath != nullptr && std::string(envModelRepoPath).length() > 0) {
        defaultConfigPath = std::string(envModelRepoPath) + "/config.json";
    }
    if (result->count("add_to_config")) {
        serverSettings.serverMode = MODIFY_CONFIG_MODE;
        serverSettings.exportConfigType = ENABLE_MODEL;
    }

    if (result->count("remove_from_config")) {
        serverSettings.serverMode = MODIFY_CONFIG_MODE;
        serverSettings.exportConfigType = DISABLE_MODEL;
    }

    serverSettings.grpcPort = result->operator[]("port").as<uint32_t>();
    serverSettings.restPort = result->operator[]("rest_port").as<uint32_t>();
    serverSettings.metricsEnabled = result->operator[]("metrics_enable").as<bool>();
    serverSettings.metricsList = result->operator[]("metrics_list").as<std::string>();
    serverSettings.filesystemPollWaitMilliseconds = result->operator[]("file_system_poll_wait_seconds").as<uint32_t>() * 1000;

    serverSettings.resourcesCleanerPollWaitSeconds = result->operator[]("custom_node_resources_cleaner_interval_seconds").as<uint32_t>();
    serverSettings.grpcWorkers = result->operator[]("grpc_workers").as<uint32_t>();

    if (result->count("log_level"))
        serverSettings.logLevel = result->operator[]("log_level").as<std::string>();
    if (result->count("log_path"))
        serverSettings.logPath = result->operator[]("log_path").as<std::string>();
    if (result->count("verbose_response"))
        serverSettings.verboseResponse = result->operator[]("verbose_response").as<bool>();

    if (result->count("grpc_channel_arguments"))
        serverSettings.grpcChannelArguments = result->operator[]("grpc_channel_arguments").as<std::string>();

    if (result != nullptr && result->count("cache_dir")) {
        serverSettings.cacheDir = result->operator[]("cache_dir").as<std::string>();
    }
    if (result->count("cpu_extension")) {
        serverSettings.cpuExtensionLibraryPath = result->operator[]("cpu_extension").as<std::string>();
    }
    if (result->count("allowed_media_domains")) {
        serverSettings.allowedMediaDomains = result->operator[]("allowed_media_domains").as<std::vector<std::string>>();
    }
    if (result->count("allowed_local_media_path")) {
        serverSettings.allowedLocalMediaPath = FileSystem::normalizeConfiguredPath(result->operator[]("allowed_local_media_path").as<std::string>());
    }

    if (result->count("grpc_bind_address"))
        serverSettings.grpcBindAddress = result->operator[]("grpc_bind_address").as<std::string>();

    if (result->count("rest_bind_address"))
        serverSettings.restBindAddress = result->operator[]("rest_bind_address").as<std::string>();

    if (result->count("grpc_max_threads"))
        serverSettings.grpcMaxThreads = result->operator[]("grpc_max_threads").as<uint32_t>();

    if (result->count("grpc_memory_quota"))
        serverSettings.grpcMemoryQuota = result->operator[]("grpc_memory_quota").as<size_t>();

    if (result->count("rest_workers"))
        serverSettings.restWorkers = result->operator[]("rest_workers").as<uint32_t>();

#if (PYTHON_DISABLE == 0)
        serverSettings.withPython = true;
#endif

#ifdef MTR_ENABLED
    if (result->count("trace_path"))
        serverSettings.tracePath = result->operator[]("trace_path").as<std::string>();
#endif

    serverSettings.allowCredentials = result->operator[]("allow_credentials").as<bool>();
    serverSettings.allowedOrigins = result->operator[]("allowed_origins").as<std::string>();
    serverSettings.allowedMethods = result->operator[]("allowed_methods").as<std::string>();
    serverSettings.allowedHeaders = result->operator[]("allowed_headers").as<std::string>();
    std::filesystem::path apiKeyFile = result->operator[]("api_key_file").as<std::string>();
    serverSettings.apiKey = "";
    if (!apiKeyFile.empty()) {
        std::ifstream file(apiKeyFile);
        if (file.is_open()) {
            std::getline(file, serverSettings.apiKey);
            // Use first line and trim whitespace characters from both ends
            size_t endpos = serverSettings.apiKey.find_last_not_of(" \n\r\t");
            if (endpos != std::string::npos) {
                serverSettings.apiKey = serverSettings.apiKey.substr(0, endpos + 1);
            }
            file.close();
        } else {
            throw std::filesystem::filesystem_error("Error reading API key file: Unable to open file ", apiKeyFile, std::error_code(2, std::generic_category()));
        }
    } else {
        const char* envApiKey = std::getenv(API_KEY_ENV_VAR);
        if (envApiKey != nullptr) {
            serverSettings.apiKey = envApiKey;
        }
    }
}

void CLIParser::prepareModel(ModelsSettingsImpl& modelsSettings, HFSettingsImpl& hfSettings) {
    // Model settings
    if (result->count("model_name")) {
        modelsSettings.modelName = result->operator[]("model_name").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("model_name");
    }
    if (result->count("model_path")) {
        modelsSettings.modelPath = result->operator[]("model_path").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("model_path");
    }


    if (result->count("batch_size")) {
        modelsSettings.batchSize = result->operator[]("batch_size").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("batch_size");
    }

    if (result->count("shape")) {
        modelsSettings.shape = result->operator[]("shape").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("shape");
    }

    if (result->count("layout")) {
        modelsSettings.layout = result->operator[]("layout").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("layout");
    }

    if (result->count("mean")) {
        if (modelsSettings.layout.empty()) {
            throw std::logic_error("error parsing options - --mean parameter requires --layout to be set");
        }
        modelsSettings.mean = result->operator[]("mean").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("mean");
    }

    if (result->count("scale")) {
        if (modelsSettings.layout.empty()) {
            throw std::logic_error("error parsing options - --scale parameter requires --layout to be set");
        }
        modelsSettings.scale = result->operator[]("scale").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("scale");
    }

    if (result->count("color_format")) {
        if (modelsSettings.layout.empty()) {
            throw std::logic_error("error parsing options - --color_format parameter requires --layout to be set");
        }
        modelsSettings.colorFormat = result->operator[]("color_format").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("color_format");
    }

    if (result->count("precision")) {
        if (modelsSettings.layout.empty()) {
            throw std::logic_error("error parsing options - --precision parameter requires --layout to be set");
        }
        modelsSettings.precision = result->operator[]("precision").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("precision");
    }

    if (result->count("model_version_policy")) {
        modelsSettings.modelVersionPolicy = result->operator[]("model_version_policy").as<std::string>();
        modelsSettings.userSetSingleModelArguments.push_back("model_version_policy");
    }

    if (result->count("nireq")) {
        modelsSettings.nireq = result->operator[]("nireq").as<uint32_t>();
        modelsSettings.userSetSingleModelArguments.push_back("nireq");
    }

    if (result->count("target_device")) {
        modelsSettings.targetDevice = result->operator[]("target_device").as<std::string>();
        if (isHFPullOrPullAndStart(this->result)) {
            hfSettings.exportSettings.targetDevice = modelsSettings.targetDevice;
        } else {
            modelsSettings.userSetSingleModelArguments.push_back("target_device");
        }
    }

    if (result->count("plugin_config")) {
        modelsSettings.pluginConfig = result->operator[]("plugin_config").as<std::string>();
        hfSettings.exportSettings.pluginConfig.manualString = modelsSettings.pluginConfig;
        modelsSettings.userSetSingleModelArguments.push_back("plugin_config");
    }
    if (result->count("config_path")) {
            modelsSettings.configPath = result->operator[]("config_path").as<std::string>();
            modelsSettings.userSetSingleModelArguments.push_back("config_path");
    }
}

bool CLIParser::isHFPullOrPullAndStart(const std::unique_ptr<cxxopts::ParseResult>& result) {
    // Keep `--task` in the broad mutually exclusive task/pull CLI category so
    // parse-time checks that rely on this helper continue to reject combining
    // task-based flows with config-management modes. More specific mode
    // differentiation is handled by isInMemoryGraphMode().
    return (result->count("pull") || result->count("task") || result->count("source_model"));
}

bool CLIParser::isInMemoryGraphMode(const std::unique_ptr<cxxopts::ParseResult>& result) {
    if (result->count("source_model") || result->count("pull")) {
        return false;
    }
    return result->count("task") || inferredTaskParameter.has_value();
}

void CLIParser::prepareGraph(ServerSettingsImpl& serverSettings, HFSettingsImpl& hfSettings, const std::string& modelName) {
    // Always propagate source_model so validation can detect misuse
    if (result->count("source_model")) {
        hfSettings.sourceModel = result->operator[]("source_model").as<std::string>();
    }
    // Ovms Pull models mode || pull and start models mode
    if (isHFPullOrPullAndStart(this->result) || isInMemoryGraphMode(this->result)) {
        if (isInMemoryGraphMode(this->result)) {
            serverSettings.serverMode = IN_MEMORY_GRAPH_MODE;
        } else if (result->count("pull")) {
            serverSettings.serverMode = HF_PULL_MODE;
        } else {
            serverSettings.serverMode = HF_PULL_AND_START_MODE;
        }
        if (result->count("gguf_filename")) {
            hfSettings.ggufFilename = result->operator[]("gguf_filename").as<std::string>();
            hfSettings.downloadType = GGUF_DOWNLOAD;
        }
        if (result->count("overwrite_models")) {
            hfSettings.overwriteModels = result->operator[]("overwrite_models").as<bool>();
        }
        if (result->count("source_model")) {
            // Already set above, but keep the original flow for downloadType logic
            hfSettings.sourceModel = result->operator[]("source_model").as<std::string>();
        } else if (result->count("model_name") && !result->count("model_path")) {
            // Only use model_name as source_model when model_path is not set
            // (when model_path is set, user wants to use local model without HF pull)
            hfSettings.sourceModel = result->operator[]("model_name").as<std::string>();
        }
        if (result->count("source_loras")) {
            hfSettings.sourceLoras = result->operator[]("source_loras").as<std::string>();
        }
        if ((result->count("weight-format") || result->count("extra_quantization_params")) && isOptimumCliDownload(hfSettings.sourceModel, hfSettings.ggufFilename)) {
            hfSettings.downloadType = OPTIMUM_CLI_DOWNLOAD;
        }
        if (result->count("weight-format") && hfSettings.downloadType == GIT_CLONE_DOWNLOAD) {
            throw std::logic_error("--weight-format parameter unsupported for OpenVINO models.");
        }
        if (result->count("extra_quantization_params") && hfSettings.downloadType == GIT_CLONE_DOWNLOAD) {
            throw std::logic_error("--extra_quantization_params parameter unsupported for OpenVINO models.");
        }

        if (result->count("weight-format"))
            hfSettings.exportSettings.precision = result->operator[]("weight-format").as<std::string>();
        if (result->count("extra_quantization_params"))
            hfSettings.exportSettings.extraQuantizationParams = result->operator[]("extra_quantization_params").as<std::string>();
        if (result->count("vocoder"))
            hfSettings.exportSettings.vocoder = result->operator[]("vocoder").as<std::string>();
        hfSettings.exportSettings.restWorkers = serverSettings.restWorkers;
        hfSettings.downloadPath = result->operator[]("model_repository_path").as<std::string>();
        // When --task is used with --model_path but without --pull/--source_model,
        // use model_path as the model location (no HF download needed)
        if (!result->count("pull") && !result->count("source_model") && result->count("model_path")) {
            const auto configuredModelPath = std::filesystem::path(result->operator[]("model_path").as<std::string>());
            hfSettings.exportSettings.modelPath = std::filesystem::absolute(configuredModelPath).lexically_normal().string();
            SPDLOG_DEBUG("Using local absolute model path for graph export: {}", hfSettings.exportSettings.modelPath);
        }
        const std::string taskValue = getEffectiveTaskParameter();
        if (!taskValue.empty()) {
            hfSettings.task = stringToEnum(taskValue);
            switch (hfSettings.task) {
                case TEXT_GENERATION_GRAPH: {
                    if (std::holds_alternative<GraphCLIParser>(this->graphOptionsParser)) {
                        std::get<GraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
                    } else {
                        throw std::logic_error("Tried to prepare graph settings without graph parser initialization");
                    }
                    break;
                }
                case EMBEDDINGS_GRAPH: {
                    if (std::holds_alternative<EmbeddingsGraphCLIParser>(this->graphOptionsParser)) {
                        std::get<EmbeddingsGraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
                    } else {
                        throw std::logic_error("Tried to prepare graph settings without graph parser initialization");
                    }
                    break;
                }
                case RERANK_GRAPH: {
                    if (std::holds_alternative<RerankGraphCLIParser>(this->graphOptionsParser)) {
                        std::get<RerankGraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
                    } else {
                        throw std::logic_error("Tried to prepare graph settings without graph parser initialization");
                    }
                    break;
                }
                case IMAGE_GENERATION_GRAPH: {
                    if (std::holds_alternative<ImageGenerationGraphCLIParser>(this->graphOptionsParser)) {
                        std::get<ImageGenerationGraphCLIParser>(this->graphOptionsParser).prepare(serverSettings, hfSettings, modelName);
                    } else {
                        throw std::logic_error("Tried to prepare graph settings without graph parser initialization");
                    }
                    break;
                }
                case TEXT_TO_SPEECH_GRAPH: {
                    if (std::holds_alternative<TextToSpeechGraphCLIParser>(this->graphOptionsParser)) {
                        std::get<TextToSpeechGraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
                    } else {
                        throw std::logic_error("Tried to prepare graph settings without graph parser initialization");
                    }
                    break;
                }
                case SPEECH_TO_TEXT_GRAPH: {
                    if (std::holds_alternative<SpeechToTextGraphCLIParser>(this->graphOptionsParser)) {
                        std::get<SpeechToTextGraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
                    } else {
                        throw std::logic_error("Tried to prepare graph settings without graph parser initialization");
                    }
                    break;
                }
                case UNKNOWN_GRAPH: {
                    throw std::logic_error("Error: --task parameter unsupported value: " + taskValue);
                    break;
                }
            }
        } else {
            if (std::holds_alternative<GraphCLIParser>(this->graphOptionsParser)) {
                std::get<GraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
            } else {
                throw std::logic_error("Tried to prepare graph settings without graph parser initialization");
            }
        }
        if (!serverSettings.cacheDir.empty()) {
            hfSettings.exportSettings.pluginConfig.cacheDir = serverSettings.cacheDir;
        }

    // No pull nor pull and start mode and no start with local model_path
    } else {
        if (result->count("weight-format")) {
            throw std::logic_error("--weight-format parameter unsupported for Openvino huggingface organization models.");
        }
        if (result->count("extra_quantization_params")) {
            throw std::logic_error("--extra_quantization_params parameter unsupported for Openvino huggingface organization models.");
        }
    }
}

void CLIParser::prepareConfigExport(ModelsSettingsImpl& modelsSettings) {
    // Export config.json mode
    if (result->count("model_name")) {
        modelsSettings.modelName = result->operator[]("model_name").as<std::string>();
    }
    if (result->count("model_path")) {
        modelsSettings.modelPath = result->operator[]("model_path").as<std::string>();
    } else if (!result->operator[]("model_repository_path").as<std::string>().empty() && result->count("model_name")) {
        modelsSettings.modelPath = FileSystem::joinPath({result->operator[]("model_repository_path").as<std::string>(), modelsSettings.modelName});
    }
    std::string defaultConfigPath = "";
    const char* envModelRepoPath = std::getenv("OVMS_MODEL_REPOSITORY_PATH");
    if (envModelRepoPath != nullptr && std::string(envModelRepoPath).length() > 0) {
        defaultConfigPath = FileSystem::joinPath({std::string(envModelRepoPath), "config.json"});
    }
    if (result->count("add_to_config") || result->count("remove_from_config")) {
        if (result->count("config_path")) {
            modelsSettings.configPath = result->operator[]("config_path").as<std::string>();
            modelsSettings.userSetSingleModelArguments.push_back("config_path");
        } else {
            modelsSettings.configPath = defaultConfigPath;
            std::cout << "Using default config path: " << modelsSettings.configPath << std::endl;
        }
    }
}

void CLIParser::prepareGraphStart(HFSettingsImpl& hfSettings, ModelsSettingsImpl& modelsSettings) {
    // Ovms pull and start models mode
    // Model settings
    if (result->count("model_name")) {
        modelsSettings.modelName = result->operator[]("model_name").as<std::string>();
    } else if (!hfSettings.sourceModel.empty()) {
        modelsSettings.modelName = hfSettings.sourceModel;
    }

    // Only override modelPath if it wasn't already set via --model_path
    if (!result->count("model_path")) {
        modelsSettings.modelPath = FileSystem::joinPath({hfSettings.downloadPath, hfSettings.sourceModel});
    }
}

void CLIParser::prepare(ServerSettingsImpl* serverSettings, ModelsSettingsImpl* modelsSettings) {
    if (nullptr == result) {
        throw std::logic_error("Tried to prepare server and model settings without parse result");
    }
    this->prepareServer(*serverSettings);

    this->prepareModel(*modelsSettings, serverSettings->hfSettings);

    this->prepareGraph(*serverSettings, serverSettings->hfSettings, modelsSettings->modelName);

    if (serverSettings->serverMode == HF_PULL_AND_START_MODE)
        this->prepareGraphStart(serverSettings->hfSettings, *modelsSettings);

    if (serverSettings->exportConfigType != UNKNOWN_MODEL)
        this->prepareConfigExport(*modelsSettings);
}

}  // namespace ovms
