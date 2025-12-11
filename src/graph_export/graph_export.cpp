//***************************************************************************
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
#include "graph_export.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#pragma warning(pop)

#include "../capi_frontend/server_settings.hpp"
#include "../config.hpp"
#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../schema.hpp"
#include "../version.hpp"

#if (MEDIAPIPE_DISABLE == 0)
#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#endif
namespace ovms {

static const std::string OVMS_VERSION_GRAPH_LINE = std::string("# File created with: ") + PROJECT_NAME + std::string(" ") + PROJECT_VERSION + std::string("\n");

static std::string constructModelsPath(const std::string& modelPath, const std::optional<std::string>& ggufFilenameOpt) {
    std::string modelsPath;
    if (ggufFilenameOpt.has_value()) {
        modelsPath = FileSystem::joinPath({modelPath, ggufFilenameOpt.value()});
#if _WIN32
        // On Windows, file paths use backslashes ('\') as separators. However, the graph parser used in this project expects Unix-style paths with forward slashes ('/').
        // If Windows-style backslashes are present, the parser may fail to locate files or misinterpret the path. To ensure compatibility, we replace all backslashes with forward slashes.
        // This is safe because Windows APIs accept forward slashes in file paths.
        if (FileSystem::getOsSeparator() != "/") {
            std::replace(modelsPath.begin(), modelsPath.end(), '\\', '/');
        }
#endif
    } else {
        modelsPath = modelPath;
    }
    SPDLOG_TRACE("Models path: {}, modelPath:{}, ggufFilenameOpt:{}", modelsPath, modelPath, ggufFilenameOpt.value_or("std::nullopt"));
    return modelsPath;
}

std::string GraphExport::getDraftModelDirectoryName(std::string draftModel) {
    std::replace(draftModel.begin(), draftModel.end(), '/', '-');
    return draftModel;
}

std::string GraphExport::getDraftModelDirectoryPath(const std::string& directoryPath, const std::string& draftModel) {
    std::string fullPath = FileSystem::joinPath({directoryPath, GraphExport::getDraftModelDirectoryName(draftModel)});
    return fullPath;
}
#define GET_PLUGIN_CONFIG_OPT_OR_FAIL_AND_RETURN(EXPORT_SETTINGS)                 \
    auto pluginConfigOrStatus = GraphExport::createPluginString(EXPORT_SETTINGS); \
    if (std::holds_alternative<Status>(pluginConfigOrStatus)) {                   \
        auto status = std::get<Status>(pluginConfigOrStatus);                     \
        SPDLOG_ERROR("Failed to create plugin config: {}", status.string());      \
        return status;                                                            \
    }                                                                             \
    auto pluginConfigOpt = std::get<std::optional<std::string>>(pluginConfigOrStatus)

static Status createPbtxtFile(const std::string& directoryPath, const std::string& pbtxtContent) {
#if (MEDIAPIPE_DISABLE == 0)
    ::mediapipe::CalculatorGraphConfig config;
    SPDLOG_TRACE("Generated pbtxt: {}", pbtxtContent);
    bool success = ::google::protobuf::TextFormat::ParseFromString(pbtxtContent, &config);
    if (!success) {
        SPDLOG_ERROR("Created graph config file couldn't be parsed - check used task parameters values.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
#endif
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, pbtxtContent);
}

static Status createTextGenerationGraphTemplate(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (!std::holds_alternative<TextGenGraphSettingsImpl>(hfSettings.graphSettings)) {
        SPDLOG_ERROR("Graph options not initialized for text generation.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto& graphSettings = std::get<TextGenGraphSettingsImpl>(hfSettings.graphSettings);
    auto& ggufFilename = hfSettings.ggufFilename;
    auto& exportSettings = hfSettings.exportSettings;

    std::ostringstream oss;
    oss << OVMS_VERSION_GRAPH_LINE;
    std::string modelsPath = constructModelsPath(exportSettings.modelPath, ggufFilename);
    SPDLOG_TRACE("modelsPath: {}, directoryPath: {}, ggufFilename: {}", modelsPath, directoryPath, ggufFilename.value_or("std::nullopt"));
    GET_PLUGIN_CONFIG_OPT_OR_FAIL_AND_RETURN(exportSettings);
    // clang-format off
    oss << R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:)"
        << graphSettings.maxNumSeqs << R"(,
            device: ")"
        << exportSettings.targetDevice << R"(",
            models_path: ")"
        << modelsPath << R"(",
            )";
    if (pluginConfigOpt.has_value()) {
        oss << R"(plugin_config: ')"
        << pluginConfigOpt.value() << R"(',
            )";
    }
    oss << R"(enable_prefix_caching: )"
        << graphSettings.enablePrefixCaching << R"(,
            cache_size: )"
        << graphSettings.cacheSize << R"(,)";
    if (graphSettings.pipelineType.has_value()) {
        oss << R"(
            pipeline_type: )" << graphSettings.pipelineType.value() << R"(,)";
    }
    if (graphSettings.maxNumBatchedTokens.has_value()) {
        oss << R"(
            max_num_batched_tokens: )" << graphSettings.maxNumBatchedTokens.value() << R"(,)";
    }
    if (graphSettings.reasoningParser.has_value()) {
        oss << R"(
            reasoning_parser: ")" << graphSettings.reasoningParser.value() << R"(",)";
    }
    if (graphSettings.toolParser.has_value()) {
        oss << R"(
            tool_parser: ")" << graphSettings.toolParser.value() << R"(",)";
    }
    if (graphSettings.enableToolGuidedGeneration == "true") {
        oss << R"(
            enable_tool_guided_generation: true,)";
    }
    if (graphSettings.dynamicSplitFuse != "true") {
        oss << R"(
            dynamic_split_fuse: false,)";
    }
    if (graphSettings.draftModelDirName.has_value()) {
        oss << R"(
            # Speculative decoding configuration)";
        oss << R"(
            draft_models_path: ")" << GraphExport::getDraftModelDirectoryName(graphSettings.draftModelDirName.value()) << R"(",)";
    }
    oss << R"(
        }
    }
    input_stream_handler {
        input_stream_handler: "SyncSetInputStreamHandler",
        options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
            sync_set {
            tag_index: "LOOPBACK:0"
            }
        }
        }
    }
    })";
    return createPbtxtFile(directoryPath, oss.str());
}

static Status createRerankGraphTemplate(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (!std::holds_alternative<RerankGraphSettingsImpl>(hfSettings.graphSettings)) {
        SPDLOG_ERROR("Graph options not initialized for reranking.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto& graphSettings = std::get<RerankGraphSettingsImpl>(hfSettings.graphSettings);
    auto& ggufFilename = hfSettings.ggufFilename;
    auto& exportSettings = hfSettings.exportSettings;

    std::ostringstream oss;
    oss << OVMS_VERSION_GRAPH_LINE;
    // Windows path creation - graph parser needs forward slashes in paths
    std::string modelsPath = constructModelsPath(exportSettings.modelPath, ggufFilename);
    SPDLOG_TRACE("modelsPath: {}, directoryPath: {}, ggufFilename: {}", modelsPath, directoryPath, ggufFilename.value_or("std::nullopt"));
    GET_PLUGIN_CONFIG_OPT_OR_FAIL_AND_RETURN(exportSettings);
    // clang-format off
    oss << R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: ")"
    << exportSettings.modelName << R"(",
    calculator: "RerankCalculatorOV"
    input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {
            models_path: ")"
            << modelsPath << R"(",
            max_allowed_chunks: )"
            << graphSettings.maxAllowedChunks << R"(,
            target_device: ")" << exportSettings.targetDevice << R"(",
            )";
    if (pluginConfigOpt.has_value()) {
        oss << R"(plugin_config: ')" << pluginConfigOpt.value()  << R"(',)";
    }
    oss << R"(
        }
    }
})";
    return createPbtxtFile(directoryPath, oss.str());
}

static Status createEmbeddingsGraphTemplate(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (!std::holds_alternative<EmbeddingsGraphSettingsImpl>(hfSettings.graphSettings)) {
        SPDLOG_ERROR("Graph options not initialized for embeddings.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto& graphSettings = std::get<EmbeddingsGraphSettingsImpl>(hfSettings.graphSettings);
    auto& ggufFilename = hfSettings.ggufFilename;
    auto& exportSettings = hfSettings.exportSettings;

    std::ostringstream oss;
    oss << OVMS_VERSION_GRAPH_LINE;
    std::string modelsPath = constructModelsPath(exportSettings.modelPath, ggufFilename);
    SPDLOG_TRACE("modelsPath: {}, directoryPath: {}, ggufFilename: {}", modelsPath, directoryPath, ggufFilename.value_or("std::nullopt"));
    GET_PLUGIN_CONFIG_OPT_OR_FAIL_AND_RETURN(exportSettings);
    // clang-format off
    oss << R"(
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: ")"
    << exportSettings.modelName << R"(",
    calculator: "EmbeddingsCalculatorOV"
    input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {
            models_path: ")"
            << modelsPath << R"(",
            normalize_embeddings: )"
            << graphSettings.normalize << R"(,
            truncate: )"
            << graphSettings.truncate << R"(,
            pooling: )"
            << graphSettings.pooling << R"(,
            target_device: ")" << exportSettings.targetDevice << R"(",
            )";
    if (pluginConfigOpt.has_value()) {
        oss << R"(plugin_config: ')" << pluginConfigOpt.value() << R"(',
        )";
    }
    oss << R"(}
    }
})";
    return createPbtxtFile(directoryPath, oss.str());
}

static Status createTextToSpeechGraphTemplate(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (!std::holds_alternative<TextToSpeechGraphSettingsImpl>(hfSettings.graphSettings)) {
        SPDLOG_ERROR("Graph options not initialized for speech generation.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto& ggufFilename = hfSettings.ggufFilename;
    auto& exportSettings = hfSettings.exportSettings;

    std::ostringstream oss;
    oss << OVMS_VERSION_GRAPH_LINE;
    std::string modelsPath = constructModelsPath(exportSettings.modelPath, ggufFilename);
    SPDLOG_TRACE("modelsPath: {}, directoryPath: {}, ggufFilename: {}", modelsPath, directoryPath, ggufFilename.value_or("std::nullopt"));
    GET_PLUGIN_CONFIG_OPT_OR_FAIL_AND_RETURN(exportSettings);
    // clang-format off
    oss << R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
    name: ")"
    << exportSettings.modelName << R"("
    calculator: "T2sCalculator"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {
            models_path: ")"
            << modelsPath << R"("
            target_device: ")" << exportSettings.targetDevice << R"("
            )";
    if (pluginConfigOpt.has_value()) {
        oss << R"(plugin_config: ')" << pluginConfigOpt.value() << R"('
        )";
    }
    oss << R"(}
    }
})";

#if (MEDIAPIPE_DISABLE == 0)
    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created text2speech graph config couldn't be parsed.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
#endif
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status createSpeechToTextGraphTemplate(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (!std::holds_alternative<SpeechToTextGraphSettingsImpl>(hfSettings.graphSettings)) {
        SPDLOG_ERROR("Graph options not initialized for speech to text.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto& ggufFilename = hfSettings.ggufFilename;
    auto& exportSettings = hfSettings.exportSettings;

    std::ostringstream oss;
    oss << OVMS_VERSION_GRAPH_LINE;
    std::string modelsPath = constructModelsPath(exportSettings.modelPath, ggufFilename);
    SPDLOG_TRACE("modelsPath: {}, directoryPath: {}, ggufFilename: {}", modelsPath, directoryPath, ggufFilename.value_or("std::nullopt"));
    GET_PLUGIN_CONFIG_OPT_OR_FAIL_AND_RETURN(exportSettings);
    // clang-format off
    oss << R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
    name: ")"
    << exportSettings.modelName << R"("
    calculator: "S2tCalculator"
    input_side_packet: "STT_NODE_RESOURCES:s2t_servable"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.S2tCalculatorOptions]: {
            models_path: ")"
            << modelsPath << R"("
            target_device: ")" << exportSettings.targetDevice << R"("
            )";
    if (pluginConfigOpt.has_value()) {
        oss << R"(plugin_config: ')" << pluginConfigOpt.value() << R"('
        )";
    }
    oss << R"(}
    }
})";
#if (MEDIAPIPE_DISABLE == 0)
    ::mediapipe::CalculatorGraphConfig config;
    bool success = ::google::protobuf::TextFormat::ParseFromString(oss.str(), &config);
    if (!success) {
        SPDLOG_ERROR("Created speech2text graph config couldn't be parsed.");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
#endif
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "graph.pbtxt"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

static Status createImageGenerationGraphTemplate(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (!std::holds_alternative<ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings)) {
        SPDLOG_ERROR("Graph options not initialized for image generation.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto& graphSettings = std::get<ImageGenerationGraphSettingsImpl>(hfSettings.graphSettings);
    auto& exportSettings = hfSettings.exportSettings;
    auto& ggufFilename = hfSettings.ggufFilename;
    std::string modelsPath = constructModelsPath(exportSettings.modelPath, ggufFilename);
    SPDLOG_TRACE("modelsPath: {}, directoryPath: {}, ggufFilename: {}", modelsPath, directoryPath, ggufFilename.value_or("std::nullopt"));
    GET_PLUGIN_CONFIG_OPT_OR_FAIL_AND_RETURN(exportSettings);

    std::ostringstream oss;
    oss << OVMS_VERSION_GRAPH_LINE;
    // clang-format off
    oss << R"(
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: ")" << modelsPath << R"("
          device: ")" << exportSettings.targetDevice << R"(")";
    if (pluginConfigOpt.has_value()) {
        oss << R"(
          plugin_config: ')" << pluginConfigOpt.value() << R"(')";
    }

    if (graphSettings.resolution.size()) {
        oss << R"(
          resolution: ")" << graphSettings.resolution << R"(")";
    }

    if (graphSettings.numImagesPerPrompt.has_value()) {
        oss << R"(
          num_images_per_prompt: )" << graphSettings.numImagesPerPrompt.value();
    }

    if (graphSettings.guidanceScale.has_value()) {
        oss << R"(
          guidance_scale: )" << graphSettings.guidanceScale.value();
    }

    if (graphSettings.maxResolution.size()) {
        oss << R"(
          max_resolution: ")" << graphSettings.maxResolution << R"(")";
    }

    if (graphSettings.defaultResolution.size()) {
        oss << R"(
          default_resolution: ")" << graphSettings.defaultResolution << R"(")";
    }

    if (graphSettings.maxNumberImagesPerPrompt.has_value()) {
        oss << R"(
          max_num_images_per_prompt: )" << graphSettings.maxNumberImagesPerPrompt.value();
    }

    if (graphSettings.defaultNumInferenceSteps.has_value()) {
        oss << R"(
          default_num_inference_steps: )" << graphSettings.defaultNumInferenceSteps.value();
    }

    if (graphSettings.maxNumInferenceSteps.has_value()) {
        oss << R"(
          max_num_inference_steps: )" << graphSettings.maxNumInferenceSteps.value();
    }

    oss << R"(
      }
  }
}
)";
    // clang-format on
    return createPbtxtFile(directoryPath, oss.str());
}

GraphExport::GraphExport() {
}

Status GraphExport::createServableConfig(const std::string& directoryPath, const HFSettingsImpl& hfSettings) {
    if (directoryPath.empty()) {
        SPDLOG_ERROR("Directory path empty: {}", directoryPath);
        return StatusCode::PATH_INVALID;
    }

    bool exists = false;
    auto status = LocalFileSystem::exists(directoryPath, &exists);
    if (!status.ok())
        return status;
    if (!hfSettings.ggufFilename.has_value()) {
        bool is_dir = false;
        status = LocalFileSystem::isDir(directoryPath, &is_dir);
        if (!status.ok()) {
            SPDLOG_ERROR("Failed to check if graph path is directory: {}: {}", directoryPath, status.string());
            return status;
        }
        if (!is_dir) {
            SPDLOG_ERROR("Graph path is not a directory: {}", directoryPath);
            return StatusCode::PATH_INVALID;
        }
    }
    if (hfSettings.task == TEXT_GENERATION_GRAPH) {
        return createTextGenerationGraphTemplate(directoryPath, hfSettings);
    } else if (hfSettings.task == EMBEDDINGS_GRAPH) {
        return createEmbeddingsGraphTemplate(directoryPath, hfSettings);
    } else if (hfSettings.task == RERANK_GRAPH) {
        return createRerankGraphTemplate(directoryPath, hfSettings);
    } else if (hfSettings.task == IMAGE_GENERATION_GRAPH) {
        return createImageGenerationGraphTemplate(directoryPath, hfSettings);
    } else if (hfSettings.task == TEXT_TO_SPEECH_GRAPH) {
        return createTextToSpeechGraphTemplate(directoryPath, hfSettings);
    } else if (hfSettings.task == SPEECH_TO_TEXT_GRAPH) {
        return createSpeechToTextGraphTemplate(directoryPath, hfSettings);
    } else if (hfSettings.task == UNKNOWN_GRAPH) {
        SPDLOG_ERROR("Graph options not initialized.");
        return StatusCode::INTERNAL_ERROR;
    }
    SPDLOG_ERROR("Graph options not initialized.");
    return StatusCode::INTERNAL_ERROR;
}

std::variant<std::optional<std::string>, Status> GraphExport::createPluginString(const ExportSettings& exportSettings) {
    bool configNotEmpty = false;
    auto& stringPluginConfig = exportSettings.pluginConfig.manualString;
    auto& pluginConfig = exportSettings.pluginConfig;
    SPDLOG_TRACE("Creating plugin config string from export settings. Manual string: {}, pluginConfig.numStreams: {}, pluginConfig.kvCachePrecision: {}, pluginConfig.maxPromptLength: {}, pluginConfig.modelDistributionPolicy: {}, pluginConfig.cacheDir: {}", pluginConfig.manualString.value_or("std::nullopt"), pluginConfig.numStreams.value_or(0), pluginConfig.kvCachePrecision.value_or("std::nullopt"), pluginConfig.maxPromptLength.value_or(0), pluginConfig.modelDistributionPolicy.value_or("std::nullopt"), exportSettings.pluginConfig.cacheDir.value_or("std::nullopt"));
    rapidjson::Document d;
    d.SetObject();
    if (stringPluginConfig.has_value() && !stringPluginConfig.value().empty()) {
        configNotEmpty = true;
        if (d.Parse(stringPluginConfig.value().c_str()).HasParseError()) {
            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
        }
    }
    if (pluginConfig.kvCachePrecision.has_value()) {
        rapidjson::Value name;
        name.SetString(pluginConfig.kvCachePrecision.value().c_str(), d.GetAllocator());
        auto itr = d.FindMember("KV_CACHE_PRECISION");
        if (itr != d.MemberEnd()) {
            return Status(StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS, "Doubled KV_CACHE_PRECISION parameter in plugin config.");
        }
        d.AddMember("KV_CACHE_PRECISION", name, d.GetAllocator());
        configNotEmpty = true;
    }
    if (pluginConfig.maxPromptLength.has_value()) {
        rapidjson::Value value;
        value.SetUint(pluginConfig.maxPromptLength.value());
        auto itr = d.FindMember("MAX_PROMPT_LEN");
        if (itr != d.MemberEnd()) {
            return Status(StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS, "Doubled MAX_PROMPT_LEN parameter in plugin config.");
        }
        d.AddMember("MAX_PROMPT_LEN", value, d.GetAllocator());
        configNotEmpty = true;
    }
    if (pluginConfig.modelDistributionPolicy.has_value()) {
        rapidjson::Value value;
        value.SetString(pluginConfig.modelDistributionPolicy.value().c_str(), d.GetAllocator());
        auto itr = d.FindMember("MODEL_DISTRIBUTION_POLICY");
        if (itr != d.MemberEnd()) {
            return Status(StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS, "Doubled MODEL_DISTRIBUTION_POLICY parameter in plugin config.");
        }
        d.AddMember("MODEL_DISTRIBUTION_POLICY", value, d.GetAllocator());
        configNotEmpty = true;
    }
    if (pluginConfig.numStreams.has_value()) {
        rapidjson::Value value;
        // convert pluginConfig.numStreams.value() to string
        std::string numStreamsStr = std::to_string(pluginConfig.numStreams.value());
        value.SetString(numStreamsStr.c_str(), d.GetAllocator());
        auto itr = d.FindMember("NUM_STREAMS");
        if (itr != d.MemberEnd()) {
            if (pluginConfig.numStreams.value() == 1) {
                // ignoring double setting NUM_STREAMS is required for embeddings & rerank
                // since 1 is default value coming from CLI
                SPDLOG_DEBUG("Doubled NUM_STREAMS parameter in plugin config. Will ignore `--num_streams` CLI parameter.");
            } else {
                return Status(StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS, "Doubled NUM_STREAMS parameter in plugin config.");
            }
        } else {
            d.AddMember("NUM_STREAMS", value, d.GetAllocator());
            configNotEmpty = true;
        }
    }
    if (exportSettings.pluginConfig.cacheDir.has_value()) {
        rapidjson::Value value;
        value.SetString(exportSettings.pluginConfig.cacheDir.value().c_str(), d.GetAllocator());
        auto itr = d.FindMember("CACHE_DIR");
        if (itr != d.MemberEnd()) {
            return Status(StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS, "Doubled CACHE_DIR parameter in plugin config.");
        }
        d.AddMember("CACHE_DIR", value, d.GetAllocator());
        configNotEmpty = true;
    }
    if (pluginConfig.useNpuPrefixCaching.has_value()) {
        rapidjson::Value value;
        value.SetBool(pluginConfig.useNpuPrefixCaching.value());
        auto itr = d.FindMember("NPUW_LLM_ENABLE_PREFIX_CACHING");
        if (itr != d.MemberEnd()) {
            return Status(StatusCode::PLUGIN_CONFIG_CONFLICTING_PARAMETERS, "Doubled NPUW_LLM_ENABLE_PREFIX_CACHING parameter in plugin config.");
        }
        d.AddMember("NPUW_LLM_ENABLE_PREFIX_CACHING", value, d.GetAllocator());
        configNotEmpty = true;
    }
    if (configNotEmpty) {
        // Serialize the document to a JSON string
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);

        // Output the JSON string
        return buffer.GetString();
    } else {
        return std::nullopt;
    }
}

}  // namespace ovms
