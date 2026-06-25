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
#include "default_task.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6313)
#endif
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "pull_module/curl_downloader.hpp"
#include "pull_module/hf_env_vars.hpp"
#include "status.hpp"
#include "stringutils.hpp"

namespace ovms {

static constexpr const char* MODEL_CONFIG_FILENAME = "config.json";
static constexpr const char* MODEL_INDEX_FILENAME = "model_index.json";

static const std::map<std::string, std::string> architectureToTask = {
    {"BertForSequenceClassification", "rerank"},
    {"BertModel", "embeddings"},
    {"CLIPTextModel", "image_generation"},
    {"FluxTransformer2DModel", "image_generation"},
    {"InternVLChatModel", "text_generation"},
    {"JinaBertModel", "embeddings"},
    {"MPNetModel", "embeddings"},
    {"ParlerTTSForConditionalGeneration", "text2speech"},
    {"Qwen2ForSequenceClassification", "rerank"},
    {"Qwen2Model", "embeddings"},
    {"Qwen3ASRForConditionalGeneration", "speech2text"},
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

// architecture: {default task, {task, pattern}}
static const std::map<std::string, std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> questionableArchitectureTaskKeywords = {
    {"Qwen3ForCausalLM", {"text_generation", {{"rerank", "rerank"}, {"embeddings", "embed"}}}},
};

static std::string getEnvOrDefault(const char* envName, const std::string& defaultValue = "") {
    const char* envValue = std::getenv(envName);
    if (envValue == nullptr) {
        return defaultValue;
    }
    return envValue;
}

static std::string ensureTrailingSlash(std::string path) {
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

std::string getTaskForQuestionableArchitecture(const std::string& architecture, const std::string& normalizedModelIdentifier) {
    const auto architectureRules = questionableArchitectureTaskKeywords.find(architecture);
    if (architectureRules == questionableArchitectureTaskKeywords.end()) {
        return "";
    }
    const auto& [defaultTask, patternRules] = architectureRules->second;
    for (const auto& [task, keyword] : patternRules) {
        if (normalizedModelIdentifier.find(keyword) != std::string::npos) {
            return task;
        }
    }
    return defaultTask;
}

static std::string determineTaskFromArchitectures(const rapidjson::Value& architecturesNode, const std::string& modelIdentifier) {
    if (!architecturesNode.IsArray() || architecturesNode.Empty()) {
        throw std::logic_error("config.json does not contain a non-empty architectures array");
    }
    const std::string normalizedModelIdentifier = toLower(modelIdentifier);
    std::optional<std::string> resolvedTask;
    for (const auto& architecture : architecturesNode.GetArray()) {
        if (!architecture.IsString()) {
            continue;
        }
        const std::string architectureName = architecture.GetString();
        std::string task = getTaskForQuestionableArchitecture(architectureName, normalizedModelIdentifier);
        if (task.empty() && questionableArchitectureTaskKeywords.find(architectureName) == questionableArchitectureTaskKeywords.end()) {
            task = getTaskForArchitecture(architectureName);
        }
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

static std::string determineTaskFromNullArchitectures(const rapidjson::Document& configJson, const std::string& configSourceDescription) {
    if (configJson.HasMember("n_mels")) {
        return "text2speech";
    }
    throw std::logic_error(configSourceDescription + " has null architectures and does not contain recognized special fields for task detection");
}

std::string determineTaskFromConfigStream(std::istream& configStream, const std::string& configSourceDescription, const std::string& modelIdentifier) {
    rapidjson::Document configJson;
    rapidjson::IStreamWrapper wrapper(configStream);
    configJson.ParseStream(wrapper);
    if (configJson.HasParseError()) {
        throw std::logic_error("failed to parse " + configSourceDescription + ": " + std::string(rapidjson::GetParseError_En(configJson.GetParseError())));
    }
    if (!configJson.HasMember("architectures")) {
        throw std::logic_error(configSourceDescription + " does not contain architectures field");
    }
    const auto& architecturesNode = configJson["architectures"];
    if (architecturesNode.IsNull()) {
        return determineTaskFromNullArchitectures(configJson, configSourceDescription);
    }
    return determineTaskFromArchitectures(architecturesNode, modelIdentifier);
}

std::string determineTaskFromConfigContents(const std::string& configContents, const std::string& configSourceDescription, const std::string& modelIdentifier) {
    rapidjson::Document configJson;
    configJson.Parse(configContents.c_str());
    if (configJson.HasParseError()) {
        throw std::logic_error("failed to parse " + configSourceDescription + ": " + std::string(rapidjson::GetParseError_En(configJson.GetParseError())));
    }
    if (!configJson.HasMember("architectures")) {
        throw std::logic_error(configSourceDescription + " does not contain architectures field");
    }
    const auto& architecturesNode = configJson["architectures"];
    if (architecturesNode.IsNull()) {
        return determineTaskFromNullArchitectures(configJson, configSourceDescription);
    }
    return determineTaskFromArchitectures(architecturesNode, modelIdentifier);
}

std::string determineTaskFromModelIndex(std::istream& indexStream, const std::string& indexSourceDescription) {
    rapidjson::Document indexJson;
    rapidjson::IStreamWrapper wrapper(indexStream);
    indexJson.ParseStream(wrapper);
    if (indexJson.HasParseError()) {
        throw std::logic_error("failed to parse " + indexSourceDescription + ": " + std::string(rapidjson::GetParseError_En(indexJson.GetParseError())));
    }
    if (!indexJson.HasMember("_class_name") || !indexJson["_class_name"].IsString()) {
        throw std::logic_error(indexSourceDescription + " does not contain a valid _class_name field");
    }
    const std::string className = indexJson["_class_name"].GetString();
    if (className.find("StableDiffusion") != std::string::npos || className.find("Flux") != std::string::npos) {
        return "image_generation";
    }
    throw std::logic_error(indexSourceDescription + " _class_name '" + className + "' does not map to a supported default task");
}

bool graphPbtxtExists(const std::string& modelPath) {
    const auto graphPath = std::filesystem::path(modelPath) / "graph.pbtxt";
    return std::filesystem::exists(graphPath);
}

bool hasTaskSpecificParameters(const std::vector<std::string>& unmatchedOptions) {
    return !unmatchedOptions.empty();
}

std::string determineDefaultTaskParameter(const std::optional<std::string>& modelPath, const std::optional<std::string>& sourceModel, const std::optional<std::string>& modelRepositoryPath) {
    if (modelPath.has_value() && !modelPath->empty()) {
        const auto configPath = std::filesystem::path(*modelPath) / MODEL_CONFIG_FILENAME;
        std::ifstream configFile(configPath);
        if (configFile.is_open()) {
            return determineTaskFromConfigStream(configFile, configPath.string(), *modelPath);
        }
        const auto indexPath = std::filesystem::path(*modelPath) / MODEL_INDEX_FILENAME;
        std::ifstream indexFile(indexPath);
        if (indexFile.is_open()) {
            return determineTaskFromModelIndex(indexFile, indexPath.string());
        }
        throw std::logic_error("failed to open model config file: " + configPath.string() + " or " + indexPath.string());
    }

    if (!sourceModel.has_value() || sourceModel->empty()) {
        throw std::logic_error("cannot determine default --task without model_path or source_model");
    }

    if (modelRepositoryPath.has_value() && !modelRepositoryPath->empty()) {
        const auto localModelDirectory = std::filesystem::path(*modelRepositoryPath) / *sourceModel;
        if (std::filesystem::exists(localModelDirectory)) {
            const auto configPath = localModelDirectory / MODEL_CONFIG_FILENAME;
            std::ifstream configFile(configPath);
            if (configFile.is_open()) {
                return determineTaskFromConfigStream(configFile, configPath.string(), *sourceModel);
            }
            const auto indexPath = localModelDirectory / MODEL_INDEX_FILENAME;
            std::ifstream indexFile(indexPath);
            if (indexFile.is_open()) {
                return determineTaskFromModelIndex(indexFile, indexPath.string());
            }
            throw std::logic_error("failed to open model config file: " + configPath.string() + " or " + indexPath.string());
        }
    }

    std::string responseBody;
    const std::string hfEndpoint = ensureTrailingSlash(getEnvOrDefault(HF_ENDPOINT_ENV_VAR, DEFAULT_HF_ENDPOINT));
    const std::string configUrl = hfEndpoint + *sourceModel + "/resolve/main/" + MODEL_CONFIG_FILENAME;
    const auto status = fetchUrlToString(configUrl, getEnvOrDefault(HF_TOKEN_ENV_VAR), responseBody);
    if (!status.ok()) {
        throw std::logic_error("failed to download model config file from: " + configUrl);
    }
    return determineTaskFromConfigContents(responseBody, configUrl, *sourceModel);
}

}  // namespace ovms
