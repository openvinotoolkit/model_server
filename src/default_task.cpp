//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pull_module/curl_downloader.hpp"
#include "pull_module/hf_env_vars.hpp"
#include "servable_type_detector.hpp"
#include "status.hpp"

namespace ovms {

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

bool graphPbtxtExists(const std::string& modelPath) {
    const auto graphPath = std::filesystem::path(modelPath) / "graph.pbtxt";
    return std::filesystem::exists(graphPath);
}

bool hasTaskSpecificParameters(const std::vector<std::string>& unmatchedOptions) {
    return !unmatchedOptions.empty();
}

std::string determineDefaultTaskParameter(const std::optional<std::string>& modelPath, const std::optional<std::string>& sourceModel, const std::optional<std::string>& modelRepositoryPath) {
    DefaultTaskDetector detector;

    if (modelPath.has_value() && !modelPath->empty()) {
        const std::filesystem::path pathFs(*modelPath);
        ModelCatalogContext ctx(pathFs, *modelPath);
        const std::string task = detector.detect(ctx);
        if (task.empty()) {
            throw std::logic_error("cannot determine default task for model path: " + *modelPath);
        }
        return task;
    }

    if (!sourceModel.has_value() || sourceModel->empty()) {
        throw std::logic_error("cannot determine default --task without model_path or source_model");
    }

    // Try local model repository path before downloading from HuggingFace
    if (modelRepositoryPath.has_value() && !modelRepositoryPath->empty()) {
        const auto localModelDir = std::filesystem::path(*modelRepositoryPath) / *sourceModel;
        if (std::filesystem::exists(localModelDir)) {
            ModelCatalogContext ctx(localModelDir, *sourceModel);
            const std::string task = detector.detect(ctx);
            if (task.empty()) {
                throw std::logic_error("cannot determine default task for source model: " + *sourceModel);
            }
            return task;
        }
    }

    // Download config.json from HuggingFace
    std::string responseBody;
    const std::string hfEndpoint = ensureTrailingSlash(getEnvOrDefault(HF_ENDPOINT_ENV_VAR, DEFAULT_HF_ENDPOINT));
    const std::string configUrl = hfEndpoint + *sourceModel + "/resolve/main/config.json";
    const auto status = fetchUrlToString(configUrl, getEnvOrDefault(HF_TOKEN_ENV_VAR), responseBody);
    if (!status.ok()) {
        throw std::logic_error("failed to download model config file from: " + configUrl);
    }

    ModelCatalogContext ctx(std::filesystem::path{}, *sourceModel);
    ctx.addContent("config.json", std::move(responseBody));
    const std::string task = detector.detect(ctx);
    if (task.empty()) {
        throw std::logic_error("cannot determine default task for source model: " + *sourceModel);
    }
    return task;
}

}  // namespace ovms
