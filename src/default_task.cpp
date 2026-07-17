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

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pull_module/curl_downloader.hpp"
#include "pull_module/hf_env_vars.hpp"
#include "default_task_detector.hpp"
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

std::optional<std::string> determineDefaultTaskParameter(const std::optional<std::string>& modelPath, const std::optional<std::string>& sourceModel, const std::optional<std::string>& modelRepositoryPath) {
    DefaultTaskDetector detector;

    if (modelPath.has_value() && !modelPath->empty()) {
        // Normalize first to remove any trailing separator so that filename()
        // always returns the leaf directory name, e.g. "/models/llama/" → "llama".
        const std::filesystem::path pathFs = std::filesystem::path(*modelPath).lexically_normal();
        // Use only the leaf directory name for keyword-based disambiguation
        // (e.g. "Qwen3-Embedding-0.6B"), not the full path which may contain
        // unrelated keywords from parent directories.
        const std::string identifier = pathFs.filename().empty() ? pathFs.string() : pathFs.filename().string();
        ModelCatalogContext ctx(pathFs, identifier);
        const std::string task = detector.detect(ctx);
        if (task.empty()) {
            return std::nullopt;
        }
        return task;
    }

    if (!sourceModel.has_value() || sourceModel->empty()) {
        return std::nullopt;
    }

    // Try local model repository path before downloading from HuggingFace
    if (modelRepositoryPath.has_value() && !modelRepositoryPath->empty()) {
        const auto localModelDir = std::filesystem::path(*modelRepositoryPath) / *sourceModel;
        if (std::filesystem::exists(localModelDir)) {
            ModelCatalogContext ctx(localModelDir, *sourceModel);
            const std::string task = detector.detect(ctx);
            if (task.empty()) {
                return std::nullopt;
            }
            return task;
        }
    }

    // Download config files from HuggingFace.
    // config.json is tried first (covers transformer-style models).
    // model_index.json is only attempted when config.json is absent — it is
    // specific to Diffusers pipeline repos (e.g. StableDiffusion, Flux) that
    // do not have a standard config.json architectures array.
    const std::string hfEndpoint = ensureTrailingSlash(getEnvOrDefault(HF_ENDPOINT_ENV_VAR, DEFAULT_HF_ENDPOINT));
    const std::string token = getEnvOrDefault(HF_TOKEN_ENV_VAR);

    ModelCatalogContext ctx(std::filesystem::path{}, *sourceModel);

    std::string configBody;
    const std::string configUrl = hfEndpoint + *sourceModel + "/resolve/main/config.json";
    const auto configStatus = fetchUrlToString(configUrl, token, configBody);
    if (configStatus.ok()) {
        ctx.addContent("config.json", std::move(configBody));
    } else {
        // config.json not available — try model_index.json (Diffusers repos)
        std::string indexBody;
        const std::string indexUrl = hfEndpoint + *sourceModel + "/resolve/main/model_index.json";
        const auto indexStatus = fetchUrlToString(indexUrl, token, indexBody);
        if (indexStatus.ok()) {
            ctx.addContent("model_index.json", std::move(indexBody));
        } else {
            return std::nullopt;
        }
    }
    const std::string task = detector.detect(ctx);
    if (task.empty()) {
        return std::nullopt;
    }
    return task;
}

}  // namespace ovms
