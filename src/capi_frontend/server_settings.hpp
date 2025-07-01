#pragma once
//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "../graph_export/graph_export_types.hpp"
#include "../config_export_module/config_export_types.hpp"

namespace ovms {

enum OvmsServerMode : int {
    SERVING_MODELS_MODE,
    HF_PULL_MODE,
    HF_PULL_AND_START_MODE,
    LIST_MODELS_MODE,
    MODIFY_CONFIG_MODE,
    UNKNOWN_MODE
};

struct PluginConfigSettingsImpl {
    std::optional<std::string> kvCachePrecision;
    std::optional<uint32_t> maxPromptLength;
    std::optional<std::string> modelDistributionPolicy;
};

struct TextGenGraphSettingsImpl {
    std::string modelPath = "./";
    std::string modelName = "";
    uint32_t maxNumSeqs = 256;
    std::string targetDevice = "CPU";
    std::string enablePrefixCaching = "true";
    uint32_t cacheSize = 10;
    std::string dynamicSplitFuse = "true";
    PluginConfigSettingsImpl pluginConfig;
    std::optional<uint32_t> maxNumBatchedTokens;
    std::optional<std::string> draftModelDirName;
    std::optional<std::string> pipelineType;
};

struct EmbeddingsGraphSettingsImpl {
    std::string modelPath = "./";
    std::string targetDevice = "CPU";
    std::string modelName = "";
    uint32_t numStreams = 1;
    std::string normalize = "true";
    std::string meanPooling = "false";
};

struct RerankGraphSettingsImpl {
    std::string modelPath = "./";
    std::string targetDevice = "CPU";
    std::string modelName = "";
    uint32_t numStreams = 1;
    uint64_t maxAllowedChunks = 10000;
};

struct ImageGenerationGraphSettingsImpl {
    std::string modelName = "";
    std::string modelPath = "./";
    std::string targetDevice = "CPU";
    std::string maxResolution = "";
    std::string defaultResolution = "";
    std::optional<uint32_t> maxNumberImagesPerPrompt;
    std::optional<uint32_t> defaultNumInferenceSteps;
    std::optional<uint32_t> maxNumInferenceSteps;
    std::string pluginConfig;
};

struct HFSettingsImpl {
    std::string targetDevice = "CPU";
    std::string sourceModel = "";
    std::string downloadPath = "";
    bool overwriteModels = false;
    GraphExportType task = TEXT_GENERATION_GRAPH;
    std::variant<TextGenGraphSettingsImpl, RerankGraphSettingsImpl, EmbeddingsGraphSettingsImpl, ImageGenerationGraphSettingsImpl> graphSettings;
};

struct ServerSettingsImpl {
    uint32_t grpcPort = 0;
    uint32_t restPort = 0;
    uint32_t grpcWorkers = 1;
    std::string grpcBindAddress = "0.0.0.0";
    std::optional<uint32_t> restWorkers;
    std::optional<uint32_t> grpcMaxThreads;
    std::string restBindAddress = "0.0.0.0";
    bool metricsEnabled = false;
    std::string metricsList;
    std::string cpuExtensionLibraryPath;
    std::optional<std::string> allowedLocalMediaPath;
    std::string logLevel = "INFO";
    std::string logPath;
    bool allowCredentials = false;
    std::string allowedOrigins{"*"};
    std::string allowedMethods{"*"};
    std::string allowedHeaders{"*"};
#ifdef MTR_ENABLED
    std::string tracePath;
#endif
    std::optional<size_t> grpcMemoryQuota;
    std::string grpcChannelArguments;
    uint32_t filesystemPollWaitMilliseconds = 1000;
    uint32_t sequenceCleanerPollWaitMinutes = 5;
    uint32_t resourcesCleanerPollWaitSeconds = 1;
    std::string cacheDir;
    bool withPython = false;
    bool startedWithCLI = false;
    ConfigExportType exportConfigType = UNKNOWN_MODEL;
    HFSettingsImpl hfSettings;
    OvmsServerMode serverMode = SERVING_MODELS_MODE;
};

struct ModelsSettingsImpl {
    std::string modelName;
    std::string modelPath;
    std::string batchSize;
    std::string shape;
    std::string layout;
    std::string modelVersionPolicy;
    uint32_t nireq = 0;
    std::string targetDevice;
    std::string pluginConfig;
    std::optional<bool> stateful;
    std::optional<bool> lowLatencyTransformation;
    std::optional<uint32_t> maxSequenceNumber;
    std::optional<bool> idleSequenceCleanup;
    std::vector<std::string> userSetSingleModelArguments;

    std::string configPath;
};

}  // namespace ovms
