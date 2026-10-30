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
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "src/stringutils.hpp"
#include "s2t_servable.hpp"
#include "mediapipe/framework/calculator.pb.h"
#include "src/audio/speech_to_text/s2t_calculator.pb.h"

#include "src/logging.hpp"
#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_istreamwrapper.hpp"

namespace ovms {
namespace {
bool isQwen3AsrContinuousBatchingModel(const std::filesystem::path& modelsPath) {
    std::ifstream configFile(modelsPath / "config.json");
    if (!configFile) {
        return false;
    }
    rapidjson::Document document;
    rapidjson::IStreamWrapper stream(configFile);
    document.ParseStream(stream);
        const bool isQwen3Asr = document.IsObject() && document.HasMember("model_type") &&
                    document["model_type"].IsString() &&
                    std::string(document["model_type"].GetString()) == "qwen3_asr";
        return isQwen3Asr &&
            std::filesystem::exists(modelsPath / "openvino_language_model.xml") &&
            std::filesystem::exists(modelsPath / "openvino_text_embeddings_model.xml") &&
            std::filesystem::exists(modelsPath / "openvino_audio_encoder_model.xml");
}
}  // namespace

class SttNodeInitializer : public NodeInitializer {
    static constexpr const char* CALCULATOR_NAME = "S2tCalculator";

public:
    bool matches(const std::string& calculatorName) const override {
        return endsWith(calculatorName, CALCULATOR_NAME);
    }
    Status initialize(
        const ::mediapipe::CalculatorGraphConfig_Node& nodeConfig,
        const std::string& graphName,
        const std::string& basePath,
        GraphSidePackets& sidePackets,
        PythonBackend* /*pythonBackend*/) override {
        auto& sttServableMap = sidePackets.sttServableMap;
        if (!nodeConfig.node_options().size()) {
            SPDLOG_ERROR("SpeechToText node missing options in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_OPTIONS;
        }
        if (nodeConfig.name().empty()) {
            SPDLOG_ERROR("SpeechToText node name is missing in graph: {}. ", graphName);
            return StatusCode::LLM_NODE_MISSING_NAME;
        }
        std::string nodeName = nodeConfig.name();
        if (sttServableMap.find(nodeName) != sttServableMap.end()) {
            SPDLOG_ERROR("SpeechToText node name: {} already used in graph: {}. ", nodeName, graphName);
            return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
        }
        mediapipe::S2tCalculatorOptions nodeOptions;
        const auto& calculatorOptions = nodeConfig.node_options(0);
        if (!calculatorOptions.UnpackTo(&nodeOptions)) {
            SPDLOG_ERROR("Failed to unpack calculator options");
            return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
        }
        const std::filesystem::path modelsPath = std::filesystem::path(nodeOptions.models_path()).is_relative() ?
            std::filesystem::path(basePath) / nodeOptions.models_path() : std::filesystem::path(nodeOptions.models_path());
        std::shared_ptr<SttServable> servable;
        if (isQwen3AsrContinuousBatchingModel(modelsPath)) {
            SPDLOG_DEBUG("Speech-to-text graph {} uses Qwen3 ASR continuous batching backend from {}", graphName, modelsPath.string());
            servable = std::make_shared<Qwen3AsrContinuousBatchingServable>(nodeOptions, basePath);
        } else {
            SPDLOG_DEBUG("Speech-to-text graph {} uses ASR pipeline backend from {}", graphName, modelsPath.string());
            servable = std::make_shared<AsrPipelineServable>(nodeOptions, basePath);
        }
        sttServableMap.insert(std::pair<std::string, std::shared_ptr<SttServable>>(nodeName, std::move(servable)));
        return StatusCode::OK;
    }
};

static bool sttNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<SttNodeInitializer>());
    return true;
}();
}  // namespace ovms
