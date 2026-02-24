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
#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "openvino/runtime/core.hpp"
#include "../../ovinferrequestsqueue.hpp"

#include <espeak-ng/speak_lib.h>
#include <rapidjson/document.h>

#include "src/audio/kokoro/kokoro_calculator.pb.h"
#include "src/logging.hpp"

namespace ovms {

struct VocabIndex {
    std::unordered_map<std::string, int> by_token;
    size_t max_token_bytes = 1;
};

struct VoicePack {
    std::vector<float> data;   // flat [numEntries * STYLE_DIM]
    size_t numEntries = 0;
};

class EspeakInstance {
public:
    static EspeakInstance& instance() {
        static EspeakInstance inst;
        return inst;
    }

    bool isReady() const { return ready_; }
    std::mutex& mutex() { return mutex_; }

private:
    EspeakInstance() {
        ready_ = tryInit();
        if (!ready_) {
            SPDLOG_ERROR("eSpeak-NG initialization failed (data path or voice not found)");
        } else {
            SPDLOG_INFO("eSpeak-NG initialized successfully");
        }
    }

    ~EspeakInstance() {
        if (ready_) {
            espeak_Terminate();
        }
    }

    EspeakInstance(const EspeakInstance&) = delete;
    EspeakInstance& operator=(const EspeakInstance&) = delete;

    bool tryInit() {
        auto try_path = [](const char* path) -> bool {
            int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,
                                       0, path,
                                       espeakINITIALIZE_DONT_EXIT);
            if (sr <= 0) return false;
            if (espeak_SetVoiceByName("en-us") != EE_OK &&
                espeak_SetVoiceByName("en") != EE_OK) {
                return false;
            }
            return true;
        };

        if (try_path(nullptr)) return true;

        static const char* ngPaths[] = {
            "/usr/share/espeak-ng-data",
            "/opt/homebrew/share/espeak-ng-data",
            "/usr/local/share/espeak-ng-data",
            "espeak-ng-data",
            nullptr
        };
        for (int i = 0; ngPaths[i]; ++i)
            if (try_path(ngPaths[i])) return true;

        static const char* esPaths[] = {
            "/usr/share/espeak-data",
            "/usr/local/share/espeak-data",
            "espeak-data",
            nullptr
        };
        for (int i = 0; esPaths[i]; ++i)
            if (try_path(esPaths[i])) return true;

        return false;
    }

    bool ready_ = false;
    std::mutex mutex_;
};

struct KokoroServable {
    static constexpr size_t STYLE_DIM = 256;

    std::filesystem::path parsedModelsPath;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiledModel;
    std::unique_ptr<OVInferRequestsQueue> inferRequestsQueue;
    VocabIndex vocabIndex;
    std::unordered_map<std::string, VoicePack> voicePacks;
    std::string defaultVoiceName;

    KokoroServable(const std::string& modelDir, const std::string& targetDevice, const std::string& graphPath) {
        EspeakInstance::instance();

        auto fsModelsPath = std::filesystem::path(modelDir);
        if (fsModelsPath.is_relative()) {
            parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
        } else {
            parsedModelsPath = fsModelsPath;
        }

        vocabIndex = loadVocabFromConfig(parsedModelsPath);
        loadVoicePacks(parsedModelsPath);

        ov::AnyMap properties = {
            // Use ACCURACY execution mode to avoid fast-math approximation errors
            // that accumulate in the deep decoder network and cause energy fade.
            ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY),
        };
        ov::Core core;
        auto m_model = core.read_model(parsedModelsPath / std::filesystem::path("openvino_model.xml"), {}, properties);
        compiledModel = core.compile_model(m_model, targetDevice, properties);
        inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(compiledModel, 5);
    }

    OVInferRequestsQueue& getInferRequestsQueue() {
        return *inferRequestsQueue;
    }

    const VocabIndex& getVocabIndex() const {
        return vocabIndex;
    }

    // Returns pointer to 256 floats for the given voice and token count.
    // voiceName: requested voice (e.g. "af_alloy"). Falls back to default voice if not found.
    // numContentTokens: number of token IDs excluding BOS/EOS padding.
    const float* getVoiceSlice(const std::string& voiceName, size_t numContentTokens) const {
        auto it = voicePacks.find(voiceName);
        if (it == voicePacks.end()) {
            it = voicePacks.find(defaultVoiceName);
            if (it == voicePacks.end()) {
                return nullptr;
            }
        }
        const auto& pack = it->second;
        size_t idx = std::min(numContentTokens, pack.numEntries - 1);
        return pack.data.data() + (idx * STYLE_DIM);
    }

    bool hasVoice(const std::string& voiceName) const {
        return voicePacks.count(voiceName) > 0;
    }

    const std::string& getDefaultVoiceName() const {
        return defaultVoiceName;
    }

private:
    static VocabIndex loadVocabFromConfig(const std::filesystem::path& modelDir) {
        VocabIndex ix;
        auto configPath = modelDir / "config.json";
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
            SPDLOG_ERROR("Failed to open Kokoro config: {}", configPath.string());
            return ix;
        }

        std::stringstream buffer;
        buffer << ifs.rdbuf();
        std::string jsonStr = buffer.str();

        rapidjson::Document doc;
        doc.Parse(jsonStr.c_str());
        if (doc.HasParseError()) {
            SPDLOG_ERROR("Failed to parse Kokoro config JSON: {}", configPath.string());
            return ix;
        }

        if (!doc.HasMember("vocab") || !doc["vocab"].IsObject()) {
            SPDLOG_ERROR("Kokoro config missing 'vocab' object: {}", configPath.string());
            return ix;
        }

        const auto& vocab = doc["vocab"];
        ix.by_token.reserve(vocab.MemberCount());
        for (auto it = vocab.MemberBegin(); it != vocab.MemberEnd(); ++it) {
            if (!it->name.IsString() || !it->value.IsInt()) continue;
            std::string token = it->name.GetString();
            int id = it->value.GetInt();
            ix.by_token.emplace(token, id);
            ix.max_token_bytes = std::max(ix.max_token_bytes, token.size());
        }

        SPDLOG_INFO("Loaded Kokoro vocabulary: {} tokens, max_token_bytes={}",
                     ix.by_token.size(), ix.max_token_bytes);
        return ix;
    }

    void loadVoicePacks(const std::filesystem::path& modelDir) {
        auto voicesDir = modelDir / "voices";
        if (!std::filesystem::exists(voicesDir) || !std::filesystem::is_directory(voicesDir)) {
            SPDLOG_WARN("No voices directory found at: {}", voicesDir.string());
            return;
        }

        for (const auto& entry : std::filesystem::directory_iterator(voicesDir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".bin")
                continue;

            std::string name = entry.path().stem().string();
            auto fileSize = std::filesystem::file_size(entry.path());
            if (fileSize == 0 || fileSize % (STYLE_DIM * sizeof(float)) != 0) {
                SPDLOG_ERROR("Voice file {} has invalid size {} (must be multiple of {})",
                             entry.path().string(), fileSize, STYLE_DIM * sizeof(float));
                continue;
            }

            VoicePack pack;
            pack.numEntries = fileSize / (STYLE_DIM * sizeof(float));
            pack.data.resize(pack.numEntries * STYLE_DIM);

            std::ifstream ifs(entry.path(), std::ios::binary);
            if (!ifs.read(reinterpret_cast<char*>(pack.data.data()), fileSize)) {
                SPDLOG_ERROR("Failed to read voice file: {}", entry.path().string());
                continue;
            }

            SPDLOG_INFO("Loaded voice pack '{}': {} entries x {} dims from {}",
                         name, pack.numEntries, STYLE_DIM, entry.path().string());

            if (defaultVoiceName.empty()) {
                defaultVoiceName = name;
            }
            voicePacks.emplace(name, std::move(pack));
        }

        SPDLOG_INFO("Loaded {} voice pack(s), default: '{}'", voicePacks.size(), defaultVoiceName);
    }
};

using KokoroServableMap = std::unordered_map<std::string, std::shared_ptr<KokoroServable>>;
}  // namespace ovms
