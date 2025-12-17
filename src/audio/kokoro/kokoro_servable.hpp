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
            if (espeak_SetVoiceByName("en") != EE_OK &&
                espeak_SetVoiceByName("en-us") != EE_OK) {
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
    std::filesystem::path parsedModelsPath;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiledModel;
    std::unique_ptr<OVInferRequestsQueue> inferRequestsQueue;
    VocabIndex vocabIndex;

    KokoroServable(const std::string& modelDir, const std::string& targetDevice, const std::string& graphPath) {
        EspeakInstance::instance();

        auto fsModelsPath = std::filesystem::path(modelDir);
        if (fsModelsPath.is_relative()) {
            parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
        } else {
            parsedModelsPath = fsModelsPath;
        }

        vocabIndex = loadVocabFromConfig(parsedModelsPath);

        ov::AnyMap properties;
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
};

using KokoroServableMap = std::unordered_map<std::string, std::shared_ptr<KokoroServable>>;
}  // namespace ovms
