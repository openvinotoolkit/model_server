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

#include <memory>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "src/audio/text_to_speech/t2s_calculator.pb.h"
#include "src/status.hpp"
#include "src/logging.hpp"
#include "src/json_parser.hpp"
#include "src/ov_utils.hpp"

#include "src/audio/text_to_speech/t2s_servable.hpp"

namespace ovms {

static constexpr const char* VOICES_DIR_NAME = "voices";

static size_t getShapeElementsCount(const ov::Shape& shape) {
    size_t elementsCount = 1;
    for (const auto dim : shape) {
        if (dim != 0 && elementsCount > std::numeric_limits<size_t>::max() / dim) {
            throw std::runtime_error("Speaker embedding shape is too large.");
        }
        elementsCount *= dim;
    }
    return elementsCount;
}

static std::vector<std::filesystem::path> getVoiceEmbeddingPaths(const std::filesystem::path& voicesDir) {
    std::vector<std::filesystem::path> voicePaths;
    std::error_code errorCode;
    std::filesystem::directory_iterator directoryIt(voicesDir, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to open voices directory '" + voicesDir.string() + "': " + errorCode.message());
    }

    const std::filesystem::directory_iterator directoryEnd;
    for (; directoryIt != directoryEnd; directoryIt.increment(errorCode)) {
        if (errorCode) {
            throw std::runtime_error("Failed to iterate voices directory '" + voicesDir.string() + "': " + errorCode.message());
        }
        const auto& entry = *directoryIt;
        const bool isRegularFile = entry.is_regular_file(errorCode);
        if (errorCode) {
            throw std::runtime_error("Failed to inspect entry '" + entry.path().string() + "' in voices directory '" + voicesDir.string() + "': " + errorCode.message());
        }
        if (!isRegularFile) {
            continue;
        }
        if (entry.path().extension() == ".bin") {
            voicePaths.emplace_back(entry.path());
        }
    }
    std::sort(voicePaths.begin(), voicePaths.end());
    return voicePaths;
}

static ov::Tensor readSpeakerEmbedding(const std::filesystem::path& filePath, const ov::Shape& expectedShape) {
    std::ifstream input(filePath, std::ios::binary);
    if (input.fail()) {
        std::stringstream ss;
        ss << "Failed to open file: " << filePath.string();
        throw std::runtime_error(ss.str());
    }

    // Get file size
    input.seekg(0, std::ios::end);
    if (input.fail()) {
        throw std::runtime_error("Failed to seek to the end of file.");
    }
    const std::streampos endPosition = input.tellg();
    if (endPosition == std::streampos(-1)) {
        throw std::runtime_error("Failed to determine file size.");
    }
    const size_t bufferSize = static_cast<size_t>(endPosition);
    input.seekg(0, std::ios::beg);
    if (input.fail()) {
        throw std::runtime_error("Failed to seek to the beginning of file.");
    }

    // Check size is multiple of float
    if (bufferSize % sizeof(float) != 0) {
        throw std::runtime_error("File size is not a multiple of float size.");
    }
    const size_t numFloats = bufferSize / sizeof(float);
    const size_t expectedElements = getShapeElementsCount(expectedShape);
    if (numFloats != expectedElements) {
        std::stringstream ss;
        ss << "File must contain speaker embedding with " << expectedElements
           << " 32-bit floats. Got: " << numFloats;
        throw std::runtime_error(ss.str());
    }

    ov::Tensor floatsTensor(ov::element::f32, expectedShape);
    input.read(reinterpret_cast<char*>(floatsTensor.data()), bufferSize);
    if (input.fail()) {
        throw std::runtime_error("Failed to read all data from file.");
    }

    return floatsTensor;
}

TtsServable::TtsServable(const std::string& modelDir, const std::string& targetDevice, const google::protobuf::RepeatedPtrField<mediapipe::T2sCalculatorOptions_SpeakerEmbeddings>& graphVoices, const std::string& pluginConfig, const std::string& graphPath) {
    const std::filesystem::path graphDir = std::filesystem::path(graphPath).parent_path();
    auto fsModelsPath = std::filesystem::path(modelDir);
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = graphDir / fsModelsPath;
    } else {
        parsedModelsPath = fsModelsPath;
    }
    std::string device = targetDevice;
    if (device.empty()) {
        device = recommendTargetDevice();
        SPDLOG_INFO("No device specified for TTS model, using recommended device: {}", device);
    }
    // Normalize the path to use OS-appropriate separators
    parsedModelsPath = std::filesystem::absolute(parsedModelsPath);
    ov::AnyMap config;
    Status status = JsonParser::parsePluginConfig(pluginConfig, config);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", pluginConfig);
        throw std::runtime_error("Error during plugin_config option parsing");
    }
    applyGlobalCacheDirFallback(config);
    ttsPipeline = std::make_shared<ov::genai::Text2SpeechPipeline>(parsedModelsPath.string(), device, config);
    const ov::Shape speakerEmbeddingShape = ttsPipeline->get_speaker_embedding_shape();
    const std::filesystem::path voicesDir = parsedModelsPath / VOICES_DIR_NAME;
    std::error_code ec;
    if (std::filesystem::is_directory(voicesDir, ec)) {
        try {
            for (const auto& voicePath : getVoiceEmbeddingPaths(voicesDir)) {
                const std::string voiceName = voicePath.stem().string();
                voices.insert_or_assign(voiceName, readSpeakerEmbedding(voicePath, speakerEmbeddingShape));
            }
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to load voices from {}: {}", voicesDir.string(), e.what());
        }
    } else {
        SPDLOG_DEBUG("Voices directory not found: {}", voicesDir.string());
    }

    for (const auto& voice : graphVoices) {
        std::filesystem::path voicePath(voice.path());
        if (voicePath.is_relative()) {
            voicePath = graphDir / voicePath;
        }
        if (!std::filesystem::exists(voicePath))
            throw std::runtime_error{"Requested voice speaker embeddings file does not exist: " + voicePath.string()};
        if (voices.find(voice.name()) != voices.end()) {
            SPDLOG_DEBUG("Voice '{}' is already configured and will be overwritten with tensor from: {}", voice.name(), voicePath.string());
        }
        voices[voice.name()] = readSpeakerEmbedding(voicePath, speakerEmbeddingShape);
    }
}
}  // namespace ovms
