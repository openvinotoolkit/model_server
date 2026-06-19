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
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <limits>

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "src/audio/text_to_speech/t2s_calculator.pb.h"
#include "src/status.hpp"
#include "src/logging.hpp"
#include "src/json_parser.hpp"

#include "src/audio/text_to_speech/t2s_servable.hpp"

namespace ovms {

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
    ov::AnyMap config;
    Status status = JsonParser::parsePluginConfig(pluginConfig, config);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", pluginConfig);
        throw std::runtime_error("Error during plugin_config option parsing");
    }
    ttsPipeline = std::make_shared<ov::genai::Text2SpeechPipeline>(parsedModelsPath.string(), targetDevice, config);
    const ov::Shape speakerEmbeddingShape = ttsPipeline->get_speaker_embedding_shape();
    for (const auto& voice : graphVoices) {
        std::filesystem::path voicePath(voice.path());
        if (voicePath.is_relative()) {
            voicePath = graphDir / voicePath;
        }
        if (!std::filesystem::exists(voicePath))
            throw std::runtime_error{"Requested voice speaker embeddings file does not exist: " + voicePath.string()};
        voices[voice.name()] = readSpeakerEmbedding(voicePath, speakerEmbeddingShape);
    }
}
}  // namespace ovms
