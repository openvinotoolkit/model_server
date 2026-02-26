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

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "src/audio/text_to_speech/t2s_calculator.pb.h"
#include "src/status.hpp"
#include "src/logging.hpp"
#include "src/json_parser.hpp"

#include "src/audio/text_to_speech/t2s_servable.hpp"

namespace ovms {

static ov::Tensor read_speaker_embedding(const std::filesystem::path& file_path) {
    std::ifstream input(file_path, std::ios::binary);
    if (input.fail()) {
        std::stringstream ss;
        ss << "Failed to open file: " << file_path.string();
        throw std::runtime_error(ss.str());
    }

    // Get file size
    input.seekg(0, std::ios::end);
    size_t buffer_size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    // Check size is multiple of float
    if (buffer_size % sizeof(float) != 0) {
        throw std::runtime_error("File size is not a multiple of float size.");
    }
    size_t num_floats = buffer_size / sizeof(float);
    // if (num_floats != 512) {
    //     throw std::runtime_error("File must contain speaker embedding including 512 32-bit floats.");
    // }

    ov::Tensor floats_tensor(ov::element::f32, ov::Shape{1, num_floats});
    input.read(reinterpret_cast<char*>(floats_tensor.data()), buffer_size);
    if (input.fail()) {
        throw std::runtime_error("Failed to read all data from file.");
    }

    return floats_tensor;
}

TtsServable::TtsServable(const std::string& modelDir, const std::string& targetDevice, const google::protobuf::RepeatedPtrField<mediapipe::T2sCalculatorOptions_SpeakerEmbeddings>& graphVoices, const std::string& pluginConfig, const std::string& graphPath) {
    auto fsModelsPath = std::filesystem::path(modelDir);
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
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
    for (auto voice : graphVoices) {
        if (!std::filesystem::exists(voice.path()))
            throw std::runtime_error{"Requested voice speaker embeddings file does not exist: " + voice.path()};
        voices[voice.name()] = read_speaker_embedding(voice.path());
    }
}
}  // namespace ovms
