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

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "src/audio/text_to_speech/t2s_calculator.pb.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace ovms {

class TtsServable {
public:
    std::shared_ptr<ov::genai::Text2SpeechPipeline> ttsPipeline;
    std::unordered_map<std::string, ov::Tensor> voices;
    std::mutex ttsPipelineMutex;
    std::filesystem::path parsedModelsPath;

    TtsServable(const std::string& modelDir, const std::string& targetDevice, const google::protobuf::RepeatedPtrField<mediapipe::T2sCalculatorOptions_SpeakerEmbeddings>& graphVoices, const std::string& pluginConfig, const std::string& graphPath);
};

using TtsServableMap = std::unordered_map<std::string, std::shared_ptr<TtsServable>>;
}  // namespace ovms
