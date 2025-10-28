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

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <string>

bool isWavBuffer(const std::string buf);

ov::genai::RawSpeechInput readWav(const std::string_view& wavData);
ov::genai::RawSpeechInput readMp3(const std::string_view& mp3Data);
void prepareAudioOutput(void** ppData, size_t& pDataSize, uint16_t bitsPerSample, size_t speechSize, const float* waveformPtr);
