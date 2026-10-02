//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************/
#include "stt_audio_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/logging.hpp"

namespace ovms {

namespace {
constexpr uint32_t SttSampleRate = 16000;
}

SttAudioAdapter::SttAudioAdapter(std::string modelPath, std::string device) :
    pipeline_(std::make_shared<ov::genai::ASRPipeline>(modelPath, device, ov::AnyMap{})),
    worker_(&SttAudioAdapter::run, this) {
}

SttAudioAdapter::~SttAudioAdapter() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_one();
    if (worker_.joinable())
        worker_.join();
}

void SttAudioAdapter::submit(AudioChunk utterance, TextCallback textCallback, ErrorCallback errorCallback,
    CompletionCallback completionCallback) {
    if (utterance.sampleRate == 0 || utterance.channels == 0 || utterance.samples.empty() ||
        utterance.samples.size() % utterance.channels != 0) {
        throw std::invalid_argument("Invalid speech-to-text utterance");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
        return;
    requests_.push(Request{std::move(utterance), std::move(textCallback), std::move(errorCallback),
        std::move(completionCallback)});
    condition_.notify_one();
}

void SttAudioAdapter::run() {
    while (true) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !requests_.empty(); });
            if (stopping_)
                return;
            request = std::move(requests_.front());
            requests_.pop();
        }
        try {
            transcribe(request);
        } catch (...) {
            if (request.errorCallback)
                request.errorCallback(std::current_exception());
        }
    }
}

void SttAudioAdapter::transcribe(Request& request) {
    const auto audio = prepareAudio(request.utterance);
    SPDLOG_LOGGER_INFO(webrtc_logger, "Starting Whisper transcription: samples={}, length_ms={}",
        audio.size(), audio.size() * 1000 / SttSampleRate);
    auto config = pipeline_->get_generation_config();
    config.language = "<|en|>";
    auto textCallback = [callback = request.textCallback](std::string text) {
        if (callback)
            callback(text);
        return ov::genai::StreamingStatus::RUNNING;
    };
    const ov::genai::ASRDecodedResults result = pipeline_->generate(audio, config, textCallback);
    const std::string transcript = static_cast<std::string>(result);
    SPDLOG_LOGGER_INFO(webrtc_logger, "Whisper transcription finished: \"{}\"", transcript);
    if (request.completionCallback)
        request.completionCallback(transcript);
}

std::vector<float> SttAudioAdapter::prepareAudio(const AudioChunk& utterance) {
    const size_t frameCount = utterance.samples.size() / utterance.channels;
    std::vector<float> monoAudio(frameCount);
    for (size_t frame = 0; frame < frameCount; ++frame) {
        float channelSum = 0.0f;
        for (uint32_t channel = 0; channel < utterance.channels; ++channel)
            channelSum += utterance.samples[frame * utterance.channels + channel];
        monoAudio[frame] = channelSum / static_cast<float>(utterance.channels);
    }
    if (utterance.sampleRate == SttSampleRate)
        return monoAudio;

    const size_t outputSize = static_cast<size_t>(monoAudio.size() *
        static_cast<double>(SttSampleRate) / utterance.sampleRate);
    if (outputSize == 0)
        throw std::invalid_argument("Speech-to-text utterance is shorter than one output sample");
    std::vector<float> output(outputSize);
    const double ratio = static_cast<double>(utterance.sampleRate) / SttSampleRate;
    for (size_t index = 0; index < outputSize; ++index) {
        const double sourceIndex = index * ratio;
        const size_t left = std::min(static_cast<size_t>(sourceIndex), monoAudio.size() - 1);
        const size_t right = std::min(left + 1, monoAudio.size() - 1);
        const float fraction = static_cast<float>(sourceIndex - left);
        output[index] = monoAudio[left] + (monoAudio[right] - monoAudio[left]) * fraction;
    }
    return output;
}

}  // namespace ovms