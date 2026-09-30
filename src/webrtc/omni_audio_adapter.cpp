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
#include "omni_audio_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <vector>

#include <openvino/genai/omni/talker_speech_config.hpp>

namespace ovms {

struct OmniAudioAdapter::Conversation {
    std::vector<ov::Tensor> audioHistory;
    std::vector<std::string> assistantHistory;
};

namespace {
constexpr uint32_t OmniSampleRate = 24000;
constexpr uint32_t WebRtcSampleRate = 48000;
constexpr size_t MaxRememberedTurns = 4;

std::vector<float> mono(const AudioChunk& input) {
    if (input.channels == 1)
        return input.samples;
    if (input.channels == 0 || input.samples.size() % input.channels != 0)
        throw std::invalid_argument("Invalid Omni input channel layout");

    std::vector<float> output(input.samples.size() / input.channels);
    for (size_t frame = 0; frame < output.size(); ++frame) {
        float sum = 0.0f;
        for (uint32_t channel = 0; channel < input.channels; ++channel)
            sum += input.samples[frame * input.channels + channel];
        output[frame] = sum / static_cast<float>(input.channels);
    }
    return output;
}
}  // namespace

OmniAudioAdapter::OmniAudioAdapter(std::shared_ptr<ov::genai::OmniPipeline> pipeline) :
    pipeline_(std::move(pipeline)) {
    if (!pipeline_)
        throw std::invalid_argument("Omni audio adapter requires a pipeline");
    worker_ = std::thread(&OmniAudioAdapter::run, this);
}

OmniAudioAdapter::~OmniAudioAdapter() {
    cancel();
    if (worker_.joinable())
        worker_.join();
}

OmniAudioAdapter::ConversationPtr OmniAudioAdapter::createConversation() const {
    return std::make_shared<Conversation>();
}

void OmniAudioAdapter::submit(ConversationPtr conversation, AudioChunk utterance, AudioCallback audioCallback,
    ErrorCallback errorCallback, CompletionCallback completionCallback) {
    if (!conversation)
        throw std::invalid_argument("Omni utterance requires a conversation");
    if (utterance.sampleRate == 0 || utterance.channels == 0 || utterance.samples.empty())
        throw std::invalid_argument("Invalid Omni utterance");
    if (!audioCallback)
        throw std::invalid_argument("Omni utterance requires an audio callback");
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
        return;
    requests_.push(Request{std::move(conversation), std::move(utterance), std::move(audioCallback),
        std::move(errorCallback), std::move(completionCallback)});
    condition_.notify_one();
}

void OmniAudioAdapter::cancel() {
    stopping_ = true;
    condition_.notify_one();
}

void OmniAudioAdapter::run() {
    while (!stopping_) {
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
            generate(*request.conversation, request.utterance, request.audioCallback);
            if (request.completionCallback)
                request.completionCallback();
        } catch (...) {
            if (request.errorCallback)
                request.errorCallback(std::current_exception());
        }
    }
}

void OmniAudioAdapter::generate(Conversation& conversation, const AudioChunk& utterance, const AudioCallback& audioCallback) {
    const AudioChunk monoInput{mono(utterance), utterance.sampleRate, utterance.timestampUs, 1};
    const auto input = resample(monoInput, OmniSampleRate);
    ov::Tensor audio(ov::element::f32, ov::Shape{input.size()});
    std::copy(input.begin(), input.end(), audio.data<float>());

    std::vector<ov::Tensor> audios = conversation.audioHistory;
    audios.push_back(audio);
    ov::genai::ChatHistory history;
    for (const auto& assistantText : conversation.assistantHistory) {
        history.push_back({});
        history.last()["role"] = "user";
        history.last()["content"] = "Audio input";
        history.push_back({});
        history.last()["role"] = "assistant";
        history.last()["content"] = assistantText;
    }
    history.push_back({});
    history.last()["role"] = "user";
    history.last()["content"] = "Audio input";
    ov::genai::GenerationConfig generationConfig;
    generationConfig.max_new_tokens = 128;
    ov::genai::OmniTalkerSpeechConfig speechConfig;
    speechConfig.return_audio = true;
    speechConfig.audio_chunk_frames = 4;

    std::string assistantText;
    auto textCallback = [&assistantText](std::string token) {
        assistantText += token;
        return ov::genai::StreamingStatus::RUNNING;
    };
    auto speechCallback = [this, audioCallback, timestampUs = utterance.timestampUs](const ov::Tensor& chunk) {
        const float* data = chunk.data<const float>();
        const std::vector<float> samples(data, data + chunk.get_size());
        AudioChunk output{resample(AudioChunk{samples, OmniSampleRate, timestampUs, 1}, WebRtcSampleRate),
            WebRtcSampleRate, timestampUs, 1};
        audioCallback(std::move(output));
        return ov::genai::StreamingStatus::RUNNING;
    };

    std::vector<ov::genai::VideoMetadata> videosMetadata;
    pipeline_->generate(history, {}, {}, videosMetadata, audios,
        generationConfig, speechConfig, textCallback, speechCallback);

    conversation.audioHistory.push_back(std::move(audio));
    conversation.assistantHistory.push_back(std::move(assistantText));
    while (conversation.audioHistory.size() > MaxRememberedTurns) {
        conversation.audioHistory.erase(conversation.audioHistory.begin());
        conversation.assistantHistory.erase(conversation.assistantHistory.begin());
    }
}

std::vector<float> OmniAudioAdapter::resample(const AudioChunk& input, uint32_t outputRate) {
    if (input.sampleRate == outputRate)
        return input.samples;
    if (input.channels != 1 || input.sampleRate == 0 || outputRate == 0)
        throw std::invalid_argument("Omni resampler requires mono audio and valid rates");

    const size_t outputSize = static_cast<size_t>(std::llround(
        input.samples.size() * static_cast<double>(outputRate) / input.sampleRate));
    std::vector<float> output(outputSize);
    for (size_t index = 0; index < outputSize; ++index) {
        const double source = index * static_cast<double>(input.sampleRate) / outputRate;
        const size_t left = std::min(static_cast<size_t>(source), input.samples.size() - 1);
        const size_t right = std::min(left + 1, input.samples.size() - 1);
        const float fraction = static_cast<float>(source - left);
        output[index] = input.samples[left] + (input.samples[right] - input.samples[left]) * fraction;
    }
    return output;
}

}  // namespace ovms