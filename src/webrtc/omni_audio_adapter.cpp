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
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <vector>

#include <openvino/genai/omni/talker_speech_config.hpp>

#include "src/logging.hpp"

namespace ovms {

struct OmniAudioAdapter::Conversation {
    std::vector<ov::Tensor> audioHistory;
    std::vector<std::string> assistantHistory;
};

namespace {
// Audio encoder input rate, per the model's preprocessor_config.json.
constexpr uint32_t OmniInputSampleRate = 16000;
constexpr uint32_t OmniOutputSampleRate = 24000;
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

OmniAudioAdapter::OmniAudioAdapter(std::shared_ptr<ov::genai::OmniPipeline> pipeline, size_t audioChunkFrames) :
    pipeline_(std::move(pipeline)),
    audioChunkFrames_(audioChunkFrames) {
    if (!pipeline_)
        throw std::invalid_argument("Omni audio adapter requires a pipeline");
    if (audioChunkFrames_ == 0)
        throw std::invalid_argument("Omni audio chunk frames must be positive");
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
    TextCallback textCallback, ErrorCallback errorCallback, CompletionCallback completionCallback) {
    if (!conversation)
        throw std::invalid_argument("Omni utterance requires a conversation");
    if (utterance.sampleRate == 0 || utterance.channels == 0 || utterance.samples.empty())
        throw std::invalid_argument("Invalid Omni utterance");
    if (!audioCallback)
        throw std::invalid_argument("Omni utterance requires an audio callback");
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
        return;
    requests_.push(Request{std::move(conversation), std::move(utterance), std::move(audioCallback), std::move(textCallback),
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
            generate(*request.conversation, request.utterance, request.audioCallback, request.textCallback);
            if (request.completionCallback)
                request.completionCallback(request.conversation->assistantHistory.back());
        } catch (...) {
            if (request.errorCallback)
                request.errorCallback(std::current_exception());
        }
    }
}

void OmniAudioAdapter::generate(Conversation& conversation, const AudioChunk& utterance, const AudioCallback& audioCallback,
    const TextCallback& textCallback) {
    const AudioChunk monoInput{mono(utterance), utterance.sampleRate, utterance.timestampUs, 1};
    const auto input = resample(monoInput, OmniInputSampleRate);
    ov::Tensor audio(ov::element::f32, ov::Shape{input.size()});
    std::copy(input.begin(), input.end(), audio.data<float>());

    std::vector<ov::Tensor> audios = conversation.audioHistory;
    audios.push_back(audio);
    ov::genai::ChatHistory history;
    history.push_back({});
    history.last()["role"] = "system";
    history.last()["content"] = "Respond in English only. Use English for both your text response and generated speech. "
                                "This is a spoken conversation: keep answers to one or two short sentences.";
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
    generationConfig.max_new_tokens = 64;
    ov::genai::OmniTalkerSpeechConfig speechConfig;
    speechConfig.return_audio = true;
    speechConfig.audio_chunk_frames = audioChunkFrames_;

    std::string assistantText;
    // Temporary text/audio timing measurement, relative to generation start.
    const auto generationStart = std::chrono::steady_clock::now();
    const auto elapsedMs = [generationStart] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart).count();
    };
    uint64_t generatedAudioSamples = 0;
    size_t audioChunkCount = 0;
    int64_t firstTextMs = -1;
    int64_t firstAudioMs = -1;
    auto generationTextCallback = [&assistantText, &firstTextMs, &elapsedMs, textCallback](std::string token) {
        const auto nowMs = elapsedMs();
        if (firstTextMs < 0)
            firstTextMs = nowMs;
        assistantText += token;
        SPDLOG_LOGGER_INFO(webrtc_logger, "[omni-timing] text t_ms={} text_chars={} token=\"{}\"", nowMs, assistantText.size(), token);
        if (textCallback)
            textCallback(token);
        return ov::genai::StreamingStatus::RUNNING;
    };
    auto speechCallback = [this, audioCallback, timestampUs = utterance.timestampUs, &assistantText, &generatedAudioSamples,
                              &audioChunkCount, &firstAudioMs, &elapsedMs](const ov::Tensor& chunk) {
        const auto nowMs = elapsedMs();
        if (firstAudioMs < 0)
            firstAudioMs = nowMs;
        const float* data = chunk.data<const float>();
        const std::vector<float> samples(data, data + chunk.get_size());
        generatedAudioSamples += samples.size();
        ++audioChunkCount;
        float maxVolume = 0.0f;
        for (const float sample : samples)
            maxVolume = std::max(maxVolume, std::fabs(sample));
        SPDLOG_LOGGER_INFO(webrtc_logger,
            "[omni-timing] audio t_ms={} chunk={} chunk_audio_ms={} total_audio_ms={} text_chars={} max_volume={:.4f}",
            nowMs, audioChunkCount, samples.size() * 1000 / OmniOutputSampleRate,
            generatedAudioSamples * 1000 / OmniOutputSampleRate, assistantText.size(), maxVolume);
        AudioChunk output{resample(AudioChunk{samples, OmniOutputSampleRate, timestampUs, 1}, WebRtcSampleRate),
            WebRtcSampleRate, timestampUs, 1};
        audioCallback(std::move(output));
        return ov::genai::StreamingStatus::RUNNING;
    };

    std::vector<ov::genai::VideoMetadata> videosMetadata;
    pipeline_->generate(history, {}, {}, videosMetadata, audios,
        generationConfig, speechConfig, generationTextCallback, speechCallback);
    const auto totalMs = elapsedMs();
    const uint64_t totalAudioMs = generatedAudioSamples * 1000 / OmniOutputSampleRate;
    SPDLOG_LOGGER_INFO(webrtc_logger,
        "[omni-timing] summary generation_ms={} first_text_ms={} first_audio_ms={} audio_chunks={} total_audio_ms={} "
        "text_chars={} realtime_factor={:.2f}",
        totalMs, firstTextMs, firstAudioMs, audioChunkCount, totalAudioMs, assistantText.size(),
        totalMs > 0 ? static_cast<double>(totalAudioMs) / static_cast<double>(totalMs) : 0.0);

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