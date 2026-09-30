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
//*****************************************************************************/
#include "audio_utterance_buffer.hpp"

#include <cmath>
#include <stdexcept>

namespace ovms {

AudioUtteranceBuffer::AudioUtteranceBuffer() :
    AudioUtteranceBuffer(Config{}) {}

AudioUtteranceBuffer::AudioUtteranceBuffer(Config config) :
    config_(config) {
    if (config_.rmsThreshold < 0.0f || config_.minimumSpeechFrames == 0 || config_.silenceHangoverFrames == 0) {
        throw std::invalid_argument("Invalid audio utterance buffer configuration");
    }
}

std::optional<AudioChunk> AudioUtteranceBuffer::push(const AudioChunk& frame) {
    if (frame.samples.empty() || frame.sampleRate == 0 || frame.channels == 0) {
        throw std::invalid_argument("Audio utterance frame is invalid");
    }
    if ((frame.samples.size() % frame.channels) != 0) {
        throw std::invalid_argument("Audio utterance frame has incomplete channels");
    }
    if (sampleRate_ != 0 && (sampleRate_ != frame.sampleRate || channels_ != frame.channels)) {
        throw std::invalid_argument("Audio utterance frame format changed");
    }

    const bool speech = isSpeech(frame);
    if (speechFrames_ == 0 && !speech)
        return std::nullopt;
    if (sampleRate_ == 0) {
        sampleRate_ = frame.sampleRate;
        channels_ = frame.channels;
        startTimestampUs_ = frame.timestampUs;
    }
    if (speech) {
        ++speechFrames_;
        silentFrames_ = 0;
    } else if (speechFrames_ > 0) {
        ++silentFrames_;
    }
    samples_.insert(samples_.end(), frame.samples.begin(), frame.samples.end());

    if (speechFrames_ >= config_.minimumSpeechFrames && silentFrames_ >= config_.silenceHangoverFrames) {
        AudioChunk utterance{std::move(samples_), sampleRate_, startTimestampUs_, channels_};
        reset();
        return utterance;
    }
    return std::nullopt;
}

void AudioUtteranceBuffer::reset() {
    samples_.clear();
    sampleRate_ = 0;
    channels_ = 0;
    startTimestampUs_ = 0;
    speechFrames_ = 0;
    silentFrames_ = 0;
}

bool AudioUtteranceBuffer::isSpeech(const AudioChunk& frame) const {
    float squaredSampleSum = 0.0f;
    for (const float sample : frame.samples) {
        squaredSampleSum += sample * sample;
    }
    const float rms = std::sqrt(squaredSampleSum / frame.samples.size());
    return rms >= config_.rmsThreshold;
}

}  // namespace ovms
