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
#include "voxtral_audio_adapter.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

#include <openvino/genai/automatic_speech_recognition/pipeline.hpp>

#include "src/logging.hpp"
#include "stt_audio_adapter.hpp"

namespace ovms {

namespace {
constexpr size_t MaxQueuedFrames = 1500;
}

VoxtralAudioAdapter::VoxtralAudioAdapter(std::string modelPath, std::string device) :
    pipeline_(std::make_shared<ov::genai::ASRPipeline>(modelPath, device, ov::AnyMap{})) {}

std::shared_ptr<VoxtralAudioAdapter::Stream> VoxtralAudioAdapter::createStream(
    TextCallback deltaCallback, TextCallback finalCallback, ErrorCallback errorCallback) const {
    return std::make_shared<Stream>(pipeline_, std::move(deltaCallback), std::move(finalCallback),
        std::move(errorCallback));
}

VoxtralAudioAdapter::Stream::Stream(std::shared_ptr<ov::genai::ASRPipeline> pipeline,
    TextCallback deltaCallback, TextCallback finalCallback, ErrorCallback errorCallback) :
    pipeline_(std::move(pipeline)),
    deltaCallback_(std::move(deltaCallback)),
    finalCallback_(std::move(finalCallback)),
    errorCallback_(std::move(errorCallback)),
    worker_(&Stream::run, this) {}

VoxtralAudioAdapter::Stream::~Stream() {
    close();
}

void VoxtralAudioAdapter::Stream::push(AudioChunk frame) {
    if (frame.sampleRate == 0 || frame.channels == 0 || frame.samples.empty() ||
        frame.samples.size() % frame.channels != 0) {
        throw std::invalid_argument("Invalid Voxtral audio frame");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (finishing_)
        return;
    if (frames_.size() >= MaxQueuedFrames)
        throw std::runtime_error("Voxtral audio queue is full");
    frames_.push(std::move(frame));
    condition_.notify_one();
}

void VoxtralAudioAdapter::Stream::finish() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finishing_ = true;
    }
    condition_.notify_one();
}

void VoxtralAudioAdapter::Stream::close() {
    finish();
    if (worker_.joinable())
        worker_.join();
}

void VoxtralAudioAdapter::Stream::run() {
    try {
        auto stream = pipeline_->create_stream();
        while (true) {
            AudioChunk frame;
            size_t queuedFrames = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return finishing_ || !frames_.empty(); });
                if (frames_.empty())
                    break;
                frame = std::move(frames_.front());
                frames_.pop();
                queuedFrames = frames_.size();
            }
            const auto start = std::chrono::steady_clock::now();
            const std::string delta = stream.push_audio(SttAudioAdapter::prepareAudio(frame));
            SPDLOG_LOGGER_TRACE(webrtc_logger, "Voxtral push_audio: samples={}, queued_frames={}, elapsed_ms={}, delta_bytes={}",
                frame.samples.size(), queuedFrames,
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count(),
                delta.size());
            if (!delta.empty()) {
                SPDLOG_LOGGER_INFO(webrtc_logger, "Voxtral transcript delta: {}", delta);
            }
            if (!delta.empty() && deltaCallback_)
                deltaCallback_(delta);
        }
        const std::string transcript = stream.finish();
        SPDLOG_LOGGER_INFO(webrtc_logger, "Voxtral final transcript: {}", transcript);
        if (finalCallback_)
            finalCallback_(transcript);
    } catch (...) {
        if (errorCallback_)
            errorCallback_(std::current_exception());
    }
}

}  // namespace ovms