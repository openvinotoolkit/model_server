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

template <typename GenAiStream>
class GenAiInferenceStream : public VoxtralAudioAdapter::InferenceStream {
public:
    explicit GenAiInferenceStream(GenAiStream stream) :
        stream_(std::move(stream)) {}

    void pushAudio(const std::vector<float>& audio) override {
        stream_.push_audio(audio);
    }

    void endAudio() override {
        stream_.end_audio();
    }

    std::string generate(VoxtralAudioAdapter::TextCallback deltaCallback) override {
        const auto result = stream_.generate([deltaCallback = std::move(deltaCallback)](std::string delta) {
            if (deltaCallback)
                deltaCallback(delta);
            return ov::genai::StreamingStatus::RUNNING;
        });
        return static_cast<std::string>(result);
    }

private:
    GenAiStream stream_;
};

VoxtralAudioAdapter::StreamFactory makeGenAiStreamFactory(
    std::shared_ptr<ov::genai::ASRPipeline> pipeline) {
    if (!pipeline)
        throw std::invalid_argument("Voxtral adapter requires a pipeline");
    return [pipeline = std::move(pipeline)]() -> std::unique_ptr<VoxtralAudioAdapter::InferenceStream> {
        auto stream = pipeline->create_stream();
        return std::make_unique<GenAiInferenceStream<decltype(stream)>>(std::move(stream));
    };
}
}

VoxtralAudioAdapter::VoxtralAudioAdapter(std::shared_ptr<ov::genai::ASRPipeline> pipeline) :
    VoxtralAudioAdapter(makeGenAiStreamFactory(std::move(pipeline))) {}

VoxtralAudioAdapter::VoxtralAudioAdapter(StreamFactory streamFactory) :
    streamFactory_(std::move(streamFactory)) {
    if (!streamFactory_)
        throw std::invalid_argument("Voxtral adapter requires a stream factory");
}

std::shared_ptr<VoxtralAudioAdapter::Stream> VoxtralAudioAdapter::createStream(
    TextCallback deltaCallback, TextCallback finalCallback, ErrorCallback errorCallback) const {
    return std::make_shared<Stream>(streamFactory_, std::move(deltaCallback), std::move(finalCallback),
        std::move(errorCallback));
}

VoxtralAudioAdapter::Stream::Stream(StreamFactory streamFactory,
    TextCallback deltaCallback, TextCallback finalCallback, ErrorCallback errorCallback) :
    streamFactory_(std::move(streamFactory)),
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
    hasAudio_ = true;
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

bool VoxtralAudioAdapter::Stream::isCompleted() const {
    return completed_.load();
}

bool VoxtralAudioAdapter::Stream::hasAudio() const {
    return hasAudio_.load();
}

void VoxtralAudioAdapter::Stream::run() {
    try {
        auto stream = streamFactory_();
        if (!stream)
            throw std::runtime_error("Voxtral stream factory returned an empty stream");
        std::exception_ptr inputError;
        std::thread inputWorker([this, &stream, &inputError] {
            try {
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
                    const auto audio = SttAudioAdapter::prepareAudio(frame);
                    stream->pushAudio(audio);
                    SPDLOG_LOGGER_TRACE(webrtc_logger, "Voxtral push_audio: samples={}, queued_frames={}, elapsed_ms={}",
                        frame.samples.size(), queuedFrames,
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
                }
                stream->endAudio();
            } catch (...) {
                inputError = std::current_exception();
                try {
                    stream->endAudio();
                } catch (...) {
                }
            }
        });

        std::string transcript;
        std::exception_ptr generationError;
        try {
            transcript = stream->generate([this](const std::string& delta) {
                if (!delta.empty()) {
                    SPDLOG_LOGGER_INFO(webrtc_logger, "Voxtral transcript delta: {}", delta);
                    if (deltaCallback_)
                        deltaCallback_(delta);
                }
            });
        } catch (...) {
            generationError = std::current_exception();
        }
        inputWorker.join();
        if (generationError)
            std::rethrow_exception(generationError);
        if (inputError)
            std::rethrow_exception(inputError);
        SPDLOG_LOGGER_INFO(webrtc_logger, "Voxtral final transcript: {}", transcript);
        if (finalCallback_)
            finalCallback_(transcript);
    } catch (...) {
        if (errorCallback_) {
            try {
                errorCallback_(std::current_exception());
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Voxtral error callback failed: {}", e.what());
            } catch (...) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Voxtral error callback failed with an unknown exception");
            }
        }
    }
    completed_ = true;
}

}  // namespace ovms
