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
#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "streaming_audio_model.hpp"

namespace ov::genai {
class ASRPipeline;
}

namespace ovms {

class VoxtralAudioAdapter {
public:
    using TextCallback = std::function<void(const std::string&)>;
    using ErrorCallback = std::function<void(std::exception_ptr)>;

    class InferenceStream {
    public:
        virtual ~InferenceStream() = default;
        virtual void pushAudio(const std::vector<float>& audio) = 0;
        virtual void endAudio() = 0;
        virtual std::string generate(TextCallback deltaCallback) = 0;
    };

    using StreamFactory = std::function<std::unique_ptr<InferenceStream>()>;

    class Stream {
    public:
        Stream(StreamFactory streamFactory, TextCallback deltaCallback,
            TextCallback finalCallback, ErrorCallback errorCallback);
        ~Stream();

        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        void push(AudioChunk frame);
        void finish();
        void close();
        // True once the final transcript or error callback has been delivered.
        bool isCompleted() const;
        bool hasAudio() const;

    private:
        void run();

        StreamFactory streamFactory_;
        TextCallback deltaCallback_;
        TextCallback finalCallback_;
        ErrorCallback errorCallback_;
        std::queue<AudioChunk> frames_;
        std::mutex mutex_;
        std::condition_variable condition_;
        bool finishing_ = false;
        std::atomic<bool> hasAudio_{false};
        std::atomic<bool> completed_{false};
        std::thread worker_;
    };

    /// Shares a served realtime pipeline; each stream owns its own inference state.
    explicit VoxtralAudioAdapter(std::shared_ptr<ov::genai::ASRPipeline> pipeline);
    explicit VoxtralAudioAdapter(StreamFactory streamFactory);

    std::shared_ptr<Stream> createStream(TextCallback deltaCallback, TextCallback finalCallback,
        ErrorCallback errorCallback) const;

private:
    StreamFactory streamFactory_;
};

}  // namespace ovms
