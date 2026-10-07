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

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "streaming_audio_model.hpp"

namespace ov::genai {
class ASRPipeline;
}

namespace ovms {

class VoxtralAudioAdapter {
public:
    using TextCallback = std::function<void(const std::string&)>;
    using ErrorCallback = std::function<void(std::exception_ptr)>;

    class Stream {
    public:
        Stream(std::shared_ptr<ov::genai::ASRPipeline> pipeline, TextCallback deltaCallback,
            TextCallback finalCallback, ErrorCallback errorCallback);
        ~Stream();

        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        void push(AudioChunk frame);
        void finish();
        void close();

    private:
        void run();

        std::shared_ptr<ov::genai::ASRPipeline> pipeline_;
        TextCallback deltaCallback_;
        TextCallback finalCallback_;
        ErrorCallback errorCallback_;
        std::queue<AudioChunk> frames_;
        std::mutex mutex_;
        std::condition_variable condition_;
        bool finishing_ = false;
        std::thread worker_;
    };

    VoxtralAudioAdapter(std::string modelPath, std::string device);

    std::shared_ptr<Stream> createStream(TextCallback deltaCallback, TextCallback finalCallback,
        ErrorCallback errorCallback) const;

private:
    std::shared_ptr<ov::genai::ASRPipeline> pipeline_;
};

}  // namespace ovms