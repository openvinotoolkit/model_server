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
#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <openvino/genai/automatic_speech_recognition/pipeline.hpp>

#include "streaming_audio_model.hpp"

namespace ovms {

class SttAudioAdapter {
public:
    using TextCallback = std::function<void(const std::string&)>;
    using ErrorCallback = std::function<void(std::exception_ptr)>;
    using CompletionCallback = std::function<void(const std::string&)>;

    SttAudioAdapter(std::string modelPath, std::string device);
    ~SttAudioAdapter();

    SttAudioAdapter(const SttAudioAdapter&) = delete;
    SttAudioAdapter& operator=(const SttAudioAdapter&) = delete;

    void submit(AudioChunk utterance, TextCallback textCallback, ErrorCallback errorCallback,
        CompletionCallback completionCallback);

    static std::vector<float> prepareAudio(const AudioChunk& utterance);

private:
    struct Request {
        AudioChunk utterance;
        TextCallback textCallback;
        ErrorCallback errorCallback;
        CompletionCallback completionCallback;
    };

    void run();
    void transcribe(Request& request);

    std::shared_ptr<ov::genai::ASRPipeline> pipeline_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<Request> requests_;
    std::thread worker_;
    bool stopping_ = false;
};

}  // namespace ovms