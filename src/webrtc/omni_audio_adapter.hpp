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
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <openvino/genai/omni/pipeline.hpp>

#include "streaming_audio_model.hpp"

namespace ovms {

class OmniAudioAdapter {
public:
    struct Conversation;
    using ConversationPtr = std::shared_ptr<Conversation>;
    using AudioCallback = std::function<void(AudioChunk)>;
    using ErrorCallback = std::function<void(std::exception_ptr)>;
    using CompletionCallback = std::function<void()>;

    explicit OmniAudioAdapter(std::shared_ptr<ov::genai::OmniPipeline> pipeline);
    ~OmniAudioAdapter();

    OmniAudioAdapter(const OmniAudioAdapter&) = delete;
    OmniAudioAdapter& operator=(const OmniAudioAdapter&) = delete;

    ConversationPtr createConversation() const;
    void submit(ConversationPtr conversation, AudioChunk utterance, AudioCallback audioCallback,
        ErrorCallback errorCallback = {}, CompletionCallback completionCallback = {});
    void cancel();

private:
    void run();
    void generate(Conversation& conversation, const AudioChunk& utterance, const AudioCallback& audioCallback);

    struct Request {
        ConversationPtr conversation;
        AudioChunk utterance;
        AudioCallback audioCallback;
        ErrorCallback errorCallback;
        CompletionCallback completionCallback;
    };

    static std::vector<float> resample(const AudioChunk& input, uint32_t outputRate);

    std::shared_ptr<ov::genai::OmniPipeline> pipeline_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<Request> requests_;
    std::thread worker_;
    std::atomic<bool> stopping_{false};
};

}  // namespace ovms