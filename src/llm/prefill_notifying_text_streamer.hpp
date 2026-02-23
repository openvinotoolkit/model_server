//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <functional>
#include <mutex>

#include "openvino/genai/text_streamer.hpp"

namespace ovms {

// Subclass of TextStreamer that notifies when prefill completes via on_prefill_end() callback.
// Used in legacy servables where GenAI calls on_prefill_end() on the streamer during generate().
class PrefillNotifyingTextStreamer : public ov::genai::TextStreamer {
    std::mutex& m_mutex;
    std::condition_variable& m_cv;
    bool& m_prefillEndNotified;

public:
    PrefillNotifyingTextStreamer(
        const ov::genai::Tokenizer& tokenizer,
        std::function<ov::genai::CallbackTypeVariant(std::string)> callback,
        std::mutex& mutex,
        std::condition_variable& cv,
        bool& prefillEndNotified,
        const ov::AnyMap& params = {})
        : TextStreamer(tokenizer, std::move(callback), params),
          m_mutex(mutex),
          m_cv(cv),
          m_prefillEndNotified(prefillEndNotified) {}

    void on_prefill_end() override {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_prefillEndNotified = true;
        }
        m_cv.notify_one();
    }
};

}  // namespace ovms
