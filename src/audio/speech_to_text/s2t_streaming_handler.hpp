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

#include <future>
#include <memory>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "openvino/genai/whisper_pipeline.hpp"

#include "src/http_payload.hpp"
namespace ovms {
class StreamingTextQueue;
struct SttServable;
struct SttServableExecutionContext;
}  // namespace ovms

namespace mediapipe {

// Encapsulates all streaming state and logic for the S2tCalculator.
// Manages the background generation thread, the text queue, SSE
// serialization and LOOPBACK signaling.
class S2tStreamingHandler {
public:
    static std::string serializeDeltaEvent(const std::string& delta);
    static std::string serializeDoneEvent(const std::string& text);

    absl::Status start(CalculatorContext* cc,
        std::shared_ptr<ovms::SttServable> pipe,
        const ovms::HttpPayload& payload,
        std::vector<float> rawSpeech,
        bool isTranscription,
        const std::string& loopbackTag,
        const std::string& outputTag);

    absl::Status processIteration(CalculatorContext* cc,
        const std::string& loopbackTag,
        const std::string& outputTag);

private:
    bool isStreaming_ = false;
    std::shared_ptr<ovms::StreamingTextQueue> streamingQueue_;
    std::shared_ptr<ovms::SttServableExecutionContext> executionContext_;
    ::mediapipe::Timestamp iterationTimestamp_{0};
};

}  // namespace mediapipe
