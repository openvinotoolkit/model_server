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

#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/audio/speech_to_text/s2t_executor.hpp"
#include "src/status.hpp"

namespace absl {
class Status;
}  // namespace absl

namespace mediapipe {
class S2tCalculatorOptions;
}  // namespace mediapipe

namespace ov::genai {
class WhisperPipeline;
class WhisperGenerationConfig;
}  // namespace ov::genai

namespace ovms {

struct HttpPayload;

struct SttServable {
    using StreamingJob = SttStreamingJob;

    std::filesystem::path parsedModelsPath;
    std::shared_ptr<ov::genai::WhisperPipeline> sttPipeline;
    std::mutex sttPipelineMutex;
    bool enableWordTimestamps;

    std::unique_ptr<SttExecutorWrapper> streamingExecutor;

    SttServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath);

    ~SttServable() = default;

    void addRequest(std::shared_ptr<SttServableExecutionContext> executionContext);

    static absl::Status parseTemperature(const HttpPayload& payload, float& temperature);

    static absl::Status applyTranscriptionConfig(ov::genai::WhisperGenerationConfig& config,
        const std::shared_ptr<SttServable>& servable, const HttpPayload& payload);
};

using SttServableMap = std::unordered_map<std::string, std::shared_ptr<SttServable>>;
}  // namespace ovms
