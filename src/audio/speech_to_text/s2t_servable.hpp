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
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "src/audio/speech_to_text/s2t_executor.hpp"
#include "src/status.hpp"

namespace mediapipe {
class S2tCalculatorOptions;
}  // namespace mediapipe

namespace ov::genai {
class ASRPipeline;
class ASRGenerationConfig;
}  // namespace ov::genai

namespace ovms {

struct HttpPayload;

struct SttTimestamp {
    std::string text;
    float start;
    float end;
};

struct SttTranscriptionResult {
    std::string text;
    bool wordTimestampsRequested = false;
    bool segmentTimestampsRequested = false;
    std::vector<SttTimestamp> words;
    std::vector<SttTimestamp> segments;
};

struct SttServable {
    std::filesystem::path parsedModelsPath;
    bool enableWordTimestamps;

    SttServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath);
    virtual ~SttServable() = default;

    virtual bool supportsStreaming() const = 0;
    virtual bool supportsTranslations() const = 0;
    virtual absl::Status transcribe(const std::vector<float>& rawSpeech,
        const HttpPayload& payload,
        const std::function<bool()>& isClientDisconnected,
        SttTranscriptionResult& result) = 0;

    static absl::Status parseTemperature(const HttpPayload& payload, ov::genai::ASRGenerationConfig& config);

    static absl::Status updateTranscriptionConfig(ov::genai::ASRGenerationConfig& config,
        const SttServable& servable, const HttpPayload& payload);
    static absl::Status updateTranscriptionConfig(ov::genai::ASRGenerationConfig& config,
        const std::shared_ptr<SttServable>& servable, const HttpPayload& payload);
};

struct AsrPipelineServable final : public SttServable {
    std::shared_ptr<ov::genai::ASRPipeline> sttPipeline;
    std::mutex sttPipelineMutex;

    std::unique_ptr<SttExecutorWrapper> streamingExecutor;

    AsrPipelineServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath);

    bool supportsStreaming() const override;
    bool supportsTranslations() const override;
    absl::Status transcribe(const std::vector<float>& rawSpeech,
        const HttpPayload& payload,
        const std::function<bool()>& isClientDisconnected,
        SttTranscriptionResult& result) override;

    void addRequest(std::shared_ptr<SttServableExecutionContext> executionContext);
};

class Qwen3AsrContinuousBatchingServable final : public SttServable {
public:
    Qwen3AsrContinuousBatchingServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath);
    ~Qwen3AsrContinuousBatchingServable() override;

    bool supportsStreaming() const override;
    bool supportsTranslations() const override;
    absl::Status transcribe(const std::vector<float>& rawSpeech,
        const HttpPayload& payload,
        const std::function<bool()>& isClientDisconnected,
        SttTranscriptionResult& result) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

using SttServableMap = std::unordered_map<std::string, std::shared_ptr<SttServable>>;
}  // namespace ovms
