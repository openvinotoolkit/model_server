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

#include "s2t_servable.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <string_view>
#include <stdexcept>
#include <thread>

#pragma warning(push)
#pragma warning(disable : 6386)
#include "absl/status/status.h"
#pragma warning(pop)
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

#include "src/audio/speech_to_text/s2t_calculator.pb.h"
#include "src/http_payload.hpp"
#include "src/json_parser.hpp"
#include "src/logging.hpp"
#include "src/ov_utils.hpp"
#include "src/stringutils.hpp"

namespace ovms {

namespace {
constexpr size_t ISO_LANG_CODE_MAX = 3;

std::optional<std::string_view> qwen3AsrLanguageName(std::string_view language) {
    static const std::unordered_map<std::string_view, std::string_view> languageNames{
        {"zh", "Chinese"}, {"en", "English"}, {"yue", "Cantonese"}, {"ar", "Arabic"},
        {"de", "German"}, {"fr", "French"}, {"es", "Spanish"}, {"pt", "Portuguese"},
        {"id", "Indonesian"}, {"it", "Italian"}, {"ko", "Korean"}, {"ru", "Russian"},
        {"th", "Thai"}, {"vi", "Vietnamese"}, {"ja", "Japanese"}, {"tr", "Turkish"},
        {"hi", "Hindi"}, {"ms", "Malay"}, {"nl", "Dutch"}, {"sv", "Swedish"},
        {"da", "Danish"}, {"fi", "Finnish"}, {"pl", "Polish"}, {"cs", "Czech"},
        {"fil", "Filipino"}, {"fa", "Persian"}, {"el", "Greek"}, {"ro", "Romanian"},
        {"hu", "Hungarian"}, {"mk", "Macedonian"}};
    const auto it = languageNames.find(language);
    if (it != languageNames.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::string qwen3AsrPrompt(const std::optional<std::string_view>& language) {
    std::string prompt = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    if (language.has_value()) {
        prompt += "language ";
        prompt += language.value();
        prompt += "<asr_text>";
    }
    return prompt;
}
}
SttServable::SttServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath) :
    enableWordTimestamps(nodeOptions.enable_word_timestamps()) {
    auto fsModelsPath = std::filesystem::path(nodeOptions.models_path());
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
    } else {
        parsedModelsPath = fsModelsPath;
    }
}

AsrPipelineServable::AsrPipelineServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath) :
    SttServable(nodeOptions, graphPath) {
    std::string device = nodeOptions.target_device();
    if (device.empty()) {
        device = recommendTargetDevice();
        SPDLOG_INFO("No device specified for STT model, using recommended device: {}", device);
    }
    ov::AnyMap config;
    auto status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), config);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        throw std::runtime_error("Error during plugin_config option parsing");
    }
    applyGlobalCacheDirFallback(config);
    if (enableWordTimestamps && device == "NPU") {
        config["STATIC_PIPELINE"] = true;
    }
    config["word_timestamps"] = enableWordTimestamps;
    sttPipeline = std::make_shared<ov::genai::ASRPipeline>(parsedModelsPath.string(), device, config);

    streamingExecutor = std::make_unique<SttExecutorWrapper>(sttPipeline, sttPipelineMutex);
}

bool AsrPipelineServable::supportsStreaming() const {
    return true;
}

bool AsrPipelineServable::supportsTranslations() const {
    return true;
}

absl::Status AsrPipelineServable::transcribe(const std::vector<float>& rawSpeech,
    const HttpPayload& payload,
    const std::function<bool()>& isClientDisconnected,
    SttTranscriptionResult& result) {
    ov::genai::ASRGenerationConfig config = sttPipeline->get_generation_config();
    auto status = updateTranscriptionConfig(config, *this, payload);
    if (!status.ok()) {
        return status;
    }
    auto disconnectCallback = [&isClientDisconnected](std::string) -> ov::genai::StreamingStatus {
        return isClientDisconnected() ? ov::genai::StreamingStatus::CANCEL : ov::genai::StreamingStatus::RUNNING;
    };
    std::unique_lock lock(sttPipelineMutex);
    if (isClientDisconnected()) {
        return absl::CancelledError("Client disconnected");
    }
    const ov::genai::ASRDecodedResults generated = sttPipeline->generate(rawSpeech, config, disconnectCallback);
    lock.unlock();
    if (isClientDisconnected()) {
        return absl::CancelledError("Client disconnected");
    }
    result.text = generated;
    result.wordTimestampsRequested = config.word_timestamps;
    result.segmentTimestampsRequested = config.return_timestamps;
    if (generated.words.has_value()) {
        for (const auto& wordGroup : generated.words.value()) {
            for (const auto& word : wordGroup) {
                result.words.push_back({word.text, word.start_ts, word.end_ts});
            }
        }
    }
    if (generated.chunks.has_value()) {
        for (const auto& chunkGroup : generated.chunks.value()) {
            for (const auto& chunk : chunkGroup) {
                result.segments.push_back({chunk.text, chunk.start_ts, chunk.end_ts});
            }
        }
    }
    return absl::OkStatus();
}

void AsrPipelineServable::addRequest(std::shared_ptr<SttServableExecutionContext> executionContext) {
    if (!streamingExecutor) {
        throw std::runtime_error("Cannot schedule STT streaming job - executor not initialized");
    }
    streamingExecutor->addRequest(std::move(executionContext));
}

absl::Status SttServable::parseTemperature(const HttpPayload& payload, ov::genai::ASRGenerationConfig& config) {
    std::string temperatureStr = payload.multipartParser->getFieldByName("temperature");
    if (temperatureStr.size() > 0) {
        SPDLOG_LOGGER_TRACE(s2t_calculator_logger, "Received temperature: {}", temperatureStr);
        auto temp = ovms::stof(temperatureStr);
        if (!temp.has_value())
            return absl::InvalidArgumentError("Invalid temperature type.");
        config.temperature = temp.value();
        if (config.temperature != 0) {
            config.do_sample = true;
        }
    }
    return absl::OkStatus();
}

struct Qwen3AsrContinuousBatchingServable::Impl {
    struct Request {
        std::vector<float> rawSpeech;
        std::string prompt;
        std::optional<float> temperature;
        std::promise<std::string> result;
        std::function<bool()> isClientDisconnected;
    };

    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipeline;
    std::mutex requestsMutex;
    std::condition_variable requestsCv;
    std::deque<std::shared_ptr<Request>> requests;
    std::atomic<bool> stop{false};
    std::thread executor;
    std::atomic<uint64_t> nextRequestId{0};

    Impl(const std::filesystem::path& modelsPath, const std::string& device, const ov::AnyMap& config, ov::genai::SchedulerConfig schedulerConfig) :
        pipeline(std::make_shared<ov::genai::ContinuousBatchingPipeline>(modelsPath, std::move(schedulerConfig), device, config)),
        executor([this] { run(); }) {}

    ~Impl() {
        stop = true;
        requestsCv.notify_one();
        if (executor.joinable()) {
            executor.join();
        }
    }

    void addPendingRequests(std::vector<std::pair<std::shared_ptr<Request>, ov::genai::GenerationHandle>>& activeRequests) {
        std::deque<std::shared_ptr<Request>> pendingRequests;
        {
            std::lock_guard<std::mutex> lock(requestsMutex);
            pendingRequests.swap(requests);
        }
        for (auto& request : pendingRequests) {
            if (request->isClientDisconnected()) {
                request->result.set_exception(std::make_exception_ptr(std::runtime_error("Client disconnected")));
                continue;
            }
            ov::Tensor audio(ov::element::f32, {request->rawSpeech.size()});
            std::copy(request->rawSpeech.begin(), request->rawSpeech.end(), audio.data<float>());
            const std::vector<ov::Tensor> audios{audio};
            auto generationConfig = pipeline->get_config();
            if (request->temperature.has_value()) {
                generationConfig.temperature = request->temperature.value();
                generationConfig.do_sample = generationConfig.temperature != 0;
            }
            auto handle = pipeline->add_request(nextRequestId++, request->prompt, ov::genai::audios(audios), ov::genai::generation_config(generationConfig));
            activeRequests.emplace_back(std::move(request), std::move(handle));
        }
    }

    void run() {
        std::vector<std::pair<std::shared_ptr<Request>, ov::genai::GenerationHandle>> activeRequests;
        while (!stop) {
            if (activeRequests.empty()) {
                std::unique_lock<std::mutex> lock(requestsMutex);
                requestsCv.wait(lock, [this] { return stop || !requests.empty(); });
                if (stop) {
                    return;
                }
            }
            addPendingRequests(activeRequests);
            if (activeRequests.empty()) {
                continue;
            }
            pipeline->step();
            addPendingRequests(activeRequests);
            for (auto it = activeRequests.begin(); it != activeRequests.end();) {
                const auto status = it->second->get_status();
                if (status == ov::genai::GenerationStatus::RUNNING) {
                    ++it;
                    continue;
                }
                if (status == ov::genai::GenerationStatus::FINISHED) {
                    const auto outputs = it->second->read_all();
                    it->first->result.set_value(pipeline->get_tokenizer().decode(outputs.front().generated_ids));
                } else {
                    it->first->result.set_exception(std::make_exception_ptr(std::runtime_error("Qwen3 ASR generation failed")));
                }
                it = activeRequests.erase(it);
            }
        }
    }
};

Qwen3AsrContinuousBatchingServable::Qwen3AsrContinuousBatchingServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath) :
    SttServable(nodeOptions, graphPath) {
    std::string device = nodeOptions.target_device();
    if (device.empty()) {
        device = recommendTargetDevice();
        SPDLOG_INFO("No device specified for Qwen3 ASR model, using recommended device: {}", device);
    }
    ov::AnyMap config;
    auto status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), config);
    if (!status.ok()) {
        throw std::runtime_error("Error during plugin_config option parsing");
    }
    applyGlobalCacheDirFallback(config);
    ov::genai::SchedulerConfig schedulerConfig;
    schedulerConfig.max_num_batched_tokens = nodeOptions.max_num_batched_tokens();
    schedulerConfig.max_num_seqs = nodeOptions.max_num_seqs();
    schedulerConfig.dynamic_split_fuse = nodeOptions.dynamic_split_fuse();
    impl = std::make_unique<Impl>(parsedModelsPath, device, config, std::move(schedulerConfig));
}

Qwen3AsrContinuousBatchingServable::~Qwen3AsrContinuousBatchingServable() = default;

bool Qwen3AsrContinuousBatchingServable::supportsStreaming() const {
    return false;
}

bool Qwen3AsrContinuousBatchingServable::supportsTranslations() const {
    return false;
}

absl::Status Qwen3AsrContinuousBatchingServable::transcribe(const std::vector<float>& rawSpeech,
    const HttpPayload& payload,
    const std::function<bool()>& isClientDisconnected,
    SttTranscriptionResult& result) {
    const std::string language = payload.multipartParser->getFieldByName("language");
    std::optional<std::string_view> languageName;
    if (!language.empty()) {
        languageName = qwen3AsrLanguageName(language);
        if (!languageName.has_value()) {
            return absl::InvalidArgumentError("Unsupported Qwen3 ASR language code.");
        }
    }
    if (!payload.multipartParser->getArrayFieldByName("timestamp_granularities[]").empty()) {
        return absl::InvalidArgumentError("timestamps are not supported for Qwen3 ASR continuous batching");
    }
    std::optional<float> temperature;
    const std::string temperatureString = payload.multipartParser->getFieldByName("temperature");
    if (!temperatureString.empty()) {
        temperature = ovms::stof(temperatureString);
        if (!temperature.has_value()) {
            return absl::InvalidArgumentError("Invalid temperature type.");
        }
    }
    auto request = std::make_shared<Impl::Request>();
    request->rawSpeech = rawSpeech;
    request->prompt = qwen3AsrPrompt(languageName);
    request->temperature = temperature;
    request->isClientDisconnected = isClientDisconnected;
    auto future = request->result.get_future();
    {
        std::lock_guard<std::mutex> lock(impl->requestsMutex);
        impl->requests.push_back(std::move(request));
    }
    impl->requestsCv.notify_one();
    result.text = future.get();
    return absl::OkStatus();
}

absl::Status SttServable::updateTranscriptionConfig(ov::genai::ASRGenerationConfig& config,
    const SttServable& servable, const HttpPayload& payload) {
    std::string language = payload.multipartParser->getFieldByName("language");
    if (language.size() > 0) {
        if (language.size() > ISO_LANG_CODE_MAX) {
            return absl::InvalidArgumentError("Invalid language code.");
        }
        SPDLOG_LOGGER_TRACE(s2t_calculator_logger, "Received language: {}", language);
        config.language = "<|" + language + "|>";
    }
    std::vector<std::string> timestampsTypes = payload.multipartParser->getArrayFieldByName("timestamp_granularities[]");
    config.word_timestamps = false;
    for (const auto& timestampsType : timestampsTypes) {
        SPDLOG_LOGGER_TRACE(s2t_calculator_logger, "Received timestamp type: {}", timestampsType);
        if (timestampsType == "segment") {
            config.return_timestamps = true;
        } else if (timestampsType == "word") {
            if (!servable.enableWordTimestamps)
                return absl::InvalidArgumentError("Word timestamps not supported for this model");
            config.word_timestamps = true;
        } else {
            return absl::InvalidArgumentError("Invalid timestamp_granularities type. Allowed types: \"segment\", \"word\"");
        }
    }
    auto status = parseTemperature(payload, config);
    if (status != absl::OkStatus())
        return status;
    return absl::OkStatus();
}

absl::Status SttServable::updateTranscriptionConfig(ov::genai::ASRGenerationConfig& config,
    const std::shared_ptr<SttServable>& servable, const HttpPayload& payload) {
    return updateTranscriptionConfig(config, *servable, payload);
}

}  // namespace ovms
