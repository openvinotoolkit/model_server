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

#include "legacy_executor.hpp"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>
#include <utility>

#pragma warning(push)
#pragma warning(disable : 6001)
#include "absl/strings/escaping.h"
#pragma warning(pop)

#include "src/port/rapidjson_document.hpp"
#include "servable.hpp"

namespace ovms {
OmniModelLegacyExecutor::OmniModelLegacyExecutor(std::shared_ptr<ov::genai::OmniPipeline> pipe) {
    this->pipe = std::move(pipe);
}

bool OmniModelLegacyExecutor::hasRequests() {
    return (requests.size() > 0);
}

size_t OmniModelLegacyExecutor::requestsQueueSize() {
    return requests.size();
}

void OmniModelLegacyExecutor::processRequest() {
    OVMS_PROFILE_FUNCTION();
    auto& requestExecutionContext = requests.front();
    if (requestExecutionContext->clientDisconnected) {
        requestExecutionContext->success = false;
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Client disconnected, skipping request processing.");
    } else {
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Omni generation started");
        try {
            ov::genai::OmniTalkerSpeechConfig speechConfig;
            speechConfig.audio_chunk_frames = 4;
            speechConfig.return_audio = requestExecutionContext->audioOutputRequested;
            if (!requestExecutionContext->audioVoice.empty()) {
                speechConfig.speaker = requestExecutionContext->audioVoice;
            }

            SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Omni generate: prompt length={}, images={}, videos={}, audios={}, return_audio={}",
                requestExecutionContext->inputRequest.promptText.size(),
                requestExecutionContext->inputRequest.inputImages.size(),
                requestExecutionContext->inputRequest.inputVideos.size(),
                requestExecutionContext->inputRequest.inputAudios.size(),
                requestExecutionContext->audioOutputRequested);
            for (size_t i = 0; i < requestExecutionContext->inputRequest.inputAudios.size(); i++) {
                const auto& t = requestExecutionContext->inputRequest.inputAudios[i];
                SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Omni audio[{}]: shape={}, element_type={}",
                    i, t.get_shape().to_string(), t.get_element_type().to_string());
            }

            std::vector<ov::genai::VideoMetadata> videosMetadata;

            // Create speech streamer for streaming audio output via SSE
            ov::genai::OmniSpeechStreamerVariant speechStreamer = std::monostate{};
            size_t audioChunkCount = 0;
            auto lastChunkReceiveTime = std::chrono::steady_clock::now();
            auto speechStreamStart = std::chrono::steady_clock::now();
            if (requestExecutionContext->audioOutputRequested && requestExecutionContext->textStreamer) {
                speechStreamer = [& ctx = *requestExecutionContext, &audioChunkCount, &lastChunkReceiveTime](const ov::Tensor& audio_chunk) -> ov::genai::StreamingStatus {
                    auto timeSinceLastChunk = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lastChunkReceiveTime).count();
                    auto serializationStartTime = std::chrono::steady_clock::now();
                    lastChunkReceiveTime = std::chrono::steady_clock::now();

                    if (ctx.clientDisconnected.load()) {
                        return ov::genai::StreamingStatus::CANCEL;
                    }
                    if (audioChunkCount == 0) {
                        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Omni speech: first audio chunk received in {} ms", timeSinceLastChunk);
                    } else {
                        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Omni speech: next audio chunk received in {} ms", timeSinceLastChunk);
                    }
                    // Convert float32 PCM to int16 and base64 encode
                    const float* pcm = audio_chunk.data<const float>();
                    const size_t count = audio_chunk.get_size();
                    std::vector<int16_t> pcm16(count);
                    for (size_t i = 0; i < count; i++) {
                        float s = pcm[i];
                        if (s > 1.0f)
                            s = 1.0f;
                        if (s < -1.0f)
                            s = -1.0f;
                        pcm16[i] = static_cast<int16_t>(s * 32767.0f);
                    }
                    std::string b64 = absl::Base64Escape(
                        std::string_view(reinterpret_cast<const char*>(pcm16.data()), pcm16.size() * sizeof(int16_t)));

                    rapidjson::Document audioDoc;
                    audioDoc.SetObject();
                    audioDoc.AddMember("_audio_delta",
                        rapidjson::Value(b64.c_str(), audioDoc.GetAllocator()),
                        audioDoc.GetAllocator());
                    ctx.deltaChannel.push(std::move(audioDoc));
                    audioChunkCount++;
                    auto serializationTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - serializationStartTime).count();
                    SPDLOG_LOGGER_TRACE(llm_executor_logger, "Omni : Deserialization time {} ms", serializationTimeMs);
                    lastChunkReceiveTime = std::chrono::steady_clock::now();
                    return ov::genai::StreamingStatus::RUNNING;
                };
            }

            auto generateStart = std::chrono::steady_clock::now();
            requestExecutionContext->results = pipe->generate(
                requestExecutionContext->inputRequest.promptText,
                requestExecutionContext->inputRequest.inputImages,
                requestExecutionContext->inputRequest.inputVideos,
                videosMetadata,
                requestExecutionContext->inputRequest.inputAudios,
                requestExecutionContext->inputRequest.generationConfig,
                speechConfig,
                requestExecutionContext->textStreamer,
                speechStreamer);
            auto generateEnd = std::chrono::steady_clock::now();
            auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(generateEnd - generateStart).count();
            SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Omni generate complete: total={} ms, audio_chunks={}",
                totalMs, audioChunkCount);
            if (audioChunkCount > 0) {
                auto speechMs = std::chrono::duration_cast<std::chrono::milliseconds>(generateEnd - speechStreamStart).count();
                SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Omni speech phase: {} ms for {} chunks ({:.1f} ms/chunk)",
                    speechMs, audioChunkCount, static_cast<float>(speechMs) / audioChunkCount);
            }
        } catch (std::exception& e) {
            requestExecutionContext->success = false;
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "Omni pipeline generation failed: {}.", e.what());
        }
        SPDLOG_LOGGER_TRACE(llm_executor_logger, "Omni generation ended");
    }
    requestExecutionContext->readySignal.set_value();
    requestExecutionContext->deltaChannel.signalComplete();
    std::unique_lock<std::mutex> lock(queueMutex);
    requests.pop();
}

void OmniModelLegacyExecutor::waitForRequests(std::atomic<bool>* receivedEndSignal) {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.wait(lock, [this, receivedEndSignal] { return (requests.size() > 0 || *receivedEndSignal); });
}

void OmniModelLegacyExecutor::addRequest(std::shared_ptr<OmniModelLegacyServableExecutionContext> request) {
    std::lock_guard<std::mutex> guard(queueMutex);
    requests.push(request);
    cv.notify_one();
}

void OmniModelLegacyExecutor::notify() {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.notify_one();
}

void OmniModelLegacyExecutorWrapper::run(OmniModelLegacyExecutor* executor, std::atomic<bool>* receivedEndSignal) {
    while (!(*receivedEndSignal)) {
        try {
            SPDLOG_LOGGER_TRACE(llm_executor_logger, "Omni executor all requests: {};", executor->requestsQueueSize());
            if (executor->hasRequests()) {
                executor->processRequest();
            } else {
                executor->waitForRequests(receivedEndSignal);
            }
        } catch (std::exception& e) {
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error occurred in Omni executor: {}.", e.what());
            exit(1);
        }
    }
}

OmniModelLegacyExecutorWrapper::OmniModelLegacyExecutorWrapper(std::shared_ptr<ov::genai::OmniPipeline> pipe) :
    executor(std::move(pipe)) {
    executorThread = std::thread(OmniModelLegacyExecutorWrapper::run, &executor, &finishExecutorThread);
}

OmniModelLegacyExecutorWrapper::~OmniModelLegacyExecutorWrapper() {
    finishExecutorThread = true;
    executor.notify();
    executorThread.join();
}

void OmniModelLegacyExecutorWrapper::addRequest(std::shared_ptr<OmniModelLegacyServableExecutionContext> request) {
    executor.addRequest(request);
}
}  // namespace ovms
