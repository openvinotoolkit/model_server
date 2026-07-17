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

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"

#include "../../../logging.hpp"
#include "../../../status.hpp"
#include "../../apis/openai_completions.hpp"
#include "../../apis/openai_responses.hpp"
#include "../../ovms_text_streamer.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "../../../config.hpp"
#include "../../../http_payload.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../text_utils.hpp"
#include "../../../tokenize/tokenize_parser.hpp"
#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif
#include "../../io_processing/generation_config_builder.hpp"
#include "src/audio/audio_utils.hpp"
#include "servable.hpp"

namespace ovms {

absl::Status OmniModelLegacyServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
    if (payload.parsedJson->HasParseError()) {
        return absl::InvalidArgumentError("Non-json request received in text generation calculator");
    }
    if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions") {
        executionContext->endpoint = Endpoint::CHAT_COMPLETIONS;
    } else if (payload.uri == "/v3/responses" || payload.uri == "/v3/v1/responses") {
        executionContext->endpoint = Endpoint::RESPONSES;
    } else if (TokenizeParser::isTokenizeEndpoint(payload.uri)) {
        executionContext->endpoint = Endpoint::TOKENIZE;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. Omni Servable allowed only on /v3/chat/completions, /v3/responses endpoint or /v3/tokenize");
    }
    executionContext->payload = payload;
    return absl::OkStatus();
}

std::shared_ptr<GenAiServableExecutionContext> OmniModelLegacyServable::createExecutionContext() {
    return std::make_shared<OmniModelLegacyServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> OmniModelLegacyServable::getProperties() {
    return properties;
}

absl::Status OmniModelLegacyServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto omniExecutionContext = std::static_pointer_cast<OmniModelLegacyServableExecutionContext>(executionContext);
    if (omniExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }

    omniExecutionContext->baseGenerationConfig = properties->baseGenerationConfig;
    try {
        if (omniExecutionContext->endpoint == Endpoint::RESPONSES) {
            omniExecutionContext->apiHandler = std::make_shared<OpenAIResponsesHandler>(*omniExecutionContext->payload.parsedJson,
                omniExecutionContext->endpoint,
                std::chrono::system_clock::now(),
                getProperties()->tokenizer,
                getProperties()->toolParserName,
                getProperties()->reasoningParserName);
        } else {
            omniExecutionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*omniExecutionContext->payload.parsedJson,
                omniExecutionContext->endpoint,
                std::chrono::system_clock::now(),
                getProperties()->tokenizer,
                getProperties()->toolParserName,
                getProperties()->reasoningParserName);
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to create API handler: {}", e.what());
        return absl::InvalidArgumentError(std::string("Failed to create API handler: ") + e.what());
    }
    auto& config = ovms::Config::instance();

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->maxModelLength, config.getServerSettings().allowedLocalMediaPath, config.getServerSettings().allowedMediaDomains);
    if (!status.ok()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse request: {}", status.message());
        return status;
    }

    ov::AnyMap streamerConfig;
    if (omniExecutionContext->apiHandler->isStream()) {
        if ((omniExecutionContext->apiHandler->getOutputParser() != nullptr &&
                omniExecutionContext->apiHandler->getOutputParser()->requiresStreamingWithSpecialTokens()) ||
            !omniExecutionContext->apiHandler->getRequest().skipSpecialTokens) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        const bool audioRequested = omniExecutionContext->apiHandler->getRequest().audioOutputRequested;
        auto ovmsCallback = [& ctx = *omniExecutionContext, audioRequested](rapidjson::Document delta, bool isLast) -> ov::genai::StreamingStatus {
            if (ctx.clientDisconnected.load()) {
                ctx.deltaChannel.signalComplete();
                return ov::genai::StreamingStatus::CANCEL;
            }
            // When audio output is requested, don't signal channel complete on text end.
            // Audio chunks will follow via speech_streamer, channel is completed by executor.
            ctx.deltaChannel.push(std::move(delta), isLast && !audioRequested);
            return ov::genai::StreamingStatus::RUNNING;
        };
        omniExecutionContext->textStreamer = std::make_shared<OVMSTextStreamer>(
            getProperties()->tokenizer,
            omniExecutionContext->apiHandler->getOutputParser(),
            omniExecutionContext->apiHandler->areToolsAvailable(),
            std::move(ovmsCallback),
            streamerConfig);
    } else {
        if ((omniExecutionContext->apiHandler->getOutputParser() != nullptr &&
                omniExecutionContext->apiHandler->getOutputParser()->requiresStreamingWithSpecialTokens()) ||
            !omniExecutionContext->apiHandler->getRequest().skipSpecialTokens) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        auto unaryCallback = [& ctx = *omniExecutionContext](rapidjson::Document delta, bool /*isLast*/) -> ov::genai::StreamingStatus {
            if (ctx.clientDisconnected.load()) {
                return ov::genai::StreamingStatus::CANCEL;
            }
            if (delta.HasMember("delta") && delta["delta"].IsObject() &&
                delta["delta"].HasMember("content") && delta["delta"]["content"].IsString()) {
                ctx.accumulatedUnaryText += delta["delta"]["content"].GetString();
            }
            return ov::genai::StreamingStatus::RUNNING;
        };
        omniExecutionContext->textStreamer = std::make_shared<OVMSTextStreamer>(
            getProperties()->tokenizer,
            nullptr,
            false,
            std::move(unaryCallback),
            streamerConfig);
    }
    GenerationConfigBuilder configBuilder(getProperties()->baseGenerationConfig,
        getProperties()->toolParserName,
        getProperties()->enableToolGuidedGeneration,
        getProperties()->decodingMethod);
    auto inputRequestResult = omniExecutionContext->apiHandler->extractInputRequest(configBuilder);
    if (!inputRequestResult.ok()) {
        return inputRequestResult.status();
    }
    omniExecutionContext->inputRequest = std::move(*inputRequestResult);

    // Copy audio output config from parsed request to execution context
    const auto& req = omniExecutionContext->apiHandler->getRequest();
    omniExecutionContext->audioOutputRequested = req.audioOutputRequested;
    omniExecutionContext->audioVoice = req.audioVoice;
    omniExecutionContext->audioFormat = req.audioFormat;

    return absl::OkStatus();
}

absl::Status OmniModelLegacyServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto omniExecutionContext = std::static_pointer_cast<OmniModelLegacyServableExecutionContext>(executionContext);
    std::weak_ptr<OmniModelLegacyServableExecutionContext> weakContext = omniExecutionContext;
    omniExecutionContext->payload.client->registerDisconnectionCallback([weakContext]() {
        if (auto context = weakContext.lock()) {
            context->signalDisconnection();
        }
    });
    if (omniExecutionContext->payload.client->isDisconnected()) {
        omniExecutionContext->signalDisconnection();
        return absl::CancelledError();
    }
    properties->legacyExecutor->addRequest(omniExecutionContext);
    return absl::OkStatus();
}

absl::Status OmniModelLegacyServable::readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto omniExecutionContext = std::static_pointer_cast<OmniModelLegacyServableExecutionContext>(executionContext);
    if (omniExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    omniExecutionContext->finished.wait();
    if (!omniExecutionContext->success) {
        return absl::InvalidArgumentError("Request processing failed, check its correctness.");
    }
    return absl::OkStatus();
}

absl::Status OmniModelLegacyServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto omniExecutionContext = std::static_pointer_cast<OmniModelLegacyServableExecutionContext>(executionContext);
    if (omniExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }

    const std::string& completeText = omniExecutionContext->accumulatedUnaryText;
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(
        omniExecutionContext->results, completeText);

    // If audio output was requested and waveforms are available, inject audio field into response
    if (omniExecutionContext->audioOutputRequested && !omniExecutionContext->results.speech_result.waveforms.empty()) {
        const auto& waveform = omniExecutionContext->results.speech_result.waveforms[0];
        const size_t sampleCount = waveform.get_size();
        const float* pcmData = waveform.data<const float>();

        std::string audioBase64;
        if (omniExecutionContext->audioFormat == "wav") {
            constexpr uint32_t OMNI_SAMPLE_RATE = 24000;
            constexpr uint16_t BITS_PER_SAMPLE = 32;
            void* wavData = nullptr;
            size_t wavSize = 0;
            ovms::audio_utils::prepareAudioOutput(&wavData, wavSize, OMNI_SAMPLE_RATE, BITS_PER_SAMPLE, sampleCount, pcmData);
            audioBase64 = absl::Base64Escape(std::string_view(reinterpret_cast<const char*>(wavData), wavSize));
            free(wavData);
        } else {
            // pcm16: convert float32 to int16
            std::vector<int16_t> pcm16(sampleCount);
            for (size_t i = 0; i < sampleCount; i++) {
                float sample = pcmData[i];
                if (sample > 1.0f)
                    sample = 1.0f;
                if (sample < -1.0f)
                    sample = -1.0f;
                pcm16[i] = static_cast<int16_t>(sample * 32767.0f);
            }
            audioBase64 = absl::Base64Escape(std::string_view(reinterpret_cast<const char*>(pcm16.data()), pcm16.size() * sizeof(int16_t)));
        }

        // Parse response JSON and inject audio field
        rapidjson::Document responseDoc;
        responseDoc.Parse(executionContext->response.c_str());
        if (!responseDoc.HasParseError()) {
            auto& alloc = responseDoc.GetAllocator();
            if (responseDoc.HasMember("choices") && responseDoc["choices"].IsArray() && responseDoc["choices"].Size() > 0) {
                // Chat Completions format: inject into choices[0].message.audio
                auto& message = responseDoc["choices"][0]["message"];
                rapidjson::Value audioObj(rapidjson::kObjectType);
                audioObj.AddMember("data", rapidjson::Value(audioBase64.c_str(), alloc), alloc);
                std::string transcript = message.HasMember("content") && message["content"].IsString() ? message["content"].GetString() : "";
                audioObj.AddMember("transcript", rapidjson::Value(transcript.c_str(), alloc), alloc);
                message.AddMember("audio", audioObj, alloc);
            } else if (responseDoc.HasMember("output") && responseDoc["output"].IsArray()) {
                // Responses API format: append output_audio content part to the message output item
                auto& output = responseDoc["output"];
                for (rapidjson::SizeType i = 0; i < output.Size(); i++) {
                    if (output[i].HasMember("type") && output[i]["type"].IsString() &&
                        std::string(output[i]["type"].GetString()) == "message" &&
                        output[i].HasMember("content") && output[i]["content"].IsArray()) {
                        rapidjson::Value audioPart(rapidjson::kObjectType);
                        audioPart.AddMember("type", rapidjson::Value("output_audio", alloc), alloc);
                        audioPart.AddMember("data", rapidjson::Value(audioBase64.c_str(), alloc), alloc);
                        output[i]["content"].PushBack(audioPart, alloc);
                        break;
                    }
                }
            }

            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            responseDoc.Accept(writer);
            executionContext->response = buffer.GetString();
        }
    }

    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status OmniModelLegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    executionContext->deltaChannel.waitForData();
    return absl::OkStatus();
}

absl::Status OmniModelLegacyServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto omniExecutionContext = std::static_pointer_cast<OmniModelLegacyServableExecutionContext>(executionContext);
    if (omniExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::vector<rapidjson::Document> deltas = executionContext->deltaChannel.drain();
    const bool isFinishing = executionContext->deltaChannel.complete();
    if (!isFinishing) {
        if (deltas.size() > 0 || executionContext->apiHandler->getEndpoint() == Endpoint::RESPONSES) {
            for (auto& delta : deltas) {
                if (executionContext->apiHandler->isVerboseResponse() &&
                    delta.HasMember("delta") && delta["delta"].IsObject() &&
                    delta["delta"].HasMember("content") && delta["delta"]["content"].IsString()) {
                    executionContext->apiHandler->appendVerboseRawText(delta["delta"]["content"].GetString());
                }
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(delta), ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", serialized);
                }
            }
            if (deltas.empty()) {
                if (!executionContext->lifecyclePrimed) {
                    std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                        rapidjson::Document{}, ov::genai::GenerationFinishReason::NONE);
                    if (!serialized.empty()) {
                        executionContext->response = wrapTextInServerSideEventMessage(serialized);
                        executionContext->lifecyclePrimed = true;
                    }
                }
            }
        }
        executionContext->sendLoopbackSignal = true;
    } else {
        omniExecutionContext->finished.wait();
        if (!omniExecutionContext->success) {
            return absl::InvalidArgumentError("Request processing failed, check its correctness.");
        }
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        for (auto& d : executionContext->deltaChannel.drain()) {
            deltas.push_back(std::move(d));
        }
        if (omniExecutionContext->results.finish_reasons.empty()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Missing finish reason in legacy omni streaming generation result, defaulting to STOP");
        }
        ov::genai::GenerationFinishReason finishReason = omniExecutionContext->results.finish_reasons.empty() ? ov::genai::GenerationFinishReason::STOP : omniExecutionContext->results.finish_reasons[0];
        executionContext->apiHandler->setPromptTokensUsage(omniExecutionContext->results.perf_metrics.get_num_input_tokens());
        executionContext->apiHandler->setCompletionTokensUsage(omniExecutionContext->results.perf_metrics.get_num_generated_tokens());
        if (!deltas.empty()) {
            for (size_t i = 0; i < deltas.size(); ++i) {
                const bool isLast = (i == deltas.size() - 1);
                if (executionContext->apiHandler->isVerboseResponse() &&
                    deltas[i].HasMember("delta") && deltas[i]["delta"].IsObject() &&
                    deltas[i]["delta"].HasMember("content") && deltas[i]["delta"]["content"].IsString()) {
                    executionContext->apiHandler->appendVerboseRawText(deltas[i]["delta"]["content"].GetString());
                }
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(deltas[i]),
                    isLast ? finishReason : ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                }
            }
        } else {
            std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                rapidjson::Document{}, finishReason);
            if (!serialized.empty()) {
                executionContext->response += wrapTextInServerSideEventMessage(serialized);
            }
        }
        if (executionContext->apiHandler->getStreamOptions().includeUsage)
            executionContext->response += wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingUsageChunk());

        // Emit response.audio.done if audio was streamed (Responses API only)
        if (omniExecutionContext->audioOutputRequested &&
            executionContext->apiHandler->getEndpoint() == Endpoint::RESPONSES) {
            rapidjson::StringBuffer audioDoneBuf;
            rapidjson::Writer<rapidjson::StringBuffer> audioDoneWriter(audioDoneBuf);
            audioDoneWriter.StartObject();
            audioDoneWriter.Key("type");
            audioDoneWriter.String("response.audio.done");
            audioDoneWriter.Key("sequence_number");
            audioDoneWriter.Uint64(0);
            audioDoneWriter.EndObject();
            executionContext->response += wrapTextInServerSideEventMessage(audioDoneBuf.GetString());
        }

        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", executionContext->response);
        executionContext->sendLoopbackSignal = false;
    }
    return absl::OkStatus();
}

}  // namespace ovms
