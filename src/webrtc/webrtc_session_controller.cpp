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
#include "webrtc_session_controller.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "src/audio/speech_to_text/stt_pipeline_handle.hpp"
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
#include "src/audio/speech_to_text/s2t_servable.hpp"
#endif
#include "src/logging.hpp"

namespace ovms {

namespace {

constexpr char OpenAiEventsChannelLabel[] = "oai-events";

std::string nextEventId() {
    static std::atomic<uint64_t> counter{0};
    return "event_" + std::to_string(++counter);
}

// Builds an OpenAI Realtime API server event: {"type": ..., "event_id": ..., <fields>}.
template <typename WriteFields>
std::string openAiEvent(const char* type, WriteFields&& writeFields) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("type");
    writer.String(type);
    writer.Key("event_id");
    writer.String(nextEventId().c_str());
    writeFields(writer);
    writer.EndObject();
    return buffer.GetString();
}

std::string sessionCreatedEvent(const std::string& sessionId) {
    return openAiEvent("session.created", [&sessionId](auto& writer) {
        writer.Key("session");
        writer.StartObject();
        writer.Key("id");
        writer.String(sessionId.c_str());
        writer.Key("object");
        writer.String("realtime.transcription_session");
        writer.Key("type");
        writer.String("transcription");
        writer.EndObject();
    });
}

std::string audioCommittedEvent(const std::string& itemId) {
    return openAiEvent("input_audio_buffer.committed", [&itemId](auto& writer) {
        writer.Key("previous_item_id");
        writer.Null();
        writer.Key("item_id");
        writer.String(itemId.c_str());
    });
}

std::string transcriptionDeltaEvent(const std::string& itemId, const std::string& delta) {
    return openAiEvent("conversation.item.input_audio_transcription.delta", [&](auto& writer) {
        writer.Key("item_id");
        writer.String(itemId.c_str());
        writer.Key("content_index");
        writer.Int(0);
        writer.Key("delta");
        writer.String(delta.c_str(), static_cast<rapidjson::SizeType>(delta.size()));
    });
}

std::string transcriptionCompletedEvent(const std::string& itemId, const std::string& transcript) {
    return openAiEvent("conversation.item.input_audio_transcription.completed", [&](auto& writer) {
        writer.Key("item_id");
        writer.String(itemId.c_str());
        writer.Key("content_index");
        writer.Int(0);
        writer.Key("transcript");
        writer.String(transcript.c_str(), static_cast<rapidjson::SizeType>(transcript.size()));
    });
}

std::string errorEvent(const std::string& type, const std::string& message, const std::string& itemId = {}) {
    return openAiEvent("error", [&](auto& writer) {
        writer.Key("error");
        writer.StartObject();
        writer.Key("type");
        writer.String(type.c_str());
        if (!itemId.empty()) {
            writer.Key("item_id");
            writer.String(itemId.c_str());
        }
        writer.Key("message");
        writer.String(message.c_str(), static_cast<rapidjson::SizeType>(message.size()));
        writer.EndObject();
    });
}

bool isIpv4Literal(const std::string& address) {
    size_t octets = 0;
    size_t position = 0;
    while (position <= address.size()) {
        const size_t end = std::min(address.find('.', position), address.size());
        const std::string octet = address.substr(position, end - position);
        if (octet.empty() || octet.size() > 3 || !std::all_of(octet.begin(), octet.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
            std::stoi(octet) > 255)
            return false;
        ++octets;
        position = end + 1;
    }
    return octets == 4;
}

bool isIpLiteral(const std::string& address) {
    if (address.find(':') == std::string::npos)
        return isIpv4Literal(address);
    const size_t colons = std::count(address.begin(), address.end(), ':');
    return colons >= 2 && address.size() <= 45 &&
           std::all_of(address.begin(), address.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == ':' || c == '.';
           });
}

// Adds an announced copy right after every host candidate line, keeping the original for local clients.
std::string addAnnouncedSdpCandidates(const std::string& sdp, const std::string& announcedIp) {
    if (announcedIp.empty())
        return sdp;
    std::istringstream input(sdp);
    std::string output;
    output.reserve(sdp.size() * 2);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        output += line;
        output += "\r\n";
        if (line.rfind("a=candidate:", 0) == 0) {
            const std::string announced = ovms::WebRtcSessionController::announceHostCandidate(line, announcedIp);
            if (!announced.empty()) {
                output += announced;
                output += "\r\n";
            }
        }
    }
    return output;
}

std::string preferOpusAudioCodec(const std::string& offerSdp) {
    std::istringstream input(offerSdp);
    std::vector<std::string> lines;
    std::string line;
    std::string opusPayloadType;
    size_t audioSectionStart = 0;
    bool inAudioSection = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("m=audio ", 0) == 0) {
            audioSectionStart = lines.size();
            inAudioSection = true;
        } else if (line.rfind("m=", 0) == 0) {
            inAudioSection = false;
        } else if (inAudioSection && line.rfind("a=rtpmap:", 0) == 0 && line.find(" opus/") != std::string::npos) {
            const size_t payloadStart = std::string("a=rtpmap:").size();
            const size_t payloadEnd = line.find(' ', payloadStart);
            if (payloadEnd != std::string::npos) {
                opusPayloadType = line.substr(payloadStart, payloadEnd - payloadStart);
            }
        }
        lines.push_back(line);
    }

    if (opusPayloadType.empty() || audioSectionStart >= lines.size()) {
        throw std::invalid_argument("The SDP offer must contain an Opus audio codec");
    }
    std::istringstream audioLine(lines[audioSectionStart]);
    std::vector<std::string> fields;
    std::string field;
    while (audioLine >> field) {
        fields.push_back(field);
    }
    if (fields.size() < 4) {
        return offerSdp;
    }
    std::ostringstream rewrittenAudioLine;
    rewrittenAudioLine << fields[0] << ' ' << fields[1] << ' ' << fields[2];
    rewrittenAudioLine << ' ' << opusPayloadType;
    lines[audioSectionStart] = rewrittenAudioLine.str();

    std::ostringstream output;
    bool outputInAudioSection = false;
    for (const auto& rewrittenLine : lines) {
        if (rewrittenLine.rfind("m=audio ", 0) == 0) {
            outputInAudioSection = true;
        } else if (rewrittenLine.rfind("m=", 0) == 0) {
            outputInAudioSection = false;
        }
        if (outputInAudioSection &&
            (rewrittenLine.rfind("a=rtpmap:", 0) == 0 ||
                rewrittenLine.rfind("a=fmtp:", 0) == 0 ||
                rewrittenLine.rfind("a=rtcp-fb:", 0) == 0)) {
            const size_t payloadStart = rewrittenLine.find(':') + 1;
            const size_t payloadEnd = rewrittenLine.find(' ', payloadStart);
            if (payloadEnd != std::string::npos && rewrittenLine.substr(payloadStart, payloadEnd - payloadStart) != opusPayloadType) {
                continue;
            }
        }
        output << rewrittenLine << "\r\n";
    }
    return output.str();
}

}  // namespace

WebRtcSessionController::Session::Session(rtc::Configuration configuration, std::string id) :
    id(std::move(id)),
    peer(std::move(configuration)),
    codec(OpusAudioCodec::SampleRate, 2),
    outboundWorker(&Session::runOutboundSender, this) {}

WebRtcSessionController::Session::~Session() {
    shutdown();
}

void WebRtcSessionController::Session::sendUserDelta(const std::string& itemId, const std::string& text) {
    enqueueEvent(transcriptionDeltaEvent(itemId, text), true);
}

void WebRtcSessionController::Session::sendUserFinal(const std::string& itemId, const std::string& text) {
    sendEvent(transcriptionCompletedEvent(itemId, text));
}

void WebRtcSessionController::Session::sendUserError(const std::string& text) {
    sendEvent(errorEvent("server_error", text));
}

void WebRtcSessionController::Session::sendUserError(const std::string& itemId, const std::string& text) {
    sendEvent(errorEvent("server_error", text, itemId));
}

void WebRtcSessionController::Session::attachDataChannel(std::shared_ptr<rtc::DataChannel> channel) {
    if (channel->label() != OpenAiEventsChannelLabel) {
        SPDLOG_LOGGER_DEBUG(webrtc_logger, "Ignoring data channel with label '{}', expected '{}'", channel->label(), OpenAiEventsChannelLabel);
        return;
    }
    const std::weak_ptr<Session> weakSession = weak_from_this();
    channel->onMessage([weakSession](rtc::message_variant message) {
        if (!std::holds_alternative<std::string>(message))
            return;
        if (const auto session = weakSession.lock())
            session->handleClientEvent(std::get<std::string>(message));
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        transcriptChannel = std::move(channel);
    }
    sendEvent(sessionCreatedEvent(id));
}

void WebRtcSessionController::Session::handleClientEvent(const std::string& message) {
    rapidjson::Document event;
    event.Parse(message.c_str(), message.size());
    if (event.HasParseError() || !event.IsObject()) {
        sendEvent(errorEvent("invalid_request_error", "Client event is not a valid JSON object"));
        return;
    }
    const auto type = event.FindMember("type");
    if (type == event.MemberEnd() || !type->value.IsString()) {
        sendEvent(errorEvent("invalid_request_error", "Client event is missing the 'type' field"));
        return;
    }
    const std::string eventType = type->value.GetString();
    SPDLOG_LOGGER_DEBUG(webrtc_logger, "Received realtime client event: {}", eventType);
    if (eventType == "input_audio_buffer.commit") {
        commitAudio();
    } else {
        // session.update, input_audio_buffer.clear etc. do not change streaming transcription.
        SPDLOG_LOGGER_DEBUG(webrtc_logger, "Ignoring unsupported realtime client event: {}", eventType);
    }
}

std::string WebRtcSessionController::Session::nextItemId() {
    std::lock_guard<std::mutex> lock(mutex);
    return "item_" + id + "_" + std::to_string(++itemCounter);
}

#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
void WebRtcSessionController::Session::startVoxtralSession(const SttPipelineHandle& speechModel) {
    const std::weak_ptr<Session> weakSession = weak_from_this();
    streamingSession = std::make_shared<StreamingS2TSession>(speechModel.servable,
        [weakSession](const StreamingS2TEvent& event) {
            if (const auto activeSession = weakSession.lock()) {
                switch (event.type) {
                case StreamingS2TEvent::Type::Delta:
                    activeSession->sendUserDelta(event.utteranceId, event.text);
                    break;
                case StreamingS2TEvent::Type::Final:
                    activeSession->sendUserFinal(event.utteranceId, event.text);
                    break;
                case StreamingS2TEvent::Type::Error:
                    activeSession->sendUserError(event.utteranceId, event.text);
                    break;
                }
            }
        });
    currentItemId = nextItemId();
}
#endif

void WebRtcSessionController::Session::commitAudio() {
    if (sttAdapter) {
        std::optional<AudioChunk> utterance;
        {
            std::lock_guard<std::mutex> lock(mutex);
            utterance = utteranceBuffer.flush();
        }
        if (!utterance) {
            sendEvent(errorEvent("invalid_request_error", "Error committing input audio buffer: buffer is empty"));
            return;
        }
        transcribeUtterance(std::move(*utterance));
        return;
    }
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
    std::shared_ptr<StreamingS2TSession> activeSession;
    std::string committedItemId;
    bool hasAudio = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        committedItemId = currentItemId;
        hasAudio = voxtralHasAudio;
        voxtralHasAudio = false;
        activeSession = streamingSession;
    }
    if (!activeSession) {
        sendEvent(errorEvent("invalid_request_error", "input_audio_buffer.commit requires a speech-to-text model"));
        return;
    }
    if (!hasAudio) {
        sendEvent(errorEvent("invalid_request_error", "Error committing input audio buffer: buffer is empty"));
        return;
    }
    sendEvent(audioCommittedEvent(committedItemId));
    const auto status = activeSession->finish(committedItemId);
    if (!status.ok()) {
        sendUserError(status.ToString());
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        currentItemId = "item_" + id + "_" + std::to_string(++itemCounter);
    }
#else
    sendEvent(errorEvent("invalid_request_error", "input_audio_buffer.commit requires a speech-to-text model"));
#endif
}

void WebRtcSessionController::Session::pushUtteranceAudio(const AudioChunk& frame) {
    std::optional<AudioChunk> utterance;
    {
        std::lock_guard<std::mutex> lock(mutex);
        utterance = utteranceBuffer.push(frame);
    }
    if (utterance)
        transcribeUtterance(std::move(*utterance));
}

void WebRtcSessionController::Session::transcribeUtterance(AudioChunk utterance) {
    const std::string itemId = nextItemId();
    SPDLOG_LOGGER_DEBUG(webrtc_logger, "Transcribing utterance {}: length_ms={}", itemId,
        utterance.samples.size() * 1000 / (static_cast<uint64_t>(utterance.sampleRate) * utterance.channels));
    sendEvent(audioCommittedEvent(itemId));
    const std::weak_ptr<Session> weakSession = weak_from_this();
    sttAdapter->submit(
        std::move(utterance),
        [weakSession, itemId](const std::string& text) {
            if (const auto activeSession = weakSession.lock())
                activeSession->sendUserDelta(itemId, text);
        },
        [weakSession](std::exception_ptr error) {
            try {
                if (error)
                    std::rethrow_exception(error);
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC speech-to-text failed: {}", e.what());
                if (const auto activeSession = weakSession.lock())
                    activeSession->sendUserError(e.what());
            } catch (...) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC speech-to-text failed with an unknown error");
            }
        },
        [weakSession, itemId](const std::string& text) {
            if (const auto activeSession = weakSession.lock())
                activeSession->sendUserFinal(itemId, text);
        });
}

void WebRtcSessionController::Session::shutdown() {
    std::shared_ptr<StreamingS2TSession> activeStreamingSession;
    {
        std::lock_guard<std::mutex> lock(mutex);
        transcriptChannel.reset();
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
        activeStreamingSession = std::move(streamingSession);
#endif
    }
    peer.close();
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
    if (activeStreamingSession)
        activeStreamingSession->cancel();
    activeStreamingSession.reset();
#endif
    {
        std::lock_guard<std::mutex> lock(outboundMutex);
        outboundStopping = true;
        outboundQueue.clear();
        outboundBytes = 0;
    }
    outboundReady.notify_all();
    if (outboundWorker.joinable())
        outboundWorker.join();
}

bool WebRtcSessionController::Session::isDisconnected() const {
    const auto state = peer.state();
    return state == rtc::PeerConnection::State::Disconnected ||
           state == rtc::PeerConnection::State::Failed ||
           state == rtc::PeerConnection::State::Closed;
}

void WebRtcSessionController::Session::sendEvent(const std::string& message) {
    enqueueEvent(message);
}

void WebRtcSessionController::Session::enqueueEvent(std::string message, bool delta) {
    constexpr size_t MaxOutboundEvents = 256;
    constexpr size_t MaxOutboundBytes = 1024 * 1024;
    std::lock_guard<std::mutex> lock(outboundMutex);
    if (outboundStopping)
        return;
    if (message.size() > MaxOutboundBytes) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Dropping oversized realtime event for session {}", id);
        return;
    }
    while ((outboundQueue.size() >= MaxOutboundEvents ||
               message.size() > MaxOutboundBytes - std::min(outboundBytes, MaxOutboundBytes)) &&
        !outboundQueue.empty()) {
        const auto discard = std::find_if(outboundQueue.begin(), outboundQueue.end(),
            [](const auto& event) { return event.second; });
        if (discard == outboundQueue.end()) {
            if (delta) {
                SPDLOG_LOGGER_WARN(webrtc_logger, "Dropping realtime transcript delta: outbound queue is full");
                return;
            }
            SPDLOG_LOGGER_ERROR(webrtc_logger, "Dropping realtime event: outbound queue is full of non-delta events");
            outboundBytes -= outboundQueue.front().first.size();
            outboundQueue.pop_front();
        } else {
            outboundBytes -= discard->first.size();
            outboundQueue.erase(discard);
            SPDLOG_LOGGER_WARN(webrtc_logger, "Dropping queued realtime transcript delta: outbound queue is full");
        }
    }
    outboundBytes += message.size();
    outboundQueue.emplace_back(std::move(message), delta);
    outboundReady.notify_one();
}

void WebRtcSessionController::Session::runOutboundSender() {
    while (true) {
        std::string message;
        {
            std::unique_lock<std::mutex> lock(outboundMutex);
            outboundReady.wait(lock, [this] { return outboundStopping || !outboundQueue.empty(); });
            if (outboundStopping && outboundQueue.empty())
                return;
            message = std::move(outboundQueue.front().first);
            outboundBytes -= message.size();
            outboundQueue.pop_front();
        }
        std::lock_guard<std::mutex> sendLock(transcriptSendMutex);
        std::shared_ptr<rtc::DataChannel> channel;
        {
            std::lock_guard<std::mutex> lock(mutex);
            channel = transcriptChannel;
        }
        if (!channel || !channel->isOpen()) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Transcript channel not open, dropping event");
            continue;
        }
        SPDLOG_LOGGER_DEBUG(webrtc_logger, "Sending transcript message: {}", message);
        try {
            channel->send(message);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to send realtime event for session {}: {}", id, e.what());
        }
    }
}

AudioChunk WebRtcSessionController::Session::decodeAudioFrame(const rtc::binary& data, const rtc::FrameInfo& info) {
    std::vector<uint8_t> encoded(data.size());
    for (size_t index = 0; index < data.size(); ++index)
        encoded[index] = std::to_integer<uint8_t>(data[index]);
    const auto decoded = codec.decode(encoded);
    const uint64_t timestampUs = info.timestampSeconds ? static_cast<uint64_t>(info.timestampSeconds->count() * 1000000.0) : static_cast<uint64_t>((static_cast<uint64_t>(info.timestamp) * 1000000) / OpusAudioCodec::SampleRate);
    return AudioChunk{decoded, OpusAudioCodec::SampleRate, timestampUs, static_cast<uint32_t>(codec.channels())};
}

WebRtcSessionController::WebRtcSessionController(size_t maxSessions, std::string announcedIp) :
    maxSessions_(maxSessions),
    announcedIp_(std::move(announcedIp)) {
    if (!announcedIp_.empty() && !isIpLiteral(announcedIp_)) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Ignoring invalid WebRTC announced IP '{}', expected an IPv4 or IPv6 address", announcedIp_);
        announcedIp_.clear();
    }
    if (!announcedIp_.empty())
        SPDLOG_LOGGER_INFO(webrtc_logger, "WebRTC host ICE candidates will also be announced with address {}", announcedIp_);
}

std::string WebRtcSessionController::announceHostCandidate(const std::string& candidate, const std::string& announcedIp) {
    if (announcedIp.empty())
        return {};
    // candidate:<foundation> <component> <transport> <priority> <address> <port> typ <type> [extensions...]
    std::istringstream stream(candidate);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token)
        tokens.push_back(std::move(token));
    constexpr size_t addressIndex = 4;
    if (tokens.size() < 8 || tokens[0].find("candidate:") == std::string::npos || tokens[6] != "typ" || tokens[7] != "host")
        return {};
    const std::string& address = tokens[addressIndex];
    const bool candidateIsIpv6 = address.find(':') != std::string::npos;
    const bool announcedIsIpv6 = announcedIp.find(':') != std::string::npos;
    if (address == announcedIp || candidateIsIpv6 != announcedIsIpv6 || !isIpLiteral(address))
        return {};
    tokens[addressIndex] = announcedIp;
    std::string rewritten = tokens[0];
    for (size_t i = 1; i < tokens.size(); ++i) {
        rewritten += ' ';
        rewritten += tokens[i];
    }
    return rewritten;
}

#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
bool WebRtcSessionController::isRealtimeSpeechModel(const std::string& modelPath) {
    return SttServable::isRealtimeModel(modelPath);
}
#endif

WebRtcSessionController::CreateCallResult WebRtcSessionController::createRealtimeCall(const std::string& offerSdp, const SttPipelineHandle& speechModel,
    OfferResult& result) {
    // Realtime call clients never close sessions explicitly, so reclaim dead ones before admitting new calls.
    removeDisconnectedRealtimeCalls();
    return negotiateSession(offerSdp, speechModel, result);
}

void WebRtcSessionController::removeDisconnectedRealtimeCalls() {
    std::vector<std::pair<std::string, std::shared_ptr<Session>>> disconnected;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            if (it->second->isDisconnected()) {
                disconnected.emplace_back(it->first, std::move(it->second));
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const auto& [sessionId, session] : disconnected) {
        session->shutdown();
        SPDLOG_LOGGER_INFO(webrtc_logger, "Removed disconnected realtime call {}", sessionId);
    }
}

WebRtcSessionController::CreateCallResult WebRtcSessionController::negotiateSession(const std::string& offerSdp, const SttPipelineHandle& speechModel, OfferResult& result) {
    if (offerSdp.empty()) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected realtime call creation, offer is empty");
        return CreateCallResult::InvalidOffer;
    }

    std::shared_ptr<SttAudioAdapter> sttAdapter;
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
    bool voxtralStreaming = false;
#endif
    if (speechModel.pipeline) {
        try {
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
            if (speechModel.servable && speechModel.servable->streamingBackend) {
                voxtralStreaming = true;
            } else if (isRealtimeSpeechModel(speechModel.modelsPath)) {
                throw std::runtime_error("Voxtral realtime graph resources are unavailable");
            } else {
                sttAdapter = std::make_shared<SttAudioAdapter>(speechModel.pipeline, speechModel.pipelineMutex);
            }
#else
            sttAdapter = std::make_shared<SttAudioAdapter>(speechModel.pipeline, speechModel.pipelineMutex);
#endif
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not use speech model at {} for WebRTC: {}", speechModel.modelsPath, e.what());
            return CreateCallResult::InternalError;
        }
    }

    std::string sessionId;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sessions_.size() >= maxSessions_) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached: {}", maxSessions_);
            return CreateCallResult::CapacityExhausted;
        }
        sessionId = std::to_string(nextSessionId_++);
    }

    SPDLOG_LOGGER_INFO(webrtc_logger, "Creating realtime call {} from browser offer", sessionId);
    rtc::Configuration configuration;
    configuration.portRangeBegin = 52000;
    configuration.portRangeEnd = 52100;
    if (const char* value = std::getenv("OVMS_WEBRTC_PORT_RANGE")) {
        const std::string_view range(value);
        const size_t separator = range.find('-');
        const auto parsePort = [](std::string_view text, uint16_t& port) {
            if (text.empty())
                return false;
            const auto result = std::from_chars(text.data(), text.data() + text.size(), port);
            return result.ec == std::errc{} && result.ptr == text.data() + text.size() && port != 0;
        };
        if (separator == std::string_view::npos ||
            !parsePort(range.substr(0, separator), configuration.portRangeBegin) ||
            !parsePort(range.substr(separator + 1), configuration.portRangeEnd) ||
            configuration.portRangeBegin > configuration.portRangeEnd) {
            SPDLOG_LOGGER_ERROR(webrtc_logger,
                "Invalid OVMS_WEBRTC_PORT_RANGE '{}': expected start-end with 1 <= start <= end <= 65535", value);
            return CreateCallResult::InternalError;
        }
    }
    auto session = std::make_shared<Session>(std::move(configuration), sessionId);
    session->peer.onStateChange([session](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Disconnected ||
            state == rtc::PeerConnection::State::Failed || state == rtc::PeerConnection::State::Closed) {
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
            std::shared_ptr<StreamingS2TSession> streamingSession;
            {
                std::lock_guard<std::mutex> lock(session->mutex);
                streamingSession = session->streamingSession;
            }
            if (streamingSession)
                streamingSession->cancel();
#endif
            return;
        }
        if (state != rtc::PeerConnection::State::Connected) {
            return;
        }
        rtc::Candidate localCandidate;
        rtc::Candidate remoteCandidate;
        if (session->peer.getSelectedCandidatePair(localCandidate, remoteCandidate)) {
            const std::string remoteAddress = remoteCandidate.address().value_or("unknown");
            const uint16_t remotePort = remoteCandidate.port().value_or(0);
            SPDLOG_LOGGER_INFO(webrtc_logger,
                "Selected ICE pair for WebRTC session: local_candidate={}, remote_candidate={}, outbound_destination={}:{}",
                localCandidate.candidate(), remoteCandidate.candidate(), remoteAddress, remotePort);
        } else {
            SPDLOG_LOGGER_WARN(webrtc_logger,
                "WebRTC session connected but selected ICE candidate pair is unavailable");
        }
    });
    session->peer.onLocalDescription([session](const std::string& sdp, const std::string& type) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            session->localSdp = sdp;
            session->localType = type;
        }
        session->descriptionReady.notify_all();
    });
    session->peer.onLocalCandidate([session, announcedIp = announcedIp_](const std::string& candidate, const std::string& mid) {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->localCandidates.push_back({candidate, mid});
        std::string announced = announceHostCandidate(candidate, announcedIp);
        if (!announced.empty()) {
            SPDLOG_LOGGER_DEBUG(webrtc_logger, "Announcing host ICE candidate as: {}", announced);
            session->localCandidates.push_back({std::move(announced), mid});
        }
    });
    session->peer.onDataChannel([weakSession = std::weak_ptr<Session>(session)](std::shared_ptr<rtc::DataChannel> channel) {
        if (const auto activeSession = weakSession.lock())
            activeSession->attachDataChannel(std::move(channel));
    });
    session->peer.onGatheringComplete([weakSession = std::weak_ptr<Session>(session)] {
        const auto activeSession = weakSession.lock();
        if (!activeSession)
            return;
        {
            std::lock_guard<std::mutex> lock(activeSession->mutex);
            activeSession->gatheringComplete = true;
        }
        activeSession->descriptionReady.notify_all();
    });
#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
    if (voxtralStreaming) {
        const std::weak_ptr<Session> weakSession = session;
        try {
            session->startVoxtralSession(speechModel);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not start Voxtral graph for WebRTC: {}", e.what());
            session->shutdown();
            return CreateCallResult::InternalError;
        }
        session->peer.onAudioFrame([weakSession](rtc::binary data, rtc::FrameInfo info) {
            const auto currentSession = weakSession.lock();
            if (!currentSession)
                return;
            try {
                const auto audio = currentSession->decodeAudioFrame(data, info);
                std::shared_ptr<StreamingS2TSession> activeSession;
                std::string itemId;
                {
                    std::lock_guard<std::mutex> lock(currentSession->mutex);
                    activeSession = currentSession->streamingSession;
                    itemId = currentSession->currentItemId;
                    if (activeSession) {
                        const auto status = activeSession->push(itemId, std::move(audio));
                        if (!status.ok())
                            throw std::runtime_error(status.ToString());
                        currentSession->voxtralHasAudio = true;
                    }
                }
                if (!activeSession)
                    throw std::runtime_error("Streaming transcription session is closed");
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to process WebRTC Voxtral audio: {}", e.what());
                currentSession->sendUserError(e.what());
            }
        });
    } else
#endif
    if (sttAdapter) {
        const std::weak_ptr<Session> weakSession = session;
        session->sttAdapter = sttAdapter;
        session->peer.onAudioFrame([weakSession](rtc::binary data, rtc::FrameInfo info) {
            const auto currentSession = weakSession.lock();
            if (!currentSession)
                return;
            try {
                currentSession->pushUtteranceAudio(currentSession->decodeAudioFrame(data, info));
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to process WebRTC speech-to-text audio: {}", e.what());
                currentSession->sendUserError(e.what());
            }
        });
    }
    // Reuse the browser's single sendrecv m-line (via onTrack) instead of
    // calling addAudioTrack(), which would add a second, unmatched m-line
    // and make libdatachannel emit a renegotiation offer instead of an answer.
    std::string opusPreferredOffer;
    try {
        opusPreferredOffer = preferOpusAudioCodec(offerSdp);
    } catch (const std::invalid_argument& e) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected invalid WebRTC offer: {}", e.what());
        session->shutdown();
        return CreateCallResult::InvalidOffer;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to prepare WebRTC offer: {}", e.what());
        session->shutdown();
        return CreateCallResult::InternalError;
    }
    std::optional<rtc::Description> remoteDescription;
    try {
        remoteDescription.emplace(opusPreferredOffer, "offer");
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected invalid WebRTC offer: {}", e.what());
        session->shutdown();
        return CreateCallResult::InvalidOffer;
    }
    try {
        session->peer.setRemoteDescription(*remoteDescription);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to apply WebRTC offer: {}", e.what());
        session->shutdown();
        return CreateCallResult::InternalError;
    }
    try {
        session->peer.createAnswer();
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to create WebRTC answer: {}", e.what());
        session->shutdown();
        return CreateCallResult::InternalError;
    }

    std::unique_lock<std::mutex> lock(session->mutex);
    if (!session->descriptionReady.wait_for(lock, std::chrono::seconds(5), [&session] {
            return !session->localSdp.empty();
        })) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Timed out waiting for local SDP answer");
        lock.unlock();
        session->shutdown();
        return CreateCallResult::InternalError;
    }

    result.type = session->localType;
    result.sdp = session->localSdp;
    result.candidates = session->localCandidates;
    // Realtime call clients do not trickle ICE, so the answer must carry every server candidate.
    if (!session->descriptionReady.wait_for(lock, std::chrono::seconds(5), [&session] {
            return session->gatheringComplete;
        })) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Timed out waiting for ICE gathering, answering with {} candidates",
            session->localCandidates.size());
    }
    lock.unlock();
    result.sdp = addAnnouncedSdpCandidates(session->peer.localDescriptionSdp(), announcedIp_);
    lock.lock();
    result.candidates = session->localCandidates;
    lock.unlock();

    std::lock_guard<std::mutex> sessionsLock(mutex_);
    if (sessions_.size() >= maxSessions_) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached after negotiation: {}", maxSessions_);
        session->shutdown();
        return CreateCallResult::CapacityExhausted;
    }
    result.sessionId = sessionId;
    SPDLOG_LOGGER_INFO(webrtc_logger, "WebRTC session {} created, answer type: {}, local candidates: {}", result.sessionId, result.type, result.candidates.size());
    sessions_.emplace(result.sessionId, std::move(session));
    return CreateCallResult::Success;
}

bool WebRtcSessionController::removeSession(const std::string& sessionId) {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(sessionId);
        if (it == sessions_.end()) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Cannot close unknown session: {}", sessionId);
            return false;
        }
        session = it->second;
        sessions_.erase(it);
    }
    session->shutdown();
    SPDLOG_LOGGER_INFO(webrtc_logger, "WebRTC session {} closed", sessionId);
    return true;
}

size_t WebRtcSessionController::sessionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
}

}  // namespace ovms
