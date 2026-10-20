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
#include <chrono>
#include <exception>
#include <iterator>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_istreamwrapper.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "src/audio/speech_to_text/stt_pipeline_handle.hpp"
#include "src/logging.hpp"

namespace ovms {

namespace {

std::string escapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (character < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped += hex[character >> 4];
                escaped += hex[character & 0x0f];
            } else {
                escaped += static_cast<char>(character);
            }
        }
    }
    return escaped;
}

std::string transcriptMessage(const std::string& type, const std::string& text) {
    return "{\"type\":\"" + escapeJson(type) + "\",\"text\":\"" + escapeJson(text) + "\"}";
}

constexpr char OvmsTranscriptChannelLabel[] = "transcript";
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

std::string errorEvent(const std::string& type, const std::string& message) {
    return openAiEvent("error", [&](auto& writer) {
        writer.Key("error");
        writer.StartObject();
        writer.Key("type");
        writer.String(type.c_str());
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
        return offerSdp;
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

WebRtcSessionController::Session::Session(rtc::Configuration configuration, Protocol protocol, std::string id) :
    protocol(protocol),
    id(std::move(id)),
    peer(std::move(configuration)),
    codec(OpusAudioCodec::SampleRate, 2) {}

void WebRtcSessionController::Session::sendUserDelta(const std::string& itemId, const std::string& text) {
    sendTranscript(protocol == Protocol::OpenAiRealtime ? transcriptionDeltaEvent(itemId, text) : transcriptMessage("user_delta", text));
}

void WebRtcSessionController::Session::sendUserFinal(const std::string& itemId, const std::string& text) {
    sendTranscript(protocol == Protocol::OpenAiRealtime ? transcriptionCompletedEvent(itemId, text) : transcriptMessage("user_final", text));
}

void WebRtcSessionController::Session::sendUserError(const std::string& text) {
    sendTranscript(protocol == Protocol::OpenAiRealtime ? errorEvent("server_error", text) : transcriptMessage("user_error", text));
}

void WebRtcSessionController::Session::attachDataChannel(std::shared_ptr<rtc::DataChannel> channel) {
    const char* expectedLabel = protocol == Protocol::OpenAiRealtime ? OpenAiEventsChannelLabel : OvmsTranscriptChannelLabel;
    if (channel->label() != expectedLabel) {
        SPDLOG_LOGGER_DEBUG(webrtc_logger, "Ignoring data channel with label '{}', expected '{}'", channel->label(), expectedLabel);
        return;
    }
    if (protocol == Protocol::OpenAiRealtime) {
        const std::weak_ptr<Session> weakSession = weak_from_this();
        channel->onMessage([weakSession](rtc::message_variant message) {
            if (!std::holds_alternative<std::string>(message))
                return;
            if (const auto session = weakSession.lock())
                session->handleClientEvent(std::get<std::string>(message));
        });
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        transcriptChannel = std::move(channel);
    }
    if (protocol == Protocol::OpenAiRealtime)
        sendTranscript(sessionCreatedEvent(id));
}

void WebRtcSessionController::Session::handleClientEvent(const std::string& message) {
    rapidjson::Document event;
    event.Parse(message.c_str(), message.size());
    if (event.HasParseError() || !event.IsObject()) {
        sendTranscript(errorEvent("invalid_request_error", "Client event is not a valid JSON object"));
        return;
    }
    const auto type = event.FindMember("type");
    if (type == event.MemberEnd() || !type->value.IsString()) {
        sendTranscript(errorEvent("invalid_request_error", "Client event is missing the 'type' field"));
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

void WebRtcSessionController::Session::startVoxtralStream() {
    std::string itemId = nextItemId();
    const std::weak_ptr<Session> weakSession = weak_from_this();
    auto stream = voxtralAdapter->createStream(
        [weakSession, itemId](const std::string& text) {
            if (const auto activeSession = weakSession.lock())
                activeSession->sendUserDelta(itemId, text);
        },
        [weakSession, itemId](const std::string& text) {
            if (const auto activeSession = weakSession.lock())
                activeSession->sendUserFinal(itemId, text);
        },
        [weakSession](std::exception_ptr error) {
            try {
                if (error)
                    std::rethrow_exception(error);
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC Voxtral transcription failed: {}", e.what());
                if (const auto activeSession = weakSession.lock())
                    activeSession->sendUserError(e.what());
            } catch (...) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC Voxtral transcription failed with an unknown error");
            }
        });
    std::vector<std::shared_ptr<VoxtralAudioAdapter::Stream>> finishedStreams;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto firstFinished = std::stable_partition(committedVoxtralStreams.begin(), committedVoxtralStreams.end(),
            [](const auto& committed) { return !committed->isCompleted(); });
        std::move(firstFinished, committedVoxtralStreams.end(), std::back_inserter(finishedStreams));
        committedVoxtralStreams.erase(firstFinished, committedVoxtralStreams.end());
        if (voxtralStream)
            committedVoxtralStreams.push_back(std::move(voxtralStream));
        voxtralStream = std::move(stream);
        currentItemId = std::move(itemId);
    }
    // Joining completed workers happens outside the session lock.
    finishedStreams.clear();
}

std::shared_ptr<VoxtralAudioAdapter::Stream> WebRtcSessionController::Session::currentVoxtralStream() {
    std::lock_guard<std::mutex> lock(mutex);
    return voxtralStream;
}

void WebRtcSessionController::Session::commitAudio() {
    if (sttAdapter) {
        std::optional<AudioChunk> utterance;
        {
            std::lock_guard<std::mutex> lock(mutex);
            utterance = utteranceBuffer.flush();
        }
        if (!utterance) {
            sendTranscript(errorEvent("invalid_request_error", "Error committing input audio buffer: buffer is empty"));
            return;
        }
        transcribeUtterance(std::move(*utterance));
        return;
    }
    if (!voxtralAdapter) {
        sendTranscript(errorEvent("invalid_request_error", "input_audio_buffer.commit requires a speech-to-text model"));
        return;
    }
    std::shared_ptr<VoxtralAudioAdapter::Stream> committedStream;
    std::string committedItemId;
    {
        std::lock_guard<std::mutex> lock(mutex);
        committedStream = voxtralStream;
        committedItemId = currentItemId;
    }
    sendTranscript(audioCommittedEvent(committedItemId));
    // New audio goes to a fresh item while the committed stream drains and emits its completed event.
    startVoxtralStream();
    if (committedStream)
        committedStream->finish();
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
    if (protocol == Protocol::OpenAiRealtime)
        sendTranscript(audioCommittedEvent(itemId));
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
    peer.close();
    std::vector<std::shared_ptr<VoxtralAudioAdapter::Stream>> streams;
    {
        std::lock_guard<std::mutex> lock(mutex);
        streams = std::move(committedVoxtralStreams);
        committedVoxtralStreams.clear();
        if (voxtralStream)
            streams.push_back(voxtralStream);
    }
    for (const auto& stream : streams)
        stream->close();
}

bool WebRtcSessionController::Session::isDisconnected() const {
    const auto state = peer.state();
    return state == rtc::PeerConnection::State::Disconnected ||
           state == rtc::PeerConnection::State::Failed ||
           state == rtc::PeerConnection::State::Closed;
}

void WebRtcSessionController::Session::sendTranscript(const std::string& message) {
    std::lock_guard<std::mutex> sendLock(transcriptSendMutex);
    std::shared_ptr<rtc::DataChannel> channel;
    {
        std::lock_guard<std::mutex> lock(mutex);
        channel = transcriptChannel;
    }
    if (!channel || !channel->isOpen()) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Transcript channel not open, dropping message: {}", message);
        return;
    }
    SPDLOG_LOGGER_DEBUG(webrtc_logger, "Sending transcript message: {}", message);
    channel->send(message);
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

bool WebRtcSessionController::isRealtimeSpeechModel(const std::string& modelPath) {
    const std::string configPath = modelPath + "/config.json";
    std::ifstream input(configPath);
    if (!input.is_open())
        throw std::runtime_error("Cannot open speech model configuration: " + configPath);
    rapidjson::IStreamWrapper stream(input);
    rapidjson::Document config;
    config.ParseStream(stream);
    if (config.HasParseError() || !config.IsObject())
        throw std::runtime_error("Invalid speech model configuration: " + configPath);
    const auto modelType = config.FindMember("model_type");
    if (modelType == config.MemberEnd() || !modelType->value.IsString())
        throw std::runtime_error("Missing or invalid model_type in speech model configuration: " + configPath);
    return std::string(modelType->value.GetString()) == "voxtral_realtime";
}

bool WebRtcSessionController::createSession(const std::string& offerSdp, const std::string& offerType,
    const SttPipelineHandle& speechModel, OfferResult& result) {
    return negotiateSession(offerSdp, offerType, Protocol::Ovms, speechModel, result);
}

bool WebRtcSessionController::createRealtimeCall(const std::string& offerSdp, const SttPipelineHandle& speechModel,
    OfferResult& result) {
    // Realtime call clients never close sessions explicitly, so reclaim dead ones before admitting new calls.
    removeDisconnectedRealtimeCalls();
    return negotiateSession(offerSdp, "offer", Protocol::OpenAiRealtime, speechModel, result);
}

void WebRtcSessionController::removeDisconnectedRealtimeCalls() {
    std::vector<std::pair<std::string, std::shared_ptr<Session>>> disconnected;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            if (it->second->protocol == Protocol::OpenAiRealtime && it->second->isDisconnected()) {
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

bool WebRtcSessionController::negotiateSession(const std::string& offerSdp, const std::string& offerType, Protocol protocol,
    const SttPipelineHandle& speechModel, OfferResult& result) {
    if (offerSdp.empty() || offerType != "offer") {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, invalid offer, type: {}", offerType);
        return false;
    }

    std::shared_ptr<SttAudioAdapter> sttAdapter;
    std::shared_ptr<VoxtralAudioAdapter> voxtralAdapter;
    if (speechModel.pipeline) {
        try {
            if (isRealtimeSpeechModel(speechModel.modelsPath)) {
                voxtralAdapter = std::make_shared<VoxtralAudioAdapter>(speechModel.pipeline);
            } else {
                sttAdapter = std::make_shared<SttAudioAdapter>(speechModel.pipeline, speechModel.pipelineMutex);
            }
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not use speech model at {} for WebRTC: {}", speechModel.modelsPath, e.what());
            return false;
        }
    }

    std::string sessionId;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sessions_.size() >= maxSessions_) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached: {}", maxSessions_);
            return false;
        }
        sessionId = std::to_string(nextSessionId_++);
    }

    SPDLOG_LOGGER_INFO(webrtc_logger, "Creating WebRTC session {} from browser offer, protocol: {}", sessionId,
        protocol == Protocol::OpenAiRealtime ? "openai-realtime" : "ovms");
    rtc::Configuration configuration;
    // One fixed POC port lets a UDP relay forward browser media to the container.
    configuration.portRangeBegin = 52000;
    configuration.portRangeEnd = 52000;
    auto session = std::make_shared<Session>(std::move(configuration), protocol, sessionId);
    session->peer.onStateChange([session](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Disconnected ||
            state == rtc::PeerConnection::State::Failed || state == rtc::PeerConnection::State::Closed) {
            if (const auto stream = session->currentVoxtralStream())
                stream->finish();
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
    if (protocol == Protocol::OpenAiRealtime) {
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
    }
    if (voxtralAdapter) {
        const std::weak_ptr<Session> weakSession = session;
        session->voxtralAdapter = voxtralAdapter;
        session->startVoxtralStream();
        session->peer.onAudioFrame([weakSession](rtc::binary data, rtc::FrameInfo info) {
            const auto currentSession = weakSession.lock();
            if (!currentSession)
                return;
            const auto stream = currentSession->currentVoxtralStream();
            try {
                stream->push(currentSession->decodeAudioFrame(data, info));
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to process WebRTC Voxtral audio: {}", e.what());
                currentSession->sendUserError(e.what());
                stream->finish();
            }
        });
    } else if (sttAdapter) {
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
    try {
        // Reuse the browser's single sendrecv m-line (via onTrack) instead of
        // calling addAudioTrack(), which would add a second, unmatched m-line
        // and make libdatachannel emit a renegotiation offer instead of an answer.
        const auto opusPreferredOffer = preferOpusAudioCodec(offerSdp);
        session->peer.setRemoteDescription(opusPreferredOffer, offerType);
        session->peer.createAnswer();
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to negotiate WebRTC session: {}", e.what());
        session->shutdown();
        return false;
    }

    std::unique_lock<std::mutex> lock(session->mutex);
    if (!session->descriptionReady.wait_for(lock, std::chrono::seconds(5), [&session] {
            return !session->localSdp.empty();
        })) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Timed out waiting for local SDP answer");
        lock.unlock();
        session->shutdown();
        return false;
    }

    result.type = session->localType;
    result.sdp = session->localSdp;
    result.candidates = session->localCandidates;
    if (protocol == Protocol::OpenAiRealtime) {
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
    }
    lock.unlock();

    std::lock_guard<std::mutex> sessionsLock(mutex_);
    if (sessions_.size() >= maxSessions_) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached after negotiation: {}", maxSessions_);
        session->shutdown();
        return false;
    }
    result.sessionId = sessionId;
    SPDLOG_LOGGER_INFO(webrtc_logger, "WebRTC session {} created, answer type: {}, local candidates: {}", result.sessionId, result.type, result.candidates.size());
    sessions_.emplace(result.sessionId, std::move(session));
    return true;
}

bool WebRtcSessionController::addCandidate(const std::string& sessionId, const std::string& candidate, const std::string& mid) {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(sessionId);
        if (it == sessions_.end()) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Cannot add ICE candidate, unknown session: {}", sessionId);
            return false;
        }
        session = it->second;
    }
    if (candidate.empty() || mid.empty()) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected empty ICE candidate for session: {}", sessionId);
        return false;
    }
    SPDLOG_LOGGER_DEBUG(webrtc_logger, "Adding remote ICE candidate for session: {}", sessionId);
    session->peer.addRemoteCandidate(candidate, mid);
    SPDLOG_LOGGER_DEBUG(webrtc_logger,
        "Accepted remote ICE candidate for session {}: mid={}, candidate={}", sessionId, mid, candidate);
    return true;
}

bool WebRtcSessionController::getCandidates(const std::string& sessionId, std::vector<Candidate>& candidates) const {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(sessionId);
        if (it == sessions_.end()) {
            return false;
        }
        session = it->second;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    candidates = session->localCandidates;
    return true;
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
