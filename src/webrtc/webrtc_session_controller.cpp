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
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <utility>
#include <vector>

#include <openvino/genai/omni/pipeline.hpp>
#include <openvino/genai/omni/talker.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "src/logging.hpp"
#include "omni_audio_adapter.hpp"

namespace ovms {

namespace {

std::string escapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
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

size_t audioChunkFramesFromEnv() {
    constexpr size_t defaultAudioChunkFrames = 4;
    const char* value = std::getenv("OVMS_WEBRTC_OMNI_AUDIO_CHUNK_FRAMES");
    if (value == nullptr || value[0] == '\0')
        return defaultAudioChunkFrames;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (*end != '\0' || parsed == 0) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Invalid OVMS_WEBRTC_OMNI_AUDIO_CHUNK_FRAMES value '{}', using {}", value, defaultAudioChunkFrames);
        return defaultAudioChunkFrames;
    }
    return static_cast<size_t>(parsed);
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

WebRtcSessionController::Session::Session(rtc::Configuration configuration) :
    peer(std::move(configuration)),
    codec(OpusAudioCodec::SampleRate, 2),
    model(OpusAudioCodec::SampleRate),
    processor(codec, model) {
}

void WebRtcSessionController::Session::sendGeneratedAudio(const AudioChunk& audio, rtc::FrameInfo info, bool complete) {
    pendingOutputSamples.insert(pendingOutputSamples.end(), audio.samples.begin(), audio.samples.end());
    while (pendingOutputSamples.size() >= OpusAudioCodec::FrameSamples || (complete && !pendingOutputSamples.empty())) {
        const size_t channelCount = codec.channels();
        std::vector<float> frame(OpusAudioCodec::FrameSamples * channelCount, 0.0f);
        const size_t samplesToCopy = std::min(OpusAudioCodec::FrameSamples, pendingOutputSamples.size());
        for (size_t index = 0; index < samplesToCopy; ++index) {
            const float sample = pendingOutputSamples.front();
            pendingOutputSamples.pop_front();
            for (size_t channel = 0; channel < channelCount; ++channel)
                frame[index * channelCount + channel] = sample;
        }
        const auto encoded = codec.encode(frame);
        rtc::binary packet(encoded.size());
        for (size_t index = 0; index < encoded.size(); ++index) {
            packet[index] = static_cast<std::byte>(encoded[index]);
        }
        peer.sendAudioFrame(std::move(packet), info);
    }
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

WebRtcSessionController::WebRtcSessionController(size_t maxSessions, std::string omniModelPath,
    std::string sttModelPath, std::string sttDevice) :
    maxSessions_(maxSessions),
    omniModelPath_(std::move(omniModelPath)),
    sttModelPath_(std::move(sttModelPath)),
    sttDevice_(std::move(sttDevice)) {
}

bool WebRtcSessionController::createSession(const std::string& offerSdp, const std::string& offerType, OfferResult& result) {
    if (offerSdp.empty() || offerType != "offer") {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, invalid offer, type: {}", offerType);
        return false;
    }

    std::shared_ptr<OmniAudioAdapter> omniAdapter;
    std::shared_ptr<SttAudioAdapter> sttAdapter;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sessions_.size() >= maxSessions_) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached: {}", maxSessions_);
            return false;
        }
        if (!omniModelPath_.empty() && !omniAdapter_) {
            try {
                const char* configuredDevice = std::getenv("OVMS_WEBRTC_OMNI_DEVICE");
                const std::string device = configuredDevice == nullptr || configuredDevice[0] == '\0' ? "CPU" : configuredDevice;
                const char* configuredTalkerDevice = std::getenv("OVMS_WEBRTC_TALKER_DEVICE");
                const std::string talkerDevice = configuredTalkerDevice == nullptr || configuredTalkerDevice[0] == '\0' ? device : configuredTalkerDevice;
                const size_t audioChunkFrames = audioChunkFramesFromEnv();
                auto vlm = std::make_shared<ov::genai::VLMPipeline>(omniModelPath_, device, ov::AnyMap{});
                auto talker = std::make_shared<ov::genai::Talker>(omniModelPath_, talkerDevice, ov::AnyMap{});
                auto pipeline = std::make_shared<ov::genai::OmniPipeline>(std::move(vlm), std::move(talker));
                omniAdapter_ = std::make_shared<OmniAudioAdapter>(std::move(pipeline), audioChunkFrames);
                SPDLOG_LOGGER_INFO(webrtc_logger, "Initialized WebRTC Omni pipeline from {}: text device={}, talker device={}, audio_chunk_frames={}",
                    omniModelPath_, device, talkerDevice, audioChunkFrames);
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not initialize WebRTC Omni model at {}: {}", omniModelPath_, e.what());
                return false;
            } catch (...) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not initialize WebRTC Omni model at {}", omniModelPath_);
                return false;
            }
        }
        omniAdapter = omniAdapter_;
        if (!sttModelPath_.empty() && !sttAdapter_) {
            try {
                sttAdapter_ = std::make_shared<SttAudioAdapter>(sttModelPath_, sttDevice_);
                SPDLOG_LOGGER_INFO(webrtc_logger, "Initialized WebRTC STT model from {} on {}", sttModelPath_, sttDevice_);
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not initialize WebRTC STT model at {}: {}", sttModelPath_, e.what());
                return false;
            } catch (...) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not initialize WebRTC STT model at {}", sttModelPath_);
                return false;
            }
        }
        sttAdapter = sttAdapter_;
    }

    SPDLOG_LOGGER_INFO(webrtc_logger, "Creating WebRTC session from browser offer");
    rtc::Configuration configuration;
    // One fixed POC port lets a UDP relay forward browser media to the container.
    configuration.portRangeBegin = 52000;
    configuration.portRangeEnd = 52000;
    auto session = std::make_shared<Session>(std::move(configuration));
    session->peer.onStateChange([session](rtc::PeerConnection::State state) {
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
    session->peer.onLocalCandidate([session](const std::string& candidate, const std::string& mid) {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->localCandidates.push_back({candidate, mid});
    });
    session->peer.onDataChannel([session](std::shared_ptr<rtc::DataChannel> channel) {
        if (channel->label() != "transcript")
            return;
        std::lock_guard<std::mutex> lock(session->mutex);
        session->transcriptChannel = std::move(channel);
    });
    if (omniAdapter) {
        session->conversation = omniAdapter->createConversation();
        const std::weak_ptr<Session> weakSession = session;
        session->peer.onAudioFrame([weakSession, omniAdapter, sttAdapter, conversation = session->conversation](rtc::binary data, rtc::FrameInfo info) {
            const auto currentSession = weakSession.lock();
            if (!currentSession)
                return;
            try {
                std::vector<uint8_t> encoded(data.size());
                for (size_t index = 0; index < data.size(); ++index)
                    encoded[index] = std::to_integer<uint8_t>(data[index]);
                const auto decoded = currentSession->codec.decode(encoded);
                const uint64_t timestampUs = info.timestampSeconds ?
                    static_cast<uint64_t>(info.timestampSeconds->count() * 1000000.0) :
                    static_cast<uint64_t>((static_cast<uint64_t>(info.timestamp) * 1000000) / OpusAudioCodec::SampleRate);
                auto utterance = currentSession->utteranceBuffer.push(
                    AudioChunk{decoded, OpusAudioCodec::SampleRate, timestampUs, static_cast<uint32_t>(currentSession->codec.channels())});
                if (!utterance)
                    return;

                float maxVolume = 0.0f;
                for (const float sample : utterance->samples)
                    maxVolume = std::max(maxVolume, std::fabs(sample));
                const size_t frameCount = utterance->samples.size() / std::max<uint32_t>(utterance->channels, 1);
                SPDLOG_LOGGER_INFO(webrtc_logger,
                    "Detected utterance after silence, sending to Omni: length_ms={}, samples={}, channels={}, sample_rate={}, max_volume={:.4f}",
                    frameCount * 1000 / utterance->sampleRate, utterance->samples.size(), utterance->channels,
                    utterance->sampleRate, maxVolume);

                const std::weak_ptr<Session> outputSession = currentSession;
                if (sttAdapter) {
                    try {
                        sttAdapter->submit(*utterance,
                            [outputSession](const std::string& text) {
                                if (const auto activeSession = outputSession.lock())
                                    activeSession->sendTranscript(transcriptMessage("user_delta", text));
                            },
                            [outputSession](std::exception_ptr error) {
                                try {
                                    if (error)
                                        std::rethrow_exception(error);
                                } catch (const std::exception& e) {
                                    SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC STT generation failed: {}", e.what());
                                    if (const auto activeSession = outputSession.lock())
                                        activeSession->sendTranscript(transcriptMessage("user_error", e.what()));
                                } catch (...) {
                                    SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC STT generation failed with an unknown error");
                                }
                            },
                            [outputSession](const std::string& text) {
                                if (const auto activeSession = outputSession.lock())
                                    activeSession->sendTranscript(transcriptMessage("user_final", text));
                            });
                    } catch (const std::exception& e) {
                        SPDLOG_LOGGER_ERROR(webrtc_logger, "Could not queue WebRTC STT request: {}", e.what());
                        currentSession->sendTranscript(transcriptMessage("user_error", e.what()));
                    }
                }
                omniAdapter->submit(conversation, std::move(*utterance),
                    [outputSession, info](AudioChunk audio) {
                        if (const auto activeSession = outputSession.lock())
                            activeSession->sendGeneratedAudio(audio, info, false);
                    },
                    [outputSession](const std::string& token) {
                        if (const auto activeSession = outputSession.lock())
                            activeSession->sendTranscript(transcriptMessage("assistant_delta", token));
                    },
                    [](std::exception_ptr error) {
                        try {
                            if (error)
                                std::rethrow_exception(error);
                        } catch (const std::exception& e) {
                            SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC Omni generation failed: {}", e.what());
                        } catch (...) {
                            SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC Omni generation failed with an unknown error");
                        }
                    },
                    [outputSession, info](const std::string& text) {
                        SPDLOG_LOGGER_INFO(webrtc_logger, "Omni response finished: \"{}\"", text);
                        if (const auto activeSession = outputSession.lock()) {
                            activeSession->sendTranscript(transcriptMessage("assistant_final", text));
                            activeSession->sendGeneratedAudio(AudioChunk{{}, OpusAudioCodec::SampleRate, 0, 1}, info, true);
                        }
                    });
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to buffer WebRTC audio for Omni: {}", e.what());
            }
        });
    } else {
        session->peer.onProcessedAudioFrame(session->processor, [session](rtc::binary data, rtc::FrameInfo info) {
            session->peer.sendAudioFrame(std::move(data), info);
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
        return false;
    }

    std::unique_lock<std::mutex> lock(session->mutex);
    if (!session->descriptionReady.wait_for(lock, std::chrono::seconds(5), [&session] {
            return !session->localSdp.empty();
        })) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Timed out waiting for local SDP answer");
        return false;
    }

    result.type = session->localType;
    result.sdp = session->localSdp;
    result.candidates = session->localCandidates;
    lock.unlock();

    std::lock_guard<std::mutex> sessionsLock(mutex_);
    if (sessions_.size() >= maxSessions_) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached after negotiation: {}", maxSessions_);
        return false;
    }
    result.sessionId = std::to_string(nextSessionId_++);
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
    std::lock_guard<std::mutex> lock(mutex_);
    const bool removed = sessions_.erase(sessionId) != 0;
    if (removed) {
        SPDLOG_LOGGER_INFO(webrtc_logger, "WebRTC session {} closed", sessionId);
    } else {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Cannot close unknown session: {}", sessionId);
    }
    return removed;
}

size_t WebRtcSessionController::sessionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
}

}  // namespace ovms