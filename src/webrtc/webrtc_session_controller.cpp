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

#include <chrono>
#include <exception>
#include <utility>

#include "src/logging.hpp"

namespace ovms {

WebRtcSessionController::Session::Session(rtc::Configuration configuration) :
    peer(std::move(configuration)),
    codec(OpusAudioCodec::SampleRate, OpusAudioCodec::Channels),
    model(OpusAudioCodec::SampleRate, 0.5f, 0.5f),
    processor(codec, model) {
}

WebRtcSessionController::WebRtcSessionController(size_t maxSessions) :
    maxSessions_(maxSessions) {
}

bool WebRtcSessionController::createSession(const std::string& offerSdp, const std::string& offerType, OfferResult& result) {
    if (offerSdp.empty() || offerType != "offer") {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, invalid offer, type: {}", offerType);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sessions_.size() >= maxSessions_) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached: {}", maxSessions_);
            return false;
        }
    }

    SPDLOG_LOGGER_INFO(webrtc_logger, "Creating WebRTC session from browser offer");
    rtc::Configuration configuration;
    // One fixed POC port lets a UDP relay forward browser media to the container.
    configuration.portRangeBegin = 52000;
    configuration.portRangeEnd = 52000;
    auto session = std::make_shared<Session>(std::move(configuration));
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
    session->peer.onProcessedAudioFrame(session->processor, [session](rtc::binary data, rtc::FrameInfo info) {
        session->peer.sendAudioFrame(std::move(data), info);
    });
    try {
        // Reuse the browser's single sendrecv m-line (via onTrack) instead of
        // calling addAudioTrack(), which would add a second, unmatched m-line
        // and make libdatachannel emit a renegotiation offer instead of an answer.
        session->peer.setRemoteDescription(offerSdp, offerType);
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