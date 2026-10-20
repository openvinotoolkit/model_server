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
#pragma once

#include <cstddef>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "webrtc_peer_connection.hpp"
#include "audio_utterance_buffer.hpp"
#include "mock_echo_streaming_audio_model.hpp"
#include "omni_audio_adapter.hpp"
#include "opus_audio_codec.hpp"
#include "stt_audio_adapter.hpp"
#include "streaming_audio_processor.hpp"
#include "voxtral_audio_adapter.hpp"

namespace ovms {

class WebRtcSessionController {
public:
    struct SpeechToTextConfig {
        std::string modelPath;
        std::string device;
    };

    struct Candidate {
        std::string candidate;
        std::string mid;
    };

    struct OfferResult {
        std::string sessionId;
        std::string type;
        std::string sdp;
        std::vector<Candidate> candidates;
    };

    explicit WebRtcSessionController(size_t maxSessions = 16, std::string omniModelPath = {},
        SpeechToTextConfig speechToText = {},
        std::string omniDevice = "CPU", std::string talkerDevice = "CPU", size_t audioChunkFrames = 0,
        std::string announcedIp = {});

    /// Select streaming ASR from the exported model metadata, not its repository name.
    static bool isRealtimeSpeechModel(const std::string& modelPath);
    /// 1:1 NAT mapping: returns a copy of a host ICE candidate ("candidate:..." or "a=candidate:...")
    /// with its address replaced by announcedIp and the port kept, or an empty string when the
    /// candidate is not a host candidate of the same IP family or already uses announcedIp.
    static std::string announceHostCandidate(const std::string& candidate, const std::string& announcedIp);

    /// OVMS signaling: JSON offer/answer with trickled ICE candidates, legacy events on the "transcript" channel.
    bool createSession(const std::string& offerSdp, const std::string& offerType, OfferResult& result);
    /// OpenAI Realtime API compatible signaling (POST /v1/realtime/calls): the SDP answer embeds all
    /// server ICE candidates and OpenAI transcription events are exchanged on the "oai-events" channel.
    bool createRealtimeCall(const std::string& offerSdp, OfferResult& result);
    bool addCandidate(const std::string& sessionId, const std::string& candidate, const std::string& mid);
    bool getCandidates(const std::string& sessionId, std::vector<Candidate>& candidates) const;
    bool removeSession(const std::string& sessionId);
    size_t sessionCount() const;

private:
    enum class Protocol {
        Ovms,
        OpenAiRealtime
    };

    struct Session : std::enable_shared_from_this<Session> {
        Session(rtc::Configuration configuration, Protocol protocol, std::string id);
        void sendGeneratedAudio(const AudioChunk& audio, rtc::FrameInfo info, bool complete);
        void sendTranscript(const std::string& message);
        void sendUserDelta(const std::string& itemId, const std::string& text);
        void sendUserFinal(const std::string& itemId, const std::string& text);
        void sendUserError(const std::string& text);
        void attachDataChannel(std::shared_ptr<rtc::DataChannel> channel);
        void handleClientEvent(const std::string& message);
        std::string nextItemId();
        void startVoxtralStream();
        std::shared_ptr<VoxtralAudioAdapter::Stream> currentVoxtralStream();
        void commitAudio();
        // Closes the peer connection (releasing its UDP port and callbacks) and all Voxtral streams.
        void shutdown();
        bool isDisconnected() const;
        AudioChunk decodeAudioFrame(const rtc::binary& data, const rtc::FrameInfo& info);

        const Protocol protocol;
        const std::string id;
        WebRtcPeerConnection peer;
        OpusAudioCodec codec;
        MockEchoStreamingAudioModel model;
        StreamingAudioProcessor processor;
        AudioUtteranceBuffer utteranceBuffer;
        std::deque<float> pendingOutputSamples;
        OmniAudioAdapter::ConversationPtr conversation;
        std::shared_ptr<VoxtralAudioAdapter> voxtralAdapter;
        std::shared_ptr<VoxtralAudioAdapter::Stream> voxtralStream;
        // Streams finished by input_audio_buffer.commit that may still be producing their final transcript.
        std::vector<std::shared_ptr<VoxtralAudioAdapter::Stream>> committedVoxtralStreams;
        std::string currentItemId;
        uint64_t itemCounter = 0;
        std::shared_ptr<rtc::DataChannel> transcriptChannel;
        std::mutex transcriptSendMutex;
        std::vector<Candidate> localCandidates;
        std::string localType;
        std::string localSdp;
        bool gatheringComplete = false;
        std::condition_variable descriptionReady;
        mutable std::mutex mutex;
    };

    bool negotiateSession(const std::string& offerSdp, const std::string& offerType, Protocol protocol, OfferResult& result);
    void removeDisconnectedRealtimeCalls();

    mutable std::mutex mutex_;
    size_t maxSessions_;
    std::string omniModelPath_;
    std::string sttModelPath_;
    std::string sttDevice_;
    std::string omniDevice_;
    std::string talkerDevice_;
    size_t audioChunkFrames_;
    std::string announcedIp_;
    std::shared_ptr<OmniAudioAdapter> omniAdapter_;
    std::shared_ptr<SttAudioAdapter> sttAdapter_;
    std::shared_ptr<VoxtralAudioAdapter> voxtralAdapter_;
    uint64_t nextSessionId_ = 1;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
};

}  // namespace ovms
