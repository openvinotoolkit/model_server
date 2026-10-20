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
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <rtc/rtp.hpp>
#include <rtc/rtppacketizer.hpp>

#include "src/audio/audio_utils.hpp"
#include "webrtc_session_controller.hpp"
#include "webrtc_peer_connection.hpp"

namespace ovms {

TEST(WebRtcSessionControllerTest, RejectsInvalidOffer) {
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;

    EXPECT_FALSE(controller.createSession("", "offer", result));
    EXPECT_FALSE(controller.createSession("v=0", "answer", result));
    EXPECT_EQ(controller.sessionCount(), 0);
}

TEST(WebRtcSessionControllerTest, RejectsSessionWhenConfiguredSpeechModelCannotLoad) {
    WebRtcSessionController controller(16, {"/missing/webrtc/speech/model", "CPU"});
    WebRtcSessionController::OfferResult result;

    EXPECT_FALSE(controller.createSession("v=0\r\n", "offer", result));
    EXPECT_EQ(controller.sessionCount(), 0);
}

TEST(WebRtcSessionControllerTest, ReportsWhetherSpeechModelIsConfigured) {
    EXPECT_FALSE(WebRtcSessionController().hasSpeechModel());
    EXPECT_TRUE(WebRtcSessionController(16, {"/speech", "CPU"}).hasSpeechModel());
}

class WebRtcSpeechModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
                    ("ovms-webrtc-speech-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
    }

    void TearDown() override {
        std::filesystem::remove(directory / "config.json");
        std::filesystem::remove(directory);
    }

    void writeConfig(const std::string& content) {
        std::ofstream file(directory / "config.json");
        ASSERT_TRUE(file.is_open());
        file << content;
        ASSERT_TRUE(file.good());
    }

    std::filesystem::path directory;
};

TEST_F(WebRtcSpeechModelTest, SelectsRealtimeFromMetadataRegardlessOfDirectoryName) {
    writeConfig(R"({"model_type":"voxtral_realtime"})");
    EXPECT_TRUE(WebRtcSessionController::isRealtimeSpeechModel(directory.string()));
}

TEST_F(WebRtcSpeechModelTest, SelectsWhisperFromMetadata) {
    writeConfig(R"({"model_type":"whisper"})");
    EXPECT_FALSE(WebRtcSessionController::isRealtimeSpeechModel(directory.string()));
}

TEST_F(WebRtcSpeechModelTest, RejectsMissingOrInvalidConfiguration) {
    EXPECT_THROW(WebRtcSessionController::isRealtimeSpeechModel(directory.string()), std::runtime_error);
    for (const std::string& content : {"invalid json", "[]", "{}", R"({"model_type":42})"}) {
        writeConfig(content);
        EXPECT_THROW(WebRtcSessionController::isRealtimeSpeechModel(directory.string()), std::runtime_error);
    }
}

TEST(VoxtralAudioAdapterTest, EmitsTextBeforeInputEnds) {
    const char* modelPath = std::getenv("OVMS_VOXTRAL_MODEL_PATH");
    const char* audioPath = std::getenv("OVMS_VOXTRAL_TEST_WAV");
    if (!modelPath || !audioPath)
        GTEST_SKIP() << "Set OVMS_VOXTRAL_MODEL_PATH and OVMS_VOXTRAL_TEST_WAV to run the real-model test";

    std::ifstream file(audioPath, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    const std::string wav{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    const auto samples = audio_utils::readWav(wav, 48000);
    ASSERT_GT(samples.size(), 960u);

    VoxtralAudioAdapter adapter(modelPath, "CPU");
    std::string deltas;
    std::string final;
    std::exception_ptr error;
    auto stream = adapter.createStream(
        [&deltas](const std::string& text) { deltas += text; },
        [&final](const std::string& text) { final = text; },
        [&error](std::exception_ptr exception) { error = exception; });
    for (size_t offset = 0; offset < samples.size(); offset += 960) {
        const size_t end = std::min(offset + 960, samples.size());
        stream->push(AudioChunk{{samples.begin() + offset, samples.begin() + end}, 48000,
            static_cast<uint64_t>(offset * 1000000 / 48000), 1});
    }
    stream->close();
    if (error)
        std::rethrow_exception(error);
    EXPECT_FALSE(deltas.empty());
    EXPECT_NE(final.find("biggest animal"), std::string::npos) << final;
}

namespace {
constexpr uint32_t TestSampleRate = 48000;
constexpr size_t TestFrameSamples = 960;
constexpr uint16_t ControllerUdpPort = 52000;

// libdatachannel releases the UDP socket of a connected ICE agent asynchronously after close, so wait
// until the controller's fixed port can be bound again before the next test creates a session.
bool waitForUdpPortRelease(uint16_t port) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        const int socketFd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socketFd < 0)
            return false;
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(port);
        const bool bound = bind(socketFd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0;
        close(socketFd);
        if (bound)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

std::vector<float> readTestAudio(const char* audioPath) {
    std::ifstream file(audioPath, std::ios::binary);
    if (!file.is_open())
        return {};
    const std::string wav{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    return audio_utils::readWav(wav, TestSampleRate);
}

// Streams 48 kHz mono audio to the controller over a real WebRTC connection and collects data channel
// messages until one contains finalMarker. The realtime call protocol commits the audio buffer explicitly.
void streamAudioOverWebRtc(WebRtcSessionController& controller, const std::vector<float>& samples, bool realtimeCall,
    const std::string& finalMarker, std::vector<std::string>& messages) {
    rtc::Configuration configuration;
    configuration.portRangeBegin = 52001;
    configuration.portRangeEnd = 52001;
    auto offerer = std::make_shared<rtc::PeerConnection>(configuration);
    rtc::Description::Audio audio("audio", rtc::Description::Direction::SendOnly);
    audio.addOpusCodec(111);
    auto track = offerer->addTrack(audio);
    track->setMediaHandler(std::make_shared<rtc::OpusRtpPacketizer>(
        std::make_shared<rtc::RtpPacketizationConfig>(1, "webrtc-test", 111, TestSampleRate)));
    auto channel = offerer->createDataChannel(realtimeCall ? "oai-events" : "transcript");

    std::mutex mutex;
    std::condition_variable condition;
    bool gatheringComplete = false;
    bool finalReceived = false;
    offerer->onGatheringStateChange([&](rtc::PeerConnection::GatheringState state) {
        if (state != rtc::PeerConnection::GatheringState::Complete)
            return;
        std::lock_guard<std::mutex> lock(mutex);
        gatheringComplete = true;
        condition.notify_all();
    });
    channel->onMessage([&](rtc::message_variant message) {
        if (!std::holds_alternative<std::string>(message))
            return;
        const std::string& text = std::get<std::string>(message);
        std::lock_guard<std::mutex> lock(mutex);
        messages.push_back(text);
        if (text.find(finalMarker) != std::string::npos) {
            finalReceived = true;
            condition.notify_all();
        }
    });
    offerer->setLocalDescription();
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] { return gatheringComplete; }));
    }
    const std::string offerSdp = std::string(*offerer->localDescription());

    WebRtcSessionController::OfferResult result;
    if (realtimeCall) {
        ASSERT_TRUE(controller.createRealtimeCall(offerSdp, result));
    } else {
        ASSERT_TRUE(controller.createSession(offerSdp, "offer", result));
    }
    offerer->setRemoteDescription(rtc::Description(result.sdp, result.type));
    for (const auto& candidate : result.candidates)
        offerer->addRemoteCandidate(rtc::Candidate(candidate.candidate, candidate.mid));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((!track->isOpen() || !channel->isOpen()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_TRUE(track->isOpen());
    ASSERT_TRUE(channel->isOpen());

    OpusAudioCodec codec(TestSampleRate, 2);
    for (size_t offset = 0; offset < samples.size(); offset += TestFrameSamples) {
        std::vector<float> stereo(TestFrameSamples * 2, 0.0f);
        for (size_t index = 0; index < std::min(TestFrameSamples, samples.size() - offset); ++index) {
            stereo[2 * index] = samples[offset + index];
            stereo[2 * index + 1] = samples[offset + index];
        }
        const auto encoded = codec.encode(stereo);
        rtc::binary packet(encoded.size());
        for (size_t index = 0; index < encoded.size(); ++index)
            packet[index] = static_cast<std::byte>(encoded[index]);
        rtc::FrameInfo info(static_cast<uint32_t>(offset));
        info.payloadType = 111;
        track->sendFrame(std::move(packet), info);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (realtimeCall)
        channel->send(std::string(R"({"type":"input_audio_buffer.commit"})"));

    {
        std::unique_lock<std::mutex> lock(mutex);
        EXPECT_TRUE(condition.wait_for(lock, std::chrono::seconds(60), [&] { return finalReceived; }));
    }
    EXPECT_TRUE(controller.removeSession(result.sessionId));
    offerer->close();
    EXPECT_TRUE(waitForUdpPortRelease(ControllerUdpPort));
}

bool containsMessage(const std::vector<std::string>& messages, const std::string& fragment) {
    return std::any_of(messages.begin(), messages.end(),
        [&fragment](const std::string& message) { return message.find(fragment) != std::string::npos; });
}
}  // namespace

TEST(WebRtcSessionControllerTest, VoxtralTranscribesWebRtcAudioWithoutSilenceDetection) {
    const char* modelPath = std::getenv("OVMS_VOXTRAL_MODEL_PATH");
    const char* audioPath = std::getenv("OVMS_VOXTRAL_TEST_WAV");
    if (!modelPath || !audioPath)
        GTEST_SKIP() << "Set OVMS_VOXTRAL_MODEL_PATH and OVMS_VOXTRAL_TEST_WAV to run the real-model test";
    const auto samples = readTestAudio(audioPath);
    ASSERT_FALSE(samples.empty());

    WebRtcSessionController controller(16, {modelPath, "CPU"});
    std::vector<std::string> messages;
    // The legacy protocol has no commit, so Voxtral only streams deltas until the session ends.
    ASSERT_NO_FATAL_FAILURE(streamAudioOverWebRtc(controller, samples, false, "\"type\":\"user_delta\"", messages));
    EXPECT_FALSE(containsMessage(messages, "\"type\":\"user_error\""));
}

TEST(WebRtcSessionControllerTest, WhisperTranscribesUtteranceAfterSilence) {
    const char* modelPath = std::getenv("OVMS_WHISPER_MODEL_PATH");
    const char* audioPath = std::getenv("OVMS_VOXTRAL_TEST_WAV");
    if (!modelPath || !audioPath)
        GTEST_SKIP() << "Set OVMS_WHISPER_MODEL_PATH and OVMS_VOXTRAL_TEST_WAV to run the real-model test";
    auto samples = readTestAudio(audioPath);
    ASSERT_FALSE(samples.empty());
    // Trailing silence longer than the utterance buffer hangover closes the utterance.
    samples.resize(samples.size() + TestSampleRate, 0.0f);

    WebRtcSessionController controller(16, {modelPath, "CPU"});
    std::vector<std::string> messages;
    ASSERT_NO_FATAL_FAILURE(streamAudioOverWebRtc(controller, samples, false, "\"type\":\"user_final\"", messages));
    EXPECT_TRUE(containsMessage(messages, "animal"));
    EXPECT_FALSE(containsMessage(messages, "\"type\":\"user_error\""));
}

TEST(WebRtcSessionControllerTest, WhisperTranscribesCommittedAudioInRealtimeCall) {
    const char* modelPath = std::getenv("OVMS_WHISPER_MODEL_PATH");
    const char* audioPath = std::getenv("OVMS_VOXTRAL_TEST_WAV");
    if (!modelPath || !audioPath)
        GTEST_SKIP() << "Set OVMS_WHISPER_MODEL_PATH and OVMS_VOXTRAL_TEST_WAV to run the real-model test";
    const auto samples = readTestAudio(audioPath);
    ASSERT_FALSE(samples.empty());

    WebRtcSessionController controller(16, {modelPath, "CPU"});
    std::vector<std::string> messages;
    ASSERT_NO_FATAL_FAILURE(streamAudioOverWebRtc(controller, samples, true,
        "\"type\":\"conversation.item.input_audio_transcription.completed\"", messages));
    EXPECT_TRUE(containsMessage(messages, "\"type\":\"session.created\""));
    EXPECT_TRUE(containsMessage(messages, "\"type\":\"input_audio_buffer.committed\""));
    EXPECT_TRUE(containsMessage(messages, "animal"));
}

TEST(WebRtcSessionControllerTest, CreatesAndRemovesSession) {
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    std::string offerSdp;
    std::string offerType;
    offerer.onLocalDescription([&](const std::string& sdp, const std::string& type) {
        std::lock_guard<std::mutex> lock(mutex);
        offerSdp = sdp;
        offerType = type;
        condition.notify_all();
    });
    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return !offerSdp.empty();
    }));
    lock.unlock();

    ASSERT_TRUE(controller.createSession(offerSdp, offerType, result));
    EXPECT_FALSE(result.sessionId.empty());
    EXPECT_EQ(result.type, "answer");
    EXPECT_FALSE(result.sdp.empty());
    EXPECT_EQ(controller.sessionCount(), 1);
    EXPECT_FALSE(controller.addCandidate("missing", "candidate", "0"));
    EXPECT_TRUE(controller.removeSession(result.sessionId));
    EXPECT_EQ(controller.sessionCount(), 0);
}

TEST(WebRtcSessionControllerTest, RejectsInvalidRealtimeCall) {
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;
    EXPECT_FALSE(controller.createRealtimeCall("", result));
    EXPECT_EQ(controller.sessionCount(), 0);
}

namespace {
std::string createGatheredOffer(WebRtcPeerConnection& offerer) {
    std::mutex mutex;
    std::condition_variable condition;
    bool gatheringComplete = false;
    offerer.onGatheringComplete([&]() {
        std::lock_guard<std::mutex> lock(mutex);
        gatheringComplete = true;
        condition.notify_all();
    });
    offerer.addAudioTrack(rtc::Description::Direction::SendRecv);
    offerer.createOffer();
    std::unique_lock<std::mutex> lock(mutex);
    if (!condition.wait_for(lock, std::chrono::seconds(5), [&] { return gatheringComplete; }))
        return {};
    return offerer.localDescriptionSdp();
}
}  // namespace

TEST(WebRtcSessionControllerTest, CreatesRealtimeCallWithCompleteAnswer) {
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    const std::string offerSdp = createGatheredOffer(offerer);
    ASSERT_FALSE(offerSdp.empty());

    ASSERT_TRUE(controller.createRealtimeCall(offerSdp, result));
    EXPECT_FALSE(result.sessionId.empty());
    EXPECT_EQ(result.type, "answer");
    // Non-trickle clients (OpenAI Realtime /v1/realtime/calls) need candidates inside the SDP answer.
    EXPECT_NE(result.sdp.find("a=candidate:"), std::string::npos) << result.sdp;
    // The answer must not be replaced by a renegotiation offer.
    EXPECT_EQ(result.sdp.find("a=setup:actpass"), std::string::npos) << result.sdp;
    EXPECT_EQ(controller.sessionCount(), 1);
    EXPECT_TRUE(controller.removeSession(result.sessionId));
    EXPECT_EQ(controller.sessionCount(), 0);
}

TEST(WebRtcSessionControllerTest, AnnouncesHostCandidateWithConfiguredIp) {
    EXPECT_EQ(WebRtcSessionController::announceHostCandidate(
                  "candidate:1 1 UDP 2114977791 172.17.0.8 52000 typ host", "10.91.240.7"),
        "candidate:1 1 UDP 2114977791 10.91.240.7 52000 typ host");
    EXPECT_EQ(WebRtcSessionController::announceHostCandidate(
                  "a=candidate:1 1 UDP 2114977791 172.17.0.8 52000 typ host", "10.91.240.7"),
        "a=candidate:1 1 UDP 2114977791 10.91.240.7 52000 typ host");
    EXPECT_EQ(WebRtcSessionController::announceHostCandidate(
                  "candidate:2 1 UDP 2114977535 fd00::8 52000 typ host", "2001:db8::1"),
        "candidate:2 1 UDP 2114977535 2001:db8::1 52000 typ host");
}

TEST(WebRtcSessionControllerTest, DoesNotAnnounceUnsupportedCandidates) {
    const std::string hostCandidate = "candidate:1 1 UDP 2114977791 172.17.0.8 52000 typ host";
    EXPECT_TRUE(WebRtcSessionController::announceHostCandidate(hostCandidate, "").empty());
    EXPECT_TRUE(WebRtcSessionController::announceHostCandidate(hostCandidate, "172.17.0.8").empty());
    EXPECT_TRUE(WebRtcSessionController::announceHostCandidate(hostCandidate, "2001:db8::1").empty());
    EXPECT_TRUE(WebRtcSessionController::announceHostCandidate(
        "candidate:2 1 UDP 2114977535 fd00::8 52000 typ host", "10.91.240.7")
                    .empty());
    EXPECT_TRUE(WebRtcSessionController::announceHostCandidate(
        "candidate:3 1 UDP 1677729535 34.1.2.3 52000 typ srflx raddr 172.17.0.8 rport 52000", "10.91.240.7")
                    .empty());
    EXPECT_TRUE(WebRtcSessionController::announceHostCandidate(
        "candidate:4 1 UDP 2114977791 abc.local 52000 typ host", "10.91.240.7")
                    .empty());
    EXPECT_TRUE(WebRtcSessionController::announceHostCandidate("not a candidate", "10.91.240.7").empty());
}

TEST(WebRtcSessionControllerTest, RealtimeCallAnnouncesConfiguredIp) {
    const std::string announcedIp = "10.91.240.7";
    WebRtcSessionController controller(16, {}, announcedIp);
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    const std::string offerSdp = createGatheredOffer(offerer);
    ASSERT_FALSE(offerSdp.empty());

    ASSERT_TRUE(controller.createRealtimeCall(offerSdp, result));
    EXPECT_NE(result.sdp.find(" " + announcedIp + " 52000 typ host"), std::string::npos) << result.sdp;
    const bool announcedInCandidates = std::any_of(result.candidates.begin(), result.candidates.end(),
        [&](const WebRtcSessionController::Candidate& candidate) {
            return candidate.candidate.find(" " + announcedIp + " ") != std::string::npos;
        });
    EXPECT_TRUE(announcedInCandidates);
    EXPECT_TRUE(controller.removeSession(result.sessionId));
}

TEST(WebRtcSessionControllerTest, IgnoresInvalidAnnouncedIp) {
    WebRtcSessionController controller(16, {}, "not-an-ip");
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    const std::string offerSdp = createGatheredOffer(offerer);
    ASSERT_FALSE(offerSdp.empty());

    ASSERT_TRUE(controller.createRealtimeCall(offerSdp, result));
    EXPECT_EQ(result.sdp.find("not-an-ip"), std::string::npos) << result.sdp;
    EXPECT_TRUE(controller.removeSession(result.sessionId));
}

}  // namespace ovms
