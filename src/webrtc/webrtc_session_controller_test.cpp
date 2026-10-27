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
#include <sstream>
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

#include <openvino/genai/automatic_speech_recognition/pipeline.hpp>

#include "src/audio/audio_utils.hpp"
#include "src/audio/speech_to_text/stt_pipeline_handle.hpp"
#include "src/utils/env_guard.hpp"
#include "webrtc_session_controller.hpp"
#include "webrtc_peer_connection.hpp"

namespace ovms {

TEST(WebRtcSessionControllerTest, RejectsEmptyOffer) {
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;

    EXPECT_EQ(controller.createRealtimeCall("", {}, result), WebRtcSessionController::CreateCallResult::InvalidOffer);
    EXPECT_EQ(controller.sessionCount(), 0);
}

TEST(WebRtcSessionControllerTest, RejectsMalformedSdpOffer) {
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;

    const std::string malformedSdp =
        "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
        "m=audio invalid UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n";
    EXPECT_EQ(controller.createRealtimeCall(malformedSdp, {}, result), WebRtcSessionController::CreateCallResult::InvalidOffer);
    EXPECT_EQ(controller.sessionCount(), 0);
}

TEST(WebRtcSessionControllerTest, ReportsSessionCapacityExhaustion) {
    WebRtcSessionController controller(0);
    WebRtcSessionController::OfferResult result;

    EXPECT_EQ(controller.createRealtimeCall("v=0\r\n", {}, result), WebRtcSessionController::CreateCallResult::CapacityExhausted);
    EXPECT_EQ(controller.sessionCount(), 0);
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

#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
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
#endif

#if defined(BUILD_WITH_EXPERIMENTAL_VOXTRAL)
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

    VoxtralAudioAdapter adapter(std::make_shared<ov::genai::ASRPipeline>(modelPath, "CPU", ov::AnyMap{}));
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
#endif

namespace {
constexpr uint32_t TestSampleRate = 48000;
constexpr size_t TestFrameSamples = 960;
constexpr uint16_t ControllerUdpPort = 52000;
constexpr uint16_t ClientUdpPort = 52001;

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
// messages until one contains finalMarker. Set commitAudio to exercise explicit Realtime commits.
SttPipelineHandle loadSpeechModel(const std::string& modelPath) {
    return SttPipelineHandle{std::make_shared<ov::genai::ASRPipeline>(modelPath, "CPU", ov::AnyMap{}),
        std::make_shared<std::mutex>(), modelPath};
}

void streamAudioOverWebRtc(WebRtcSessionController& controller, const SttPipelineHandle& speechModel,
    const std::vector<float>& samples, bool commitAudio, const std::string& finalMarker,
    std::vector<std::string>& messages, size_t commitCount = 1) {
    rtc::Configuration configuration;
    configuration.portRangeBegin = ClientUdpPort;
    configuration.portRangeEnd = ClientUdpPort;
    auto offerer = std::make_shared<rtc::PeerConnection>(configuration);
    rtc::Description::Audio audio("audio", rtc::Description::Direction::SendOnly);
    audio.addOpusCodec(111);
    auto track = offerer->addTrack(audio);
    track->setMediaHandler(std::make_shared<rtc::OpusRtpPacketizer>(
        std::make_shared<rtc::RtpPacketizationConfig>(1, "webrtc-test", 111, TestSampleRate)));
    auto channel = offerer->createDataChannel("oai-events");

    std::mutex mutex;
    std::condition_variable condition;
    bool gatheringComplete = false;
    size_t finalReceivedCount = 0;
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
            ++finalReceivedCount;
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
    ASSERT_EQ(controller.createRealtimeCall(offerSdp, speechModel, result), WebRtcSessionController::CreateCallResult::Success);
    offerer->setRemoteDescription(rtc::Description(result.sdp, result.type));
    for (const auto& candidate : result.candidates)
        offerer->addRemoteCandidate(rtc::Candidate(candidate.candidate, candidate.mid));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((!track->isOpen() || !channel->isOpen()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_TRUE(track->isOpen());
    ASSERT_TRUE(channel->isOpen());

    OpusAudioCodec codec(TestSampleRate, 2);
    const size_t audioPasses = commitAudio ? commitCount : 1;
    for (size_t pass = 0; pass < audioPasses; ++pass) {
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
            rtc::FrameInfo info(static_cast<uint32_t>(pass * samples.size() + offset));
            info.payloadType = 111;
            track->sendFrame(std::move(packet), info);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (commitAudio)
            channel->send(std::string(R"({"type":"input_audio_buffer.commit"})"));
    }

    {
        std::unique_lock<std::mutex> lock(mutex);
        const size_t expectedFinals = commitAudio ? commitCount : 1;
        EXPECT_TRUE(condition.wait_for(lock, std::chrono::seconds(60), [&] {
            return finalReceivedCount >= expectedFinals;
        }));
    }
    EXPECT_TRUE(controller.removeSession(result.sessionId));
    offerer->close();
    EXPECT_TRUE(waitForUdpPortRelease(ControllerUdpPort));
    EXPECT_TRUE(waitForUdpPortRelease(ClientUdpPort));
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

    WebRtcSessionController controller;
    std::vector<std::string> messages;
    ASSERT_NO_FATAL_FAILURE(streamAudioOverWebRtc(controller, loadSpeechModel(modelPath), samples, false,
        "\"type\":\"conversation.item.input_audio_transcription.delta\"", messages));
    EXPECT_FALSE(containsMessage(messages, "\"type\":\"error\""));
}

TEST(WebRtcSessionControllerTest, VoxtralCommitCreatesIndependentTranscriptItems) {
    const char* modelPath = std::getenv("OVMS_VOXTRAL_MODEL_PATH");
    const char* audioPath = std::getenv("OVMS_VOXTRAL_TEST_WAV");
    if (!modelPath || !audioPath)
        GTEST_SKIP() << "Set OVMS_VOXTRAL_MODEL_PATH and OVMS_VOXTRAL_TEST_WAV to run the real-model test";
    const auto samples = readTestAudio(audioPath);
    ASSERT_FALSE(samples.empty());

    WebRtcSessionController controller;
    std::vector<std::string> messages;
    ASSERT_NO_FATAL_FAILURE(streamAudioOverWebRtc(controller, loadSpeechModel(modelPath), samples, true,
        "\"type\":\"conversation.item.input_audio_transcription.completed\"", messages, 2));

    std::vector<std::string> completedItemIds;
    size_t committedCount = 0;
    for (const auto& message : messages) {
        if (message.find("\"type\":\"input_audio_buffer.committed\"") != std::string::npos)
            ++committedCount;
        if (message.find("\"type\":\"conversation.item.input_audio_transcription.completed\"") == std::string::npos)
            continue;
        const std::string key = "\"item_id\":\"";
        const size_t itemStart = message.find(key);
        ASSERT_NE(itemStart, std::string::npos) << message;
        const size_t valueStart = itemStart + key.size();
        const size_t valueEnd = message.find('"', valueStart);
        ASSERT_NE(valueEnd, std::string::npos) << message;
        completedItemIds.push_back(message.substr(valueStart, valueEnd - valueStart));
    }
    EXPECT_GE(committedCount, 2u);
    ASSERT_EQ(completedItemIds.size(), 2u);
    EXPECT_NE(completedItemIds[0], completedItemIds[1]);
    for (const auto& itemId : completedItemIds) {
        const std::string itemIdField = "\"item_id\":\"" + itemId + "\"";
        EXPECT_TRUE(std::any_of(messages.begin(), messages.end(), [&itemIdField](const std::string& message) {
            return message.find("\"type\":\"conversation.item.input_audio_transcription.delta\"") != std::string::npos &&
                   message.find(itemIdField) != std::string::npos;
        })) << "No delta belonged to completed item " << itemId;
    }
    EXPECT_TRUE(containsMessage(messages, "\"type\":\"session.created\""));
    EXPECT_FALSE(containsMessage(messages, "\"type\":\"error\""));
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

    WebRtcSessionController controller;
    std::vector<std::string> messages;
    ASSERT_NO_FATAL_FAILURE(streamAudioOverWebRtc(controller, loadSpeechModel(modelPath), samples, false,
        "\"type\":\"conversation.item.input_audio_transcription.completed\"", messages));
    EXPECT_TRUE(containsMessage(messages, "animal"));
    EXPECT_FALSE(containsMessage(messages, "\"type\":\"error\""));
}

TEST(WebRtcSessionControllerTest, WhisperTranscribesCommittedAudioInRealtimeCall) {
    const char* modelPath = std::getenv("OVMS_WHISPER_MODEL_PATH");
    const char* audioPath = std::getenv("OVMS_VOXTRAL_TEST_WAV");
    if (!modelPath || !audioPath)
        GTEST_SKIP() << "Set OVMS_WHISPER_MODEL_PATH and OVMS_VOXTRAL_TEST_WAV to run the real-model test";
    const auto samples = readTestAudio(audioPath);
    ASSERT_FALSE(samples.empty());

    WebRtcSessionController controller;
    std::vector<std::string> messages;
    ASSERT_NO_FATAL_FAILURE(streamAudioOverWebRtc(controller, loadSpeechModel(modelPath), samples, true,
        "\"type\":\"conversation.item.input_audio_transcription.completed\"", messages));
    EXPECT_TRUE(containsMessage(messages, "\"type\":\"session.created\""));
    EXPECT_TRUE(containsMessage(messages, "\"type\":\"input_audio_buffer.committed\""));
    EXPECT_TRUE(containsMessage(messages, "animal"));
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
    EnvGuard environment;
    environment.unset("OVMS_WEBRTC_PORT_RANGE");
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    const std::string offerSdp = createGatheredOffer(offerer);
    ASSERT_FALSE(offerSdp.empty());

    ASSERT_EQ(controller.createRealtimeCall(offerSdp, {}, result), WebRtcSessionController::CreateCallResult::Success);
    EXPECT_FALSE(result.sessionId.empty());
    EXPECT_EQ(result.type, "answer");
    // Non-trickle clients (OpenAI Realtime /v1/realtime/calls) need candidates inside the SDP answer.
    EXPECT_NE(result.sdp.find("a=candidate:"), std::string::npos) << result.sdp;
    ASSERT_FALSE(result.candidates.empty());
    for (const auto& candidate : result.candidates) {
        std::istringstream fields(candidate.candidate);
        std::string field;
        for (int i = 0; i < 5; ++i)
            ASSERT_TRUE(static_cast<bool>(fields >> field));
        unsigned int port = 0;
        ASSERT_TRUE(static_cast<bool>(fields >> port));
        EXPECT_GE(port, 52000u);
        EXPECT_LE(port, 52100u);
    }
    // The answer must not be replaced by a renegotiation offer.
    EXPECT_EQ(result.sdp.find("a=setup:actpass"), std::string::npos) << result.sdp;
    EXPECT_EQ(controller.sessionCount(), 1);
    EXPECT_TRUE(controller.removeSession(result.sessionId));
    EXPECT_EQ(controller.sessionCount(), 0);
}

TEST(WebRtcSessionControllerTest, UsesConfiguredUdpPortRange) {
    EnvGuard environment;
    environment.set("OVMS_WEBRTC_PORT_RANGE", "52200-52210");
    WebRtcSessionController controller;
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    const std::string offerSdp = createGatheredOffer(offerer);
    ASSERT_FALSE(offerSdp.empty());

    ASSERT_EQ(controller.createRealtimeCall(offerSdp, {}, result), WebRtcSessionController::CreateCallResult::Success);
    ASSERT_FALSE(result.candidates.empty());
    for (const auto& candidate : result.candidates) {
        std::istringstream fields(candidate.candidate);
        std::string field;
        for (int i = 0; i < 5; ++i)
            ASSERT_TRUE(static_cast<bool>(fields >> field));
        unsigned int port = 0;
        ASSERT_TRUE(static_cast<bool>(fields >> port));
        EXPECT_GE(port, 52200u);
        EXPECT_LE(port, 52210u);
    }
    EXPECT_TRUE(controller.removeSession(result.sessionId));
}

TEST(WebRtcSessionControllerTest, RejectsInvalidUdpPortRange) {
    EnvGuard environment;
    for (const std::string range : {"", "52000", "0-52100", "52000-65536", "52100-52000",
             "-1-52100", "52000-", "52000-52100extra", "52000-52100-52200", " 52000-52100"}) {
        SCOPED_TRACE(range);
        environment.set("OVMS_WEBRTC_PORT_RANGE", range);
        WebRtcSessionController controller;
        WebRtcSessionController::OfferResult result;
        EXPECT_EQ(controller.createRealtimeCall("offer", {}, result), WebRtcSessionController::CreateCallResult::InternalError);
        EXPECT_EQ(controller.sessionCount(), 0);
    }
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
    EnvGuard environment;
    environment.set("OVMS_WEBRTC_PORT_RANGE", "52000-52000");
    const std::string announcedIp = "10.91.240.7";
    WebRtcSessionController controller(16, announcedIp);
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    const std::string offerSdp = createGatheredOffer(offerer);
    ASSERT_FALSE(offerSdp.empty());

    ASSERT_EQ(controller.createRealtimeCall(offerSdp, {}, result), WebRtcSessionController::CreateCallResult::Success);
    EXPECT_NE(result.sdp.find(" " + announcedIp + " 52000 typ host"), std::string::npos) << result.sdp;
    const bool announcedInCandidates = std::any_of(result.candidates.begin(), result.candidates.end(),
        [&](const WebRtcSessionController::Candidate& candidate) {
            return candidate.candidate.find(" " + announcedIp + " ") != std::string::npos;
        });
    EXPECT_TRUE(announcedInCandidates);
    EXPECT_TRUE(controller.removeSession(result.sessionId));
}

TEST(WebRtcSessionControllerTest, IgnoresInvalidAnnouncedIp) {
    WebRtcSessionController controller(16, "not-an-ip");
    WebRtcSessionController::OfferResult result;
    WebRtcPeerConnection offerer(rtc::Configuration{});
    const std::string offerSdp = createGatheredOffer(offerer);
    ASSERT_FALSE(offerSdp.empty());

    ASSERT_EQ(controller.createRealtimeCall(offerSdp, {}, result), WebRtcSessionController::CreateCallResult::Success);
    EXPECT_EQ(result.sdp.find("not-an-ip"), std::string::npos) << result.sdp;
    EXPECT_TRUE(controller.removeSession(result.sessionId));
}

}  // namespace ovms
