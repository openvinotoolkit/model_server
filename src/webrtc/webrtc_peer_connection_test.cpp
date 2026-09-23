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
#include <string>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "mock_echo_streaming_audio_model.hpp"
#include "opus_audio_codec.hpp"
#include "streaming_audio_processor.hpp"
#include "webrtc_peer_connection.hpp"

using ovms::WebRtcPeerConnection;
using ovms::MockEchoStreamingAudioModel;
using ovms::OpusAudioCodec;
using ovms::StreamingAudioProcessor;

namespace {

void connectPeers(WebRtcPeerConnection& offerer, WebRtcPeerConnection& answerer, std::string* answerSdp = nullptr) {
    offerer.onLocalCandidate([&answerer](const std::string& candidate, const std::string& mid) {
        answerer.addRemoteCandidate(candidate, mid);
    });
    answerer.onLocalCandidate([&offerer](const std::string& candidate, const std::string& mid) {
        offerer.addRemoteCandidate(candidate, mid);
    });
    offerer.onLocalDescription([&answerer](const std::string& sdp, const std::string& type) {
        answerer.setRemoteDescription(sdp, type);
    });
    answerer.onLocalDescription([&offerer, answerSdp](const std::string& sdp, const std::string& type) {
        if (answerSdp) {
            *answerSdp = sdp;
        }
        offerer.setRemoteDescription(sdp, type);
    });
}

}  // namespace

TEST(WebRtcPeerConnectionTest, OfferContainsOpusAudioTrack) {
    WebRtcPeerConnection pc(rtc::Configuration{});
    std::string localSdp;
    pc.onLocalDescription([&localSdp](const std::string& sdp, const std::string& /*type*/) {
        localSdp = sdp;
    });

    pc.addAudioTrack();
    pc.createOffer();

    ASSERT_FALSE(localSdp.empty());
    EXPECT_NE(localSdp.find("m=audio"), std::string::npos);
    EXPECT_NE(localSdp.find("opus"), std::string::npos);
}

TEST(WebRtcPeerConnectionTest, DirectLibDataChannelPeersOpenAudioTracks) {
    auto offerer = std::make_shared<rtc::PeerConnection>(rtc::Configuration{});
    auto answerer = std::make_shared<rtc::PeerConnection>(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    bool offererConnected = false;
    bool answererConnected = false;
    std::shared_ptr<rtc::Track> answererRemoteTrack;

    offerer->onLocalCandidate([&](rtc::Candidate candidate) {
        answerer->addRemoteCandidate(rtc::Candidate(candidate.candidate(), candidate.mid()));
    });
    answerer->onLocalCandidate([&](rtc::Candidate candidate) {
        offerer->addRemoteCandidate(rtc::Candidate(candidate.candidate(), candidate.mid()));
    });
    offerer->onLocalDescription([&](rtc::Description description) {
        answerer->setRemoteDescription(rtc::Description(std::string(description), description.typeString()));
    });
    answerer->onLocalDescription([&](rtc::Description description) {
        offerer->setRemoteDescription(rtc::Description(std::string(description), description.typeString()));
    });
    offerer->onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            offererConnected = true;
            condition.notify_all();
        }
    });
    answerer->onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            answererConnected = true;
            condition.notify_all();
        }
    });
    answerer->onTrack([&](std::shared_ptr<rtc::Track> track) {
        ASSERT_EQ(track->description().type(), "audio");
        track->setMediaHandler(std::make_shared<rtc::OpusRtpDepacketizer>());
        std::lock_guard<std::mutex> lock(mutex);
        answererRemoteTrack = std::move(track);
        condition.notify_all();
    });
    offerer->onTrack([](std::shared_ptr<rtc::Track>) {});

    rtc::Description::Audio audio("audio", rtc::Description::Direction::SendOnly);
    audio.addOpusCodec(111);
    audio.addSSRC(1, "ovms-audio", "ovms-audio", "ovms-audio");
    auto track = offerer->addTrack(audio);
    auto packetizationConfig = std::make_shared<rtc::RtpPacketizationConfig>(1, "ovms", 111, 48000);
    track->setMediaHandler(std::make_shared<rtc::OpusRtpPacketizer>(std::move(packetizationConfig)));
    offerer->setLocalDescription();

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return offererConnected && answererConnected && answererRemoteTrack;
    }));
    lock.unlock();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((!track->isOpen() || !answererRemoteTrack->isOpen()) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    ASSERT_TRUE(track->isOpen());
    ASSERT_TRUE(answererRemoteTrack->isOpen());
}

TEST(WebRtcPeerConnectionTest, OfferContainsAudioSsrc) {
    WebRtcPeerConnection pc(rtc::Configuration{});
    std::string localSdp;
    pc.onLocalDescription([&localSdp](const std::string& sdp, const std::string&) {
        localSdp = sdp;
    });

    pc.addAudioTrack(rtc::Description::Direction::SendOnly);
    pc.createOffer();

    ASSERT_FALSE(localSdp.empty());
    EXPECT_NE(localSdp.find("a=ssrc:1"), std::string::npos);
}

TEST(WebRtcPeerConnectionTest, LocalPeersConnectBeforeFrameDelivery) {
    WebRtcPeerConnection offerer(rtc::Configuration{});
    WebRtcPeerConnection answerer(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    bool offererConnected = false;
    bool answererConnected = false;
    connectPeers(offerer, answerer);
    offerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            offererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            answererConnected = true;
            condition.notify_all();
        }
    });

    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return offererConnected && answererConnected;
    }));
}

TEST(WebRtcPeerConnectionTest, RemoteAudioTrackIsCreated) {
    WebRtcPeerConnection offerer(rtc::Configuration{});
    WebRtcPeerConnection answerer(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    bool trackReceived = false;
    connectPeers(offerer, answerer);
    answerer.onAudioTrack([&] {
        std::lock_guard<std::mutex> lock(mutex);
        trackReceived = true;
        condition.notify_all();
    });

    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return trackReceived;
    }));
}

TEST(WebRtcPeerConnectionTest, LocalAudioTrackReadiness) {
    WebRtcPeerConnection offerer(rtc::Configuration{});
    WebRtcPeerConnection answerer(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    bool offererConnected = false;
    bool answererConnected = false;
    bool localTrackOpenCallback = false;
    connectPeers(offerer, answerer);
    offerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            offererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            answererConnected = true;
            condition.notify_all();
        }
    });
    offerer.onLocalAudioTrackOpen([&] {
        std::lock_guard<std::mutex> lock(mutex);
        localTrackOpenCallback = true;
        condition.notify_all();
    });

    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return offererConnected && answererConnected;
    }));
    lock.unlock();

    EXPECT_TRUE(offerer.isLocalAudioTrackOpen())
        << "localTrackOpenCallback=" << localTrackOpenCallback;
}

TEST(WebRtcPeerConnectionTest, EncodedAudioFrameCanBeSentAfterTrackOpens) {
    WebRtcPeerConnection offerer(rtc::Configuration{});
    WebRtcPeerConnection answerer(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    bool offererConnected = false;
    bool answererConnected = false;
    bool answererAudioTrackReceived = false;
    bool answererAudioTrackOpen = false;
    size_t receivedFrameSize = 0;
    std::string answerSdp;

    connectPeers(offerer, answerer, &answerSdp);
    offerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            offererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            answererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onAudioTrack([&] {
        std::lock_guard<std::mutex> lock(mutex);
        answererAudioTrackReceived = true;
        condition.notify_all();
    });
    answerer.onAudioTrackOpen([&] {
        std::lock_guard<std::mutex> lock(mutex);
        answererAudioTrackOpen = true;
        condition.notify_all();
    });
    answerer.onAudioFrame([&](rtc::binary data, rtc::FrameInfo) {
        std::lock_guard<std::mutex> lock(mutex);
        receivedFrameSize = data.size();
        condition.notify_all();
    });
    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return offererConnected && answererConnected && answererAudioTrackReceived;
    }));
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return answererAudioTrackOpen;
    }));
    ASSERT_NE(answerSdp.find("m=audio"), std::string::npos) << answerSdp;
    ASSERT_NE(answerSdp.find("a=recvonly"), std::string::npos) << answerSdp;
    lock.unlock();

    const auto readinessDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!offerer.isLocalAudioTrackOpen() && std::chrono::steady_clock::now() < readinessDeadline) {
        std::this_thread::yield();
    }
    ASSERT_TRUE(offerer.isLocalAudioTrackOpen());
    rtc::FrameInfo frameInfo(0);
    frameInfo.payloadType = 111;
    ASSERT_TRUE(offerer.sendAudioFrame(
        rtc::binary{std::byte{0xF8}, std::byte{0xFF}, std::byte{0xFE}}, frameInfo));

    lock.lock();
    EXPECT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return receivedFrameSize > 0;
    }));
}

TEST(WebRtcPeerConnectionTest, LocalTrackReadinessWithRemoteTrackCallback) {
    WebRtcPeerConnection offerer(rtc::Configuration{});
    WebRtcPeerConnection answerer(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    bool offererConnected = false;
    bool answererConnected = false;
    bool answererAudioTrackReceived = false;
    connectPeers(offerer, answerer);
    offerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            offererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            answererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onAudioTrack([&] {
        std::lock_guard<std::mutex> lock(mutex);
        answererAudioTrackReceived = true;
        condition.notify_all();
    });
    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return offererConnected && answererConnected && answererAudioTrackReceived;
    }));
    lock.unlock();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!offerer.isLocalAudioTrackOpen() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(offerer.isLocalAudioTrackOpen());
}

TEST(WebRtcPeerConnectionTest, LocalPeersConnectAndExchangeAudioFrame) {
    WebRtcPeerConnection offerer(rtc::Configuration{});
    WebRtcPeerConnection answerer(rtc::Configuration{});
    std::mutex mutex;
    std::condition_variable condition;
    bool offererConnected = false;
    bool answererConnected = false;
    bool answererAudioTrackReceived = false;
    size_t receivedFrameSize = 0;

    connectPeers(offerer, answerer);
    offerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            offererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            answererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onAudioFrame([&](rtc::binary data, rtc::FrameInfo) {
        std::lock_guard<std::mutex> lock(mutex);
        receivedFrameSize = data.size();
        condition.notify_all();
    });
    answerer.onAudioTrack([&] {
        std::lock_guard<std::mutex> lock(mutex);
        answererAudioTrackReceived = true;
        condition.notify_all();
    });
    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return offererConnected && answererConnected && answererAudioTrackReceived;
    })) << "offererConnected=" << offererConnected << ", answererConnected=" << answererConnected
         << ", answererAudioTrackReceived=" << answererAudioTrackReceived;
    lock.unlock();

    rtc::FrameInfo frameInfo(0);
    frameInfo.payloadType = 111;
    ASSERT_TRUE(offerer.sendAudioFrame(
        rtc::binary{std::byte{0xF8}, std::byte{0xFF}, std::byte{0xFE}}, frameInfo));

    lock.lock();
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return receivedFrameSize > 0;
    }));
}

TEST(WebRtcPeerConnectionTest, ProcessesIncomingOpusThroughStreamingModel) {
    WebRtcPeerConnection offerer(rtc::Configuration{});
    WebRtcPeerConnection answerer(rtc::Configuration{});
    OpusAudioCodec codec(48000, 1);
    MockEchoStreamingAudioModel model(48000, 0.5f, 0.5f);
    StreamingAudioProcessor processor(codec, model);
    std::mutex mutex;
    std::condition_variable condition;
    bool offererConnected = false;
    bool answererConnected = false;
    bool trackReceived = false;
    rtc::binary processedOutput;
    rtc::FrameInfo processedInfo(0);

    connectPeers(offerer, answerer);
    offerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            offererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onStateChange([&](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            std::lock_guard<std::mutex> lock(mutex);
            answererConnected = true;
            condition.notify_all();
        }
    });
    answerer.onProcessedAudioFrame(processor, [&](rtc::binary data, rtc::FrameInfo info) {
        std::lock_guard<std::mutex> lock(mutex);
        processedOutput = std::move(data);
        processedInfo = info;
        trackReceived = true;
        condition.notify_all();
    });

    offerer.addAudioTrack(rtc::Description::Direction::SendOnly);
    offerer.createOffer();

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return offererConnected && answererConnected;
    }));
    lock.unlock();

    std::vector<float> input(OpusAudioCodec::FrameSamples, 0.25f);
    const auto encodedInput = codec.encode(input);
    rtc::binary inputFrame(encodedInput.size());
    for (size_t index = 0; index < encodedInput.size(); ++index) {
        inputFrame[index] = static_cast<std::byte>(encodedInput[index]);
    }
    rtc::FrameInfo inputInfo(960);
    inputInfo.payloadType = 111;
    ASSERT_TRUE(offerer.sendAudioFrame(std::move(inputFrame), inputInfo));

    lock.lock();
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return trackReceived;
    }));
    lock.unlock();

    std::vector<uint8_t> encodedOutput(processedOutput.size());
    for (size_t index = 0; index < processedOutput.size(); ++index) {
        encodedOutput[index] = std::to_integer<uint8_t>(processedOutput[index]);
    }
    const auto decodedOutput = codec.decode(encodedOutput);
    EXPECT_EQ(decodedOutput.size(), OpusAudioCodec::FrameSamples);
    EXPECT_EQ(processedInfo.timestamp, inputInfo.timestamp);
    EXPECT_NE(decodedOutput, input);
}
