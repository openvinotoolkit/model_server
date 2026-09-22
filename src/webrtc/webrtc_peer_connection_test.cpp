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

#include "webrtc_peer_connection.hpp"

using ovms::WebRtcPeerConnection;

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
        if (type == "offer") {
            answerer.createAnswer();
        }
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
