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

#include <chrono>
#include <condition_variable>
#include <mutex>

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

}  // namespace ovms