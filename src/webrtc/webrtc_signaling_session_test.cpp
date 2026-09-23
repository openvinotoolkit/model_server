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

#include "webrtc_signaling_session.hpp"

using ovms::WebRtcSignalingSession;

TEST(WebRtcSignalingSessionTest, AcceptsOfferThenAnswerAndCandidates) {
    WebRtcSignalingSession session;
    EXPECT_EQ(session.state(), WebRtcSignalingSession::State::New);
    EXPECT_TRUE(session.setRemoteOffer("offer-sdp"));
    EXPECT_EQ(session.state(), WebRtcSignalingSession::State::HaveRemoteOffer);
    EXPECT_TRUE(session.setLocalAnswer("answer-sdp"));
    EXPECT_EQ(session.state(), WebRtcSignalingSession::State::Stable);
    EXPECT_TRUE(session.addRemoteCandidate("candidate", "audio"));
    EXPECT_EQ(session.remoteCandidates().size(), 1);
}

TEST(WebRtcSignalingSessionTest, RejectsAnswerBeforeOffer) {
    WebRtcSignalingSession session;
    EXPECT_FALSE(session.setLocalAnswer("answer-sdp"));
    EXPECT_EQ(session.state(), WebRtcSignalingSession::State::New);
}
