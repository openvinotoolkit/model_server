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
#include <string>
#include <utility>
#include <vector>

namespace ovms {

class WebRtcSignalingSession {
public:
    enum class State { New, HaveRemoteOffer, Stable };

    struct Candidate {
        std::string value;
        std::string mid;
    };

    bool setRemoteOffer(std::string sdp);
    bool setLocalAnswer(std::string sdp);
    bool addRemoteCandidate(std::string candidate, std::string mid);

    State state() const;
    const std::string& remoteOffer() const;
    const std::string& localAnswer() const;
    const std::vector<Candidate>& remoteCandidates() const;

private:
    State state_ = State::New;
    std::string remoteOffer_;
    std::string localAnswer_;
    std::vector<Candidate> remoteCandidates_;
};

}  // namespace ovms
