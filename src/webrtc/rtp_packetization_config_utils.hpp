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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <rtc/description.hpp>
#include <rtc/rtppacketizationconfig.hpp>

namespace ovms {

inline std::optional<uint8_t> findOutboundMidExtensionId(rtc::Description::Media& description) {
    for (const int extensionId : description.extIds()) {
        const auto* extension = description.extMap(extensionId);
        if (extension == nullptr || extension->uri != "urn:ietf:params:rtp-hdrext:sdes:mid") {
            continue;
        }
        if (extension->direction == rtc::Description::Direction::SendOnly ||
            extension->direction == rtc::Description::Direction::Inactive) {
            continue;
        }
        return static_cast<uint8_t>(extensionId);
    }
    return std::nullopt;
}

inline std::shared_ptr<rtc::RtpPacketizationConfig> createAudioRtpPacketizationConfig(
    rtc::Description::Media& description,
    const std::string& mid,
    uint32_t ssrc,
    uint8_t payloadType,
    uint32_t clockRate) {
    auto config = std::make_shared<rtc::RtpPacketizationConfig>(ssrc, "ovms", payloadType, clockRate);
    if (const auto midExtensionId = findOutboundMidExtensionId(description)) {
        config->mid = mid;
        config->midId = *midExtensionId;
    }
    return config;
}

}  // namespace ovms
