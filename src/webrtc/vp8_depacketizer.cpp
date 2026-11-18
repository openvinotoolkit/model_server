// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.
#include "vp8_depacketizer.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <rtc/rtp.hpp>

#include "src/logging.hpp"

namespace ovms {

rtc::message_ptr Vp8Depacketizer::reassemble(message_buffer& packets) {
    constexpr size_t MaxFrameBytes = 2 * 1024 * 1024;
    if (packets.empty())
        return nullptr;
    const auto* first = reinterpret_cast<const rtc::RtpHeader*>((*packets.begin())->data());
    uint16_t sequence = first->seqNumber();
    rtc::binary frame;
    bool firstPacket = true;
    for (const auto& packet : packets) {
        const auto* header = reinterpret_cast<const rtc::RtpHeader*>(packet->data());
        const size_t baseSize = header->getSize();
        if (header->extension() && packet->size() < baseSize + sizeof(rtc::RtpExtensionHeader))
            break;
        size_t offset = baseSize + header->getExtensionHeaderSize();
        const size_t padding = header->padding() ? std::to_integer<uint8_t>(packet->back()) : 0;
        if (header->seqNumber() != sequence++ || padding > packet->size() ||
            offset >= packet->size() - padding)
            break;
        const size_t end = packet->size() - padding;
        const uint8_t descriptor = std::to_integer<uint8_t>((*packet)[offset++]);
        constexpr uint8_t StartPartitionZeroMask = 0x1f;
        constexpr uint8_t StartPartitionZero = 0x10;
        if (firstPacket && (descriptor & StartPartitionZeroMask) != StartPartitionZero)
            break;
        firstPacket = false;
        constexpr uint8_t ExtensionMask = 0x80;
        if (descriptor & ExtensionMask) {
            if (offset >= end)
                break;
            const uint8_t extension = std::to_integer<uint8_t>((*packet)[offset++]);
            constexpr uint8_t PictureIdMask = 0x80;
            constexpr uint8_t Tl0PicIdxMask = 0x40;
            constexpr uint8_t TemporalOrKeyIdxMask = 0x30;
            if (extension & PictureIdMask) {
                if (offset >= end)
                    break;
                const bool longId = (std::to_integer<uint8_t>((*packet)[offset++]) & PictureIdMask) != 0;
                if (longId)
                    ++offset;
            }
            if (extension & Tl0PicIdxMask)
                ++offset;
            if (extension & TemporalOrKeyIdxMask)
                ++offset;
        }
        if (offset >= end || frame.size() + end - offset > MaxFrameBytes)
            break;
        frame.insert(frame.end(), packet->begin() + offset, packet->begin() + end);
        if (header->marker())
            return rtc::make_message(std::move(frame), createFrameInfo(first->timestamp(), first->payloadType()));
    }
    SPDLOG_LOGGER_WARN(webrtc_logger, "Dropping incomplete, malformed, or oversized VP8 frame");
    return nullptr;
}

}  // namespace ovms
