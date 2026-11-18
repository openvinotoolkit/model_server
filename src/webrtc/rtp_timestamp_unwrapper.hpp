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
#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace ovms {

class RtpTimestampUnwrapper {
public:
    std::optional<int64_t> toMicroseconds(uint32_t timestamp) {
        if (!initialized_) {
            extendedTicks_ = timestamp;
        } else {
            const uint32_t delta = timestamp - lastRawTimestamp_;
            if (delta == 0 || delta > MaxForwardDelta ||
                extendedTicks_ > std::numeric_limits<uint64_t>::max() - delta)
                return std::nullopt;
            extendedTicks_ += delta;
        }

        if (extendedTicks_ > std::numeric_limits<uint64_t>::max() / MicrosecondsPerSecond)
            return std::nullopt;
        const uint64_t microseconds = extendedTicks_ * MicrosecondsPerSecond / RtpClockRate;
        if (microseconds > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (initialized_ && microseconds <= lastMicroseconds_))
            return std::nullopt;

        initialized_ = true;
        lastRawTimestamp_ = timestamp;
        lastMicroseconds_ = microseconds;
        return static_cast<int64_t>(microseconds);
    }

private:
    static constexpr uint32_t RtpClockRate = 90000;
    static constexpr uint64_t MicrosecondsPerSecond = 1000000;
    static constexpr uint32_t MaxForwardDelta = 0x7fffffff;

    uint32_t lastRawTimestamp_ = 0;
    uint64_t extendedTicks_ = 0;
    uint64_t lastMicroseconds_ = 0;
    bool initialized_ = false;
};

}  // namespace ovms
