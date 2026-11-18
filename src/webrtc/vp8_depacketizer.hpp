// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.
#pragma once

#include <rtc/common.hpp>
#include <rtc/rtpdepacketizer.hpp>

namespace ovms {

// RFC 7741 VP8 payload descriptors are removed before passing frames to libvpx.
class Vp8Depacketizer : public rtc::VideoRtpDepacketizer {
    rtc::message_ptr reassemble(message_buffer& packets) override;
};

}  // namespace ovms
