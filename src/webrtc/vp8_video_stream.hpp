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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mediapipe {
class ImageFrame;
}

namespace ovms {

class Vp8VideoStream {
public:
    using ImageFrameCallback = std::function<void(std::shared_ptr<mediapipe::ImageFrame>, uint32_t timestamp)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    /// Own a VP8 decoder and deliver decoded RGB frames to a live graph.
    Vp8VideoStream(ImageFrameCallback imageFrame, ErrorCallback error);
    ~Vp8VideoStream();
    /// Enqueue one complete encoded VP8 frame; report an error and stop on overflow.
    void push(std::vector<std::byte> frame, uint32_t timestamp);
    /// Stop and join the worker; no callbacks run after this returns.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ovms
