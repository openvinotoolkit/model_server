// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ovms {

class MovinetVideoModel {
public:
    /// Load the custom flattened MoViNet-A0 IR once; share it across video sessions.
    MovinetVideoModel(const std::string& modelPath, const std::string& labelsPath);
    ~MovinetVideoModel();

private:
    friend class MovinetVideoStream;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MovinetVideoStream {
public:
    using PredictionCallback = std::function<void(size_t frame, uint32_t timestamp, const std::string& label, float score)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    /// Own a VP8 decoder, inference state, and a bounded asynchronous input queue.
    MovinetVideoStream(std::shared_ptr<MovinetVideoModel> model, PredictionCallback prediction, ErrorCallback error);
    ~MovinetVideoStream();
    /// Enqueue one complete encoded VP8 frame. Overload terminates video with an error.
    void push(std::vector<std::byte> frame, uint32_t timestamp);
    /// Stop and join the worker; no callbacks run after this returns.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ovms
