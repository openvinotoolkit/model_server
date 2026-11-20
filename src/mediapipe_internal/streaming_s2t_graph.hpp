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

#include <map>
#include <memory>
#include <string>

#include "mediapipe/framework/calculator_graph.h"
#include "src/audio/speech_to_text/streaming_s2t.hpp"

namespace ovms {

/// Adds private result-ready inputs without exposing them as public model inputs.
class StreamingS2TGraphExecution {
public:
    explicit StreamingS2TGraphExecution(mediapipe::CalculatorGraph& graph);
    ~StreamingS2TGraphExecution();
    absl::Status initialize(const mediapipe::CalculatorGraphConfig& config);
    void addSidePackets(std::map<std::string, mediapipe::Packet>& packets) const;
    absl::Status waitUntilIdle();
    void invalidate();
    bool enabled() const { return !bridges_.empty(); }

private:
    mediapipe::CalculatorGraph& graph_;
    StreamingS2TBridgeMap bridges_;
};

}  // namespace ovms
