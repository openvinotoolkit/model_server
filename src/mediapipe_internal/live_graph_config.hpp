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

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/packet.h"
#include "live_graph_session.hpp"

namespace ovms {

/// Runtime-only conversion between a profile's declared payload and MediaPipe.
struct LiveGraphStreamCodec {
    std::function<Status(const LiveGraphPacket&, mediapipe::Packet&)> encode;
    std::function<Status(const mediapipe::Packet&, LiveGraphSession::Observer&, const std::string&)> observe;

    template <typename T>
    static LiveGraphStreamCodec typed() {
        return {
            [](const LiveGraphPacket& input, mediapipe::Packet& output) -> Status {
                const T* value = input.getIf<T>();
                if (!value)
                    return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Live graph input payload type mismatch");
                output = mediapipe::MakePacket<T>(*value);
                return StatusCode::OK;
            },
            [](const mediapipe::Packet& input, LiveGraphSession::Observer& observer, const std::string& stream) -> Status {
                const auto status = input.ValidateAsType<T>();
                if (!status.ok())
                    return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, status.ToString());
                const auto owner = std::make_shared<mediapipe::Packet>(input);
                return observer(stream, LiveGraphPacket::share<T>(
                    std::shared_ptr<const T>(owner, &owner->Get<T>())));
            }};
    }
};

/// Prepared at graph load time; the executor does not interpret calculator semantics.
struct LiveGraphConfig {
    mediapipe::CalculatorGraphConfig graph;
    std::map<std::string, LiveGraphStreamCodec> inputs;
    std::map<std::string, LiveGraphStreamCodec> outputs;
};

}  // namespace ovms
