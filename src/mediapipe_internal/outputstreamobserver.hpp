//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../execution_context.hpp"
#include "../model_metric_reporter.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../timer.hpp"
#pragma warning(push)
#pragma warning(disable : 4324 6001 6385 6386 6326 6011 4309 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#include "mediapipe_utils.hpp"
#include "mediapipegraphdefinition.hpp"  // for version in response and PythonNodeResourceMap
#include "packettypes.hpp"
#include "graphqueue.hpp"

namespace ovms {
class PythonBackend;
class ServableMetricReporter;
class OutputStreamObserverI {
public:
    virtual absl::Status handlePacket(const ::mediapipe::Packet& packet) = 0;
};
class NullOutputStreamObserver : public OutputStreamObserverI {
public:
    NullOutputStreamObserver() {
        SPDLOG_ERROR("NUll observer constructed:{}", (void*)this);
    }
    absl::Status handlePacket(const ::mediapipe::Packet& packet) override {
        SPDLOG_ERROR("Internal error occured:{}", (void*)this);
        throw std::runtime_error("Should not happen");
        return absl::Status(absl::StatusCode::kInternal, "Should not happen");
    }
};
/*
*/
}  // namespace ovms
