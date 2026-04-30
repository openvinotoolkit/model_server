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
#include "shutdown_state.hpp"

#include <atomic>
#include <csignal>

namespace {
std::atomic<int> shutdown_request{0};
std::atomic<int> ovms_exited{0};
// Volatile sig_atomic_t flags: async-signal-safe for use in signal handlers
volatile sig_atomic_t signal_shutdown_requested{0};
volatile sig_atomic_t signal_shutdown_value{0};
}  // namespace

namespace ovms {
int getShutdownRequestValue() {
    return shutdown_request.load(std::memory_order_relaxed);
}

void setShutdownRequestValue(int value) {
    shutdown_request.store(value, std::memory_order_relaxed);
}

int getExitStatusValue() {
    return ovms_exited.load(std::memory_order_relaxed);
}

void setExitStatusValue(int value) {
    ovms_exited.store(value, std::memory_order_relaxed);
}

bool isSignalShutdownRequested() {
    return signal_shutdown_requested != 0;
}

void processSignalShutdownRequest() {
    if (isSignalShutdownRequested()) {
        signal_shutdown_requested = 0;
        setShutdownRequestValue(signal_shutdown_value);
    }
}

void setSignalShutdownRequested(int value) {
    signal_shutdown_value = value;
    signal_shutdown_requested = 1;
}
}  // namespace ovms
