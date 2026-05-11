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

#include <csignal>

namespace ovms {
int getShutdownRequestValue();
void setShutdownRequestValue(int value);
int getExitStatusValue();
void setExitStatusValue(int value);

// Signal-safe API: for use in signal handlers and polling contexts
// Signal handlers should call setSignalShutdownRequested() which is async-signal-safe.
// Main loop/polling thread should periodically call processSignalShutdownRequest() to
// perform actual shutdown operations from a safe context.
bool isSignalShutdownRequested();
void processSignalShutdownRequest();
void setSignalShutdownRequested(int value);
}  // namespace ovms
