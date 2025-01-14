//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include "cleaner_utils.hpp"

#include <string>

#include <malloc.h>

#include "global_sequences_viewer.hpp"
#include "logging.hpp"
#include "modelmanager.hpp"

namespace ovms {
FunctorSequenceCleaner::FunctorSequenceCleaner(GlobalSequencesViewer& globalSequencesViewer) :
    globalSequencesViewer(globalSequencesViewer) {}

#ifdef _WIN32
bool malloc_trim_win() {
    HANDLE heap = GetProcessHeap();
    if (heap == NULL) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_ERROR("Failed to get process heap: {}", message);
        return false;
    }

    ULONG_PTR freed_bytes = HeapCompact(heap, 0);
    if (freed_bytes == 0) {
        DWORD error = GetLastError();
        if (error != 0) {
            std::string message = std::system_category().message(error);
            SPDLOG_ERROR("HeapCompact failed with error: {}", message);
            return false;
        }
    }
    return true;
}
#endif

void FunctorSequenceCleaner::cleanup() {
    globalSequencesViewer.removeIdleSequences();
    SPDLOG_TRACE("malloc_trim(0)");
#ifdef __linux__
    malloc_trim(0);
#elif _WIN32
    malloc_trim_win();
#endif
}

FunctorSequenceCleaner::~FunctorSequenceCleaner() = default;

FunctorResourcesCleaner::~FunctorResourcesCleaner() = default;

FunctorResourcesCleaner::FunctorResourcesCleaner(ModelManager& modelManager) :
    modelManager(modelManager) {}

void FunctorResourcesCleaner::cleanup() {
    modelManager.cleanupResources();
}
}  // namespace ovms
