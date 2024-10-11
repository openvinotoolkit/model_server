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

#include <malloc.h>

#include "global_sequences_viewer.hpp"
#include "logging.hpp"
#include "modelmanager.hpp"

namespace ovms {
FunctorSequenceCleaner::FunctorSequenceCleaner(GlobalSequencesViewer& globalSequencesViewer) :
    globalSequencesViewer(globalSequencesViewer) {}

void FunctorSequenceCleaner::cleanup() {
    globalSequencesViewer.removeIdleSequences();
    SPDLOG_TRACE("malloc_trim(0)");
#ifdef __linux__
    malloc_trim(0);
#endif
    // TODO: windows for malloc_trim(0);
}

FunctorSequenceCleaner::~FunctorSequenceCleaner() = default;

FunctorResourcesCleaner::~FunctorResourcesCleaner() = default;

FunctorResourcesCleaner::FunctorResourcesCleaner(ModelManager& modelManager) :
    modelManager(modelManager) {}

void FunctorResourcesCleaner::cleanup() {
    modelManager.cleanupResources();
}
}  // namespace ovms
