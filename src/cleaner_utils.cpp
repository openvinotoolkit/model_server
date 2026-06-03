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

#ifdef _WIN32
#include <crtdbg.h>
#endif

#include "resources_cleaner.hpp"

namespace ovms {

#ifdef _WIN32
bool malloc_trim_win() {
    return (_heapmin() == 0);
}
#endif

FunctorResourcesCleaner::~FunctorResourcesCleaner() = default;

FunctorResourcesCleaner::FunctorResourcesCleaner(ResourcesCleaner& resourcesCleaner) :
    resourcesCleaner(resourcesCleaner) {}

void FunctorResourcesCleaner::cleanup() {
    resourcesCleaner.cleanupResources();
}
}  // namespace ovms
