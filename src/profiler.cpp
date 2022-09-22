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
#include "profiler.hpp"

#include <stdio.h>

namespace ovms {

bool profiler_init(const char* file_path) {
#ifndef MTR_ENABLED
    return true;
#endif
    FILE* f = fopen(file_path, "wb");
    if (f == nullptr) {
        return false;
    }
    mtr_init_from_stream(f);
    return true;
}

void profiler_shutdown() {
#ifndef MTR_ENABLED
    return;
#endif
    mtr_flush();
    mtr_shutdown();
}

Profiler::Profiler(const std::string& file_path) {
    this->initialized = profiler_init(file_path.c_str());
}

Profiler::~Profiler() {
    if (this->isInitialized()) {
        profiler_shutdown();
    }
}

bool Profiler::isInitialized() const {
    return this->initialized;
}

}  // namespace ovms
