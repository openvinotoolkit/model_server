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
#pragma once

#include <string>

#include "minitrace.h"  // NOLINT

#define OVMS_PROFILE_SCOPE(name) MTR_SCOPE("OVMS", name);
#define OVMS_PROFILE_SCOPE_S(name, vname, cstr) MTR_SCOPE_S("OVMS", name, vname, cstr);
#define OVMS_PROFILE_FUNCTION() OVMS_PROFILE_SCOPE(__PRETTY_FUNCTION__)

#define OVMS_PROFILE_SYNC_BEGIN(name) MTR_BEGIN("OVMS", name);
#define OVMS_PROFILE_SYNC_END(name) MTR_END("OVMS", name);
#define OVMS_PROFILE_SYNC_BEGIN_S(name, vname, cstr) MTR_BEGIN_S("OVMS", name, vname, cstr);
#define OVMS_PROFILE_SYNC_END_S(name, vname, cstr) MTR_END_S("OVMS", name, vname, cstr);

#define OVMS_PROFILE_ASYNC_BEGIN(name, id) MTR_START("OVMS", name, id);
#define OVMS_PROFILE_ASYNC_END(name, id) MTR_FINISH("OVMS", name, id);

namespace ovms {

bool profiler_init(const char* file_path);
void profiler_shutdown();

class Profiler {
public:
    Profiler(const std::string& file_path);
    ~Profiler();
    bool isInitialized() const;

private:
    bool initialized = false;
};

}  // namespace ovms
