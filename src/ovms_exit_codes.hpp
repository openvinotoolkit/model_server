//*****************************************************************************
// Copyright 2024 Intel Corporation
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
// TODO: Write windows/linux specific status codes.
#ifdef __linux__
#include <sysexits.h>
#elif _WIN32
#include <ntstatus.h>
#endif

namespace ovms {

#ifdef __linux__
#define OVMS_EX_OK EX_OK
#define OVMS_EX_FAILURE 1
#define OVMS_EX_WARNING 2
#define OVMS_EX_USAGE EX_USAGE
#elif _WIN32
#define OVMS_EX_OK 0
#define OVMS_EX_FAILURE 1
#define OVMS_EX_WARNING 2
#define OVMS_EX_USAGE 3
#endif

}  // namespace ovms
