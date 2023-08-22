//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <stdint.h>

namespace ovms {
/**
 * @brief Get cpu core count on system. This can be limited by the container environment. In case of failure reading system constraints it will return total number of available cores. If it won't work the function will return 1
 * @return uint16_t Available number of cores in the system
 */
uint16_t getCoreCount();
}  // namespace ovms
