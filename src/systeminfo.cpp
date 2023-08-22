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
#include "systeminfo.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include <openvino/core/parallel.hpp>

#include "logging.hpp"
#include "status.hpp"

namespace ovms {
uint16_t getCoreCount() {
    // return parallel_get_num_threads();
    return std::thread::hardware_concurrency();
}
}  // namespace ovms
