#pragma once
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
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <spdlog/spdlog.h>
#include <sys/types.h>

#include "logging.hpp"
#include "openvino/runtime/remote_tensor.hpp"
namespace ovms {
cl_context getOCLContext();
cl_context get_cl_context(cl_platform_id& platformId, cl_device_id& deviceId);

}  // namespace ovms
