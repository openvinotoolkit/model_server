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

#include <cstddef>

namespace ov {
class Tensor;
}
namespace ovms {
namespace image_utils {
// The plugin and host are distributed together and must use the same OpenVINO ABI.
using DecodeOpenCvV1 = int (*)(const unsigned char*, size_t, ov::Tensor*, char*, size_t);
}  // namespace image_utils
}  // namespace ovms
