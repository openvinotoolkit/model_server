//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <openvino/runtime/tensor.hpp>

namespace ovms {

ov::Tensor loadImageStbi(unsigned char* image, const int x, const int y, const int desiredChannels);
ov::Tensor loadImageStbiFromMemory(const std::string& imageBytes);
ov::Tensor loadImageStbiFromFile(const char* filename);
std::string saveImageStbi(ov::Tensor tensor);

}  // namespace ovms
