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
#include <string>
#include <unordered_map>
#include <utility>

#include "packettypes.hpp"
namespace ovms {
extern const std::string KFS_REQUEST_PREFIX;
extern const std::string KFS_RESPONSE_PREFIX;
extern const std::string MP_TENSOR_PREFIX;
extern const std::string TF_TENSOR_PREFIX;
extern const std::string TFLITE_TENSOR_PREFIX;
extern const std::string OV_TENSOR_PREFIX;
extern const std::string OVMS_PY_TENSOR_PREFIX;
extern const std::string MP_IMAGE_PREFIX;

std::pair<std::string, mediapipe_packet_type_enum> getStreamNamePair(const std::string& streamFullName);
std::string getStreamName(const std::string& streamFullName);
}  // namespace ovms
