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

#include "../logging.hpp"
#include "../status.hpp"
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

#define MP_RETURN_ON_FAIL(code, message, errorCode)              \
    {                                                            \
        auto absStatus = code;                                   \
        if (!absStatus.ok()) {                                   \
            const std::string absMessage = absStatus.ToString(); \
            SPDLOG_DEBUG("{} {}", message, absMessage);          \
            return Status(errorCode, std::move(absMessage));     \
        }                                                        \
    }

#define OVMS_RETURN_ON_FAIL(code)                 \
    _Pragma("warning(push)")                      \
        _Pragma("warning(disable : 4456 6246)") { \
        auto status = code;                       \
        if (!status.ok()) {                       \
            return status;                        \
        }                                         \
    }                                             \
    _Pragma("warning(pop)")

#define OVMS_RETURN_MP_ERROR_ON_FAIL(code, message)                     \
    {                                                                   \
        auto status = code;                                             \
        if (!status.ok()) {                                             \
            SPDLOG_DEBUG("{} {}", message, status.string());            \
            return absl::Status(absl::StatusCode::kCancelled, message); \
        }                                                               \
    }

enum class MediaPipeStreamType { INPUT,
    OUTPUT };

std::pair<std::string, mediapipe_packet_type_enum> getStreamNamePair(const std::string& streamFullName, MediaPipeStreamType streamType);
std::string getStreamName(const std::string& streamFullName);
}  // namespace ovms
