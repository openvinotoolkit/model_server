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
#include "deserialization_common.hpp"

#include <chrono>
#include <openvino/runtime/infer_request.hpp>

#include "itensorfactory.hpp"
#include "logging.hpp"

namespace ovms {

template <>
Status InputSink<ov::InferRequest&>::give(const std::string& name, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    try {
        OV_LOGGER("ov::InferRequest: {}, request.set_tensor({}, tensor: {})", reinterpret_cast<void*>(&requester), name, reinterpret_cast<void*>(&tensor));
        const bool timingEnabled = spdlog::default_logger_raw()->should_log(spdlog::level::debug);
        const auto setTensorStart = timingEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        requester.set_tensor(name, tensor);
        if (timingEnabled) {
            const auto setTensorMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - setTensorStart).count();
            SPDLOG_DEBUG("Inference executor InputSink::give name={} set_tensor_ms={:.6f}", name, setTensorMs);
        }
        // OV implementation the ov::Exception is not
        // a base class for all other exceptions thrown from OV.
        // OV can throw exceptions derived from std::logic_error.
    } catch (const ov::Exception& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    }
    return status;
}
}  // namespace ovms
