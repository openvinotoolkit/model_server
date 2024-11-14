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
#include "outputkeeper.hpp"

#include "logging.hpp"

namespace ovms {
        OutputKeeper::OutputKeeper(ov::InferRequest& request, const tensor_map_t& outputsInfo) :
        request(request) {
        for (auto [name, _] : outputsInfo) {
            OV_LOGGER("ov::InferRequest: {}, request.get_tensor({})", reinterpret_cast<void*>(&request), name);
            try {
                ov::Tensor tensor = request.get_tensor(name);
                OV_LOGGER("ov::Tensor(): {}", reinterpret_cast<void*>(&tensor));
                outputs.emplace(std::make_pair(name, std::move(tensor)));
                OV_LOGGER("ov::Tensor(ov::Tensor&&): {}", reinterpret_cast<void*>(&outputs.at(name)));
            } catch (std::exception& e) {
                SPDLOG_DEBUG("Resetting output:{}; for this model  is not supported. Check C-API documentation for OVMS_InferenceRequestOutputSetData. Error:", name, e.what());
            }
        }
    }
        OutputKeeper::~OutputKeeper() {
        for (auto [name, v] : outputs) {
            OV_LOGGER("ov::InferRequest: {}, request.set_tensor({}, {})", reinterpret_cast<void*>(&request), name, reinterpret_cast<void*>(&v));
            request.set_tensor(name, v);
        }
    }
}  // namespace ovms
