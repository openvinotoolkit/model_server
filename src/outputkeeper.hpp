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

#include <unordered_map>
#include <string>

#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>

#include "tensorinfo.hpp"

namespace ovms {
struct OutputKeeper {
    std::unordered_map<std::string, ov::Tensor> outputs;
    ov::InferRequest& request;
    bool cancelled{false};
    OutputKeeper(ov::InferRequest& request, const tensor_map_t& outputsInfo);
    ~OutputKeeper();
    // Disable the output-tensor restore performed by the destructor. Used on the async
    // start_async() error path, where inference never ran (so there is nothing to restore)
    // and reentering the InferRequest from the destructor would deadlock (#2871).
    void cancel() { this->cancelled = true; }
};
}  // namespace ovms
