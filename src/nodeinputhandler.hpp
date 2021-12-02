//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <openvino/openvino.hpp>

#include "session_id.hpp"
#include "status.hpp"
#include "tensormap.hpp"

namespace ovms {

class NodeInputHandler {
protected:
    TensorMap inputTensors;
    uint32_t remainingDependencies;
    bool isUsed = false;

public:
    NodeInputHandler(uint32_t inputsMissingCount);
    virtual Status setInput(const std::string& inputName, std::shared_ptr<ov::runtime::Tensor>& blobPtr, session_id_t shardId);
    const TensorMap& getInputs() {
        isUsed = true;
        return inputTensors;
    }
    void clearInputs();
    bool isReady();
    virtual Status notifyFinishedDependency();
    virtual ~NodeInputHandler() = default;
};
}  // namespace ovms
