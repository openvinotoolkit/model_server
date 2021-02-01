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

#include <inference_engine.hpp>

#include "blobmap.hpp"
#include "session_id.hpp"
#include "status.hpp"

namespace ovms {

using BlobMap = std::unordered_map<std::string, InferenceEngine::Blob::Ptr>;

class NodeInputHandler {
protected:
    BlobMap inputBlobs;
    uint32_t remainingDependencies;

public:
    NodeInputHandler(uint32_t inputsMissingCount);
    virtual void setInput(const std::string& inputName, InferenceEngine::Blob::Ptr& blobPtr, session_id_t shardId);
    virtual const BlobMap& getInputs() const { return inputBlobs; }
    void clearInputs();
    bool isReady();
    virtual Status notifyFinishedDependency();
    virtual ~NodeInputHandler() = default;
};
}  // namespace ovms
