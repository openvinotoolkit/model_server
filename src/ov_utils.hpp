//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <map>
#include <memory>
#include <string>

#include <inference_engine.hpp>
#include <spdlog/spdlog.h>

#include "modelconfig.hpp"
#include "status.hpp"

namespace ovms {

class TensorInfo;

Status createSharedBlob(InferenceEngine::Blob::Ptr& destinationBlob, InferenceEngine::TensorDesc tensorDesc);

std::string getNetworkInputsInfoString(const InferenceEngine::InputsDataMap& inputsInfo, const ModelConfig& config);
std::string getTensorMapString(const std::map<std::string, std::shared_ptr<TensorInfo>>& tensorMap);
const InferenceEngine::SizeVector& getEffectiveShape(InferenceEngine::TensorDesc& desc);
const InferenceEngine::SizeVector& getEffectiveBlobShape(const InferenceEngine::Blob::Ptr& blob);

template <typename T>
Status blobClone(InferenceEngine::Blob::Ptr& destinationBlob, const T sourceBlob) {
    auto& description = sourceBlob->getTensorDesc();
    auto status = createSharedBlob(destinationBlob, description);
    if (!status.ok()) {
        return status;
    }

    if (destinationBlob->byteSize() != sourceBlob->byteSize()) {
        destinationBlob = nullptr;
        return StatusCode::OV_CLONE_BLOB_ERROR;
    }
    std::memcpy(InferenceEngine::as<InferenceEngine::MemoryBlob>(destinationBlob)->wmap().as<void*>(), (const void*)InferenceEngine::as<InferenceEngine::MemoryBlob>(sourceBlob)->rmap(), sourceBlob->byteSize());
    return StatusCode::OK;
}
}  // namespace ovms
