//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include "servablemetadata.hpp"

namespace ovms {
ServableMetadata::ServableMetadata(const std::string& name,
    model_version_t version,
    const tensor_map_t& inputsInfo,
    const tensor_map_t& outputsInfo) :
    name(name),
    version(version),
    inputsInfo(inputsInfo),
    outputsInfo(outputsInfo) {
    for (auto& [key, tensorInfo] : this->inputsInfo) {
        auto& inDimsMin = this->inDimMin[key];
        auto& inDimsMax = this->inDimMax[key];
        for (auto& dim : tensorInfo->getShape()) {
            inDimsMin.emplace_back(dim.isAny() ? -1 : dim.getLowerBound());
            inDimsMax.emplace_back(dim.isAny() ? -1 : dim.getUpperBound());
        }
    }
    for (auto& [key, tensorInfo] : this->outputsInfo) {
        auto& outDimsMin = this->outDimMin[key];
        auto& outDimsMax = this->outDimMax[key];
        for (auto& dim : tensorInfo->getShape()) {
            outDimsMin.emplace_back(dim.isAny() ? -1 : dim.getLowerBound());
            outDimsMax.emplace_back(dim.isAny() ? -1 : dim.getUpperBound());
        }
    }
}

ServableMetadata::ServableMetadata(const std::string& name,
    model_version_t version,
    tensor_map_t&& inputsInfo,
    tensor_map_t&& outputsInfo) :
    name(name),
    version(version),
    inputsInfo(inputsInfo),
    outputsInfo(outputsInfo) {
    for (auto& [key, tensorInfo] : this->inputsInfo) {
        auto& inDimsMin = this->inDimMin[key];
        auto& inDimsMax = this->inDimMax[key];
        for (auto& dim : tensorInfo->getShape()) {
            inDimsMin.emplace_back(dim.getLowerBound());
            inDimsMax.emplace_back(dim.getUpperBound());
        }
    }
    for (auto& [key, tensorInfo] : this->outputsInfo) {
        auto& outDimsMin = this->outDimMin[key];
        auto& outDimsMax = this->outDimMax[key];
        for (auto& dim : tensorInfo->getShape()) {
            outDimsMin.emplace_back(dim.getLowerBound());
            outDimsMax.emplace_back(dim.getUpperBound());
        }
    }
}
}  // namespace ovms
