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
#include "node_library_utils.hpp"

#include <utility>

#include "tensorinfo.hpp"

namespace ovms {

CustomNodeTensorPrecision toCustomNodeTensorPrecision(InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        return CustomNodeTensorPrecision::FP32;
    case InferenceEngine::Precision::FP16:
        return CustomNodeTensorPrecision::FP16;
    case InferenceEngine::Precision::U8:
        return CustomNodeTensorPrecision::U8;
    case InferenceEngine::Precision::I8:
        return CustomNodeTensorPrecision::I8;
    case InferenceEngine::Precision::I16:
        return CustomNodeTensorPrecision::I16;
    case InferenceEngine::Precision::U16:
        return CustomNodeTensorPrecision::U16;
    case InferenceEngine::Precision::I32:
        return CustomNodeTensorPrecision::I32;
    default:
        return CustomNodeTensorPrecision::UNSPECIFIED;
    }
}

InferenceEngine::Precision toInferenceEnginePrecision(CustomNodeTensorPrecision precision) {
    switch (precision) {
    case CustomNodeTensorPrecision::FP32:
        return InferenceEngine::Precision::FP32;
    case CustomNodeTensorPrecision::FP16:
        return InferenceEngine::Precision::FP16;
    case CustomNodeTensorPrecision::U8:
        return InferenceEngine::Precision::U8;
    case CustomNodeTensorPrecision::I8:
        return InferenceEngine::Precision::I8;
    case CustomNodeTensorPrecision::I16:
        return InferenceEngine::Precision::I16;
    case CustomNodeTensorPrecision::U16:
        return InferenceEngine::Precision::U16;
    case CustomNodeTensorPrecision::I32:
        return InferenceEngine::Precision::I32;
    default:
        return InferenceEngine::Precision::UNSPECIFIED;
    }
}

std::unique_ptr<struct CustomNodeParam[]> createCustomNodeParamArray(const std::unordered_map<std::string, std::string>& paramMap) {
    if (paramMap.size() == 0) {
        return nullptr;
    }
    auto libraryParameters = std::make_unique<struct CustomNodeParam[]>(paramMap.size());
    int i = 0;
    for (const auto& [key, value] : paramMap) {
        libraryParameters[i].key = key.c_str();
        libraryParameters[i].value = value.c_str();
        i++;
    }
    return std::move(libraryParameters);
}

std::unique_ptr<struct CustomNodeTensor[]> createCustomNodeTensorArray(const std::unordered_map<std::string, InferenceEngine::Blob::Ptr>& blobMap) {
    if (blobMap.size() == 0) {
        return nullptr;
    }
    auto inputTensors = std::make_unique<struct CustomNodeTensor[]>(blobMap.size());
    int i = 0;
    for (const auto& [name, blob] : blobMap) {
        inputTensors[i].name = static_cast<const char*>(name.c_str());
        inputTensors[i].data = static_cast<uint8_t*>(blob->buffer());
        inputTensors[i].dataLength = static_cast<uint64_t>(blob->byteSize());
        inputTensors[i].dims = static_cast<uint64_t*>(blob->getTensorDesc().getDims().data());
        inputTensors[i].dimsLength = static_cast<uint64_t>(blob->getTensorDesc().getDims().size());
        inputTensors[i].precision = toCustomNodeTensorPrecision(blob->getTensorDesc().getPrecision());
        i++;
    }
    return std::move(inputTensors);
}

Status createTensorInfoMap(struct CustomNodeTensorInfo* info, int infoLength, std::map<std::string, std::shared_ptr<TensorInfo>>& out, release_fn freeCallback) {
    if (info == nullptr) {
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED;
    }
    if (infoLength <= 0) {
        freeCallback(info);
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT;
    }
    // At this point it is important to not exit before we iterate over every info object.
    // This is due to a fact that we need to ensure to free resources allocated by shared library using freeCallback.
    for (int i = 0; i < infoLength; i++) {
        if (info[i].dims == nullptr) {
            continue;
        }
        if (info[i].dimsLength == 0) {
            freeCallback(info[i].dims);
            continue;
        }
        if (info[i].name == nullptr) {
            continue;
        }

        std::string name = std::string(info[i].name);
        InferenceEngine::Precision precision = toInferenceEnginePrecision(info[i].precision);
        shape_t shape(info[i].dims, info[i].dims + info[i].dimsLength);

        freeCallback(info[i].dims);

        out.emplace(name, std::make_shared<TensorInfo>(name, precision, std::move(shape)));
    }
    freeCallback(info);
    return StatusCode::OK;
}

}  // namespace ovms
