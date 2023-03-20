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

#include "../ov_utils.hpp"
#include "../status.hpp"
#include "../tensorinfo.hpp"

namespace ovms {

CustomNodeTensorPrecision toCustomNodeTensorPrecision(ov::element::Type_t precision) {
    switch (precision) {
    case ov::element::Type_t::f32:
        return CustomNodeTensorPrecision::FP32;
    case ov::element::Type_t::f64:
        return CustomNodeTensorPrecision::FP64;
    case ov::element::Type_t::i32:
        return CustomNodeTensorPrecision::I32;
    case ov::element::Type_t::i64:
        return CustomNodeTensorPrecision::I64;
    case ov::element::Type_t::i8:
        return CustomNodeTensorPrecision::I8;
    case ov::element::Type_t::u8:
        return CustomNodeTensorPrecision::U8;
    case ov::element::Type_t::f16:
        return CustomNodeTensorPrecision::FP16;
    case ov::element::Type_t::i16:
        return CustomNodeTensorPrecision::I16;
    case ov::element::Type_t::u16:
        return CustomNodeTensorPrecision::U16;
    default:
        return CustomNodeTensorPrecision::UNSPECIFIED;
    }
}

Precision toInferenceEnginePrecision(CustomNodeTensorPrecision precision) {
    static std::unordered_map<CustomNodeTensorPrecision, Precision> precisionMap{
        {CustomNodeTensorPrecision::FP32, Precision::FP32},
        {CustomNodeTensorPrecision::FP64, Precision::FP64},
        {CustomNodeTensorPrecision::I32, Precision::I32},
        {CustomNodeTensorPrecision::I64, Precision::I64},
        {CustomNodeTensorPrecision::I8, Precision::I8},
        {CustomNodeTensorPrecision::U8, Precision::U8},
        {CustomNodeTensorPrecision::FP16, Precision::FP16},
        {CustomNodeTensorPrecision::I16, Precision::I16},
        {CustomNodeTensorPrecision::U16, Precision::U16}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
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
    return libraryParameters;
}

std::unique_ptr<struct CustomNodeTensor[]> createCustomNodeTensorArray(const TensorMap& tensorMap, const std::unordered_map<std::string, shape_t>& tensorsDims) {
    if (tensorMap.size() == 0) {
        return nullptr;
    }
    auto inputTensors = std::make_unique<struct CustomNodeTensor[]>(tensorMap.size());
    int i = 0;
    for (const auto& [name, tensor] : tensorMap) {
        auto dimsIt = tensorsDims.find(name);
        if (dimsIt == tensorsDims.end()) {
            return nullptr;
        }
        static_assert(sizeof(size_t) == sizeof(uint64_t));
        const auto& dims = dimsIt->second;
        inputTensors[i].name = static_cast<const char*>(name.c_str());
        inputTensors[i].data = static_cast<uint8_t*>(tensor.data());
        inputTensors[i].dataBytes = static_cast<uint64_t>(tensor.get_byte_size());
        inputTensors[i].dims = const_cast<uint64_t*>(dims.data());
        inputTensors[i].dimsCount = static_cast<uint64_t>(dims.size());
        inputTensors[i].precision = toCustomNodeTensorPrecision(tensor.get_element_type());
        i++;
    }
    return inputTensors;
}

Status createTensorInfoMap(struct CustomNodeTensorInfo* info, int infoCount, std::map<std::string, std::shared_ptr<const TensorInfo>>& out, release_fn freeCallback, void* customNodeLibraryInternalManager) {
    if (info == nullptr) {
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED;
    }
    if (infoCount <= 0) {
        freeCallback(info, customNodeLibraryInternalManager);
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT;
    }
    // At this point it is important to not exit before we iterate over every info object.
    // This is due to a fact that we need to ensure to free resources allocated by shared library using freeCallback.
    for (int i = 0; i < infoCount; i++) {
        if (info[i].dims == nullptr) {
            continue;
        }
        if (info[i].dimsCount == 0) {
            freeCallback(info[i].dims, customNodeLibraryInternalManager);
            continue;
        }
        if (info[i].name == nullptr) {
            continue;
        }

        std::string name = std::string(info[i].name);
        auto precision = toInferenceEnginePrecision(info[i].precision);
        ovms::Shape shape;
        for (uint64_t j = 0; j < info[i].dimsCount; ++j) {
            auto dim = info[i].dims[j];
            shape.add(dim ? Dimension(dim) : Dimension::any());
        }

        freeCallback(info[i].dims, customNodeLibraryInternalManager);
        out.emplace(name, std::make_shared<TensorInfo>(name, precision, std::move(shape), Layout::getUnspecifiedLayout()));
    }
    freeCallback(info, customNodeLibraryInternalManager);
    return StatusCode::OK;
}

}  // namespace ovms
