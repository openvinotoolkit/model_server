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
#include <string_view>
#include <vector>
#include <utility>

#include "logging.hpp"
#include "precision.hpp"
#include "predict_request_validation_utils_impl.hpp"
#include "profiler.hpp"
#include "tensorinfo.hpp"
#include "status.hpp"

namespace inference {
class ModelInferRequest_InferInputTensor;
}
namespace ovms {
class Status;
namespace tensor_conversion {
Status convertEncodedImagesToOVTensor(const std::vector<std::string_view>& encodedImages, ov::Tensor& tensor,
    const TensorInfo& tensorInfo, const std::string* buffer);
}
template <typename TensorType>
Status convertStringRequestToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer);

template <typename TensorType>
Status convertStringRequestToOVTensor2D(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, TensorType& dst);

template <typename TensorType>
Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    auto status = validateTensor(tensorInfo, src, buffer);
    if (status != StatusCode::OK) {
        SPDLOG_DEBUG("Input native file format validation failed");
        return status;
    }
    std::vector<std::string_view> encodedImages;
    if (buffer == nullptr) {
        const int numberOfInputs = getBinaryInputsSize(src);
        if (numberOfInputs > 0) {
            encodedImages.reserve(static_cast<size_t>(numberOfInputs));
        }
        for (int i = 0; i < numberOfInputs; ++i) {
            encodedImages.emplace_back(getBinaryInput(src, i));
        }
    }
    status = tensor_conversion::convertEncodedImagesToOVTensor(encodedImages, tensor, tensorInfo, buffer);
    if (!status.ok()) {
        SPDLOG_DEBUG("Input native file format conversion failed");
        return status;
    }
    if (!tensor) {
        SPDLOG_DEBUG("Input native file format conversion failed");
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}

class InferenceTensor;

template <>
Status convertNativeFileFormatRequestTensorToOVTensor(const InferenceTensor& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer);

template <>
Status convertStringRequestToOVTensor(const InferenceTensor& src, ov::Tensor& tensor, const std::string* buffer);

template <>
Status convertStringRequestToOVTensor(const ::inference::ModelInferRequest_InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, TensorType& dst) {
    if (tensor.get_shape().size() != 2) {
        return StatusCode::INTERNAL_ERROR;
    }
    if (tensor.get_element_type() != ov::element::Type_t::u8) {
        return StatusCode::INTERNAL_ERROR;
    }
    size_t batchSize = tensor.get_shape()[0];
    size_t maxStringLen = tensor.get_shape()[1];
    setBatchSize(dst, batchSize);
    setStringPrecision(dst);
    for (size_t i = 0; i < batchSize; i++) {
        const char* strStart = reinterpret_cast<const char*>(tensor.data<unsigned char>() + i * maxStringLen);
        size_t strLen = strnlen(strStart, maxStringLen);
        createOrGetString(dst, i).assign(strStart, strLen);
    }
    return StatusCode::OK;
}
template <>
Status convertStringRequestToOVTensor(const InferenceTensor& src, ov::Tensor& tensor, const std::string* buffer);
template <>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, InferenceTensor& dst);
}  // namespace ovms
