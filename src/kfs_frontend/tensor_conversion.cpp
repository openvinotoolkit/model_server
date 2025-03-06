//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "kfs_utils.hpp"
#include "../tensor_conversion.hpp"
#include "../tensor_conversion_common.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "kfs_utils.hpp"
#include "../logging.hpp"
#pragma warning(push)
#pragma warning(disable : 6269 6294 6201)
#include <opencv2/opencv.hpp>
#pragma warning(pop)
#include "../profiler.hpp"
#include "../status.hpp"

namespace ovms {
Status convertStringRequestFromBufferToOVTensor2D(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer) {
    size_t batchSize = 0;
    size_t offset = 0;
    size_t maxStringLength = 0;
    while (offset + sizeof(uint32_t) <= buffer->size()) {
        uint64_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer->data() + offset));
        offset += (sizeof(uint32_t) + inputSize);
        maxStringLength = std::max(maxStringLength, inputSize);
        batchSize++;
    }
    if (offset != buffer->size()) {
        SPDLOG_DEBUG("Input string format conversion failed");
        return StatusCode::INVALID_STRING_INPUT;
    }
    size_t width = maxStringLength + 1;
    offset = 0;
    tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape{batchSize, width});
    for (size_t i = 0; i < batchSize; i++) {
        uint64_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer->data() + offset));
        offset += sizeof(uint32_t);
        auto data = tensor.data<unsigned char>() + i * width;
        std::memcpy(data, reinterpret_cast<const unsigned char*>(buffer->data() + offset), inputSize);
        for (size_t j = inputSize; j < width; j++) {
            data[j] = 0;
        }
        offset += inputSize;
    }
    return StatusCode::OK;
}

template Status convertStringRequestToOVTensor<::KFSRequest::InferInputTensor>(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer);
template Status convertNativeFileFormatRequestTensorToOVTensor<::KFSRequest::InferInputTensor>(const ::KFSRequest::InferInputTensor& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer);
}  // namespace ovms

// TODO we need to see declarations before @atobisze
#include "../tensor_conversion_after.hpp"

namespace ovms {
template Status convertStringRequestToOVTensor2D<KFSTensorInputProto>(const KFSTensorInputProto& src, ov::Tensor& tensor, const std::string* buffer);
template Status convertOVTensor2DToStringResponse<::KFSResponse::InferOutputTensor>(const ov::Tensor& tensor, ::KFSResponse::InferOutputTensor& dst);

}  // namespace ovms
