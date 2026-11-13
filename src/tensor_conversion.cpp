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
#include "tensor_conversion.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include "predict_request_validation_utils_impl.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "status.hpp"

namespace ovms {
namespace tensor_conversion {
Status getInputs(const std::string* buffer, std::vector<std::string>& inputs) {
    if (buffer == nullptr) {
        return StatusCode::OK;
    }
    size_t offset = 0;
    while (offset + sizeof(uint32_t) <= buffer->size()) {
        uint64_t inputSize = *(reinterpret_cast<const uint32_t*>(buffer->data() + offset));
        offset += sizeof(uint32_t);
        if (offset + inputSize > buffer->size())
            break;
        inputs.push_back(buffer->substr(offset, inputSize));
        offset += inputSize;
    }
    if (offset != buffer->size()) {
        return StatusCode::IMAGE_PARSING_FAILED;
    }
    return StatusCode::OK;
}

static shape_t getShapeFromImages(const std::vector<ov::Tensor>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    shape_t dims;
    dims.push_back(images.size());
    if (tensorInfo.isInfluencedByDemultiplexer()) {
        dims.push_back(1);
    }
    dims.push_back(images[0].get_shape()[1]);
    dims.push_back(images[0].get_shape()[2]);
    dims.push_back(images[0].get_shape()[3]);
    return dims;
}

ov::Tensor createTensorFromImages(const std::vector<ov::Tensor>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    if (images.empty()) {
        return ov::Tensor();
    }
    ov::Shape shape = getShapeFromImages(images, tensorInfo);
    ov::element::Type precision = tensorInfo.getOvPrecision();
    ov::Tensor tensor(precision, shape);
    char* ptr = (char*)tensor.data();
    const size_t firstImageSizeBytes = images[0].get_byte_size();
    for (const auto& image : images) {
        const size_t imageSizeBytes = image.get_byte_size();
        if (imageSizeBytes != firstImageSizeBytes) {
            SPDLOG_DEBUG("Image conversion failed due to inconsistent image size in one batch. Expected bytes: {} current bytes: {}",
                firstImageSizeBytes,
                imageSizeBytes);
            return ov::Tensor();
        }
        memcpy(ptr, image.data(), imageSizeBytes);
        ptr += imageSizeBytes;
    }
    return tensor;
}

Status validateResolutionAgainstFirstBatchImage(const ov::Tensor& input, const ov::Tensor* firstBatchImage) {
    OVMS_PROFILE_FUNCTION();
    if (input.get_shape()[2] == firstBatchImage->get_shape()[2] && input.get_shape()[1] == firstBatchImage->get_shape()[1]) {
        return StatusCode::OK;
    }
    SPDLOG_DEBUG("Each binary image in request needs to have resolution matched. First cols: {}, rows: {}, current cols: {}, rows: {}",
        firstBatchImage->get_shape()[2], firstBatchImage->get_shape()[1], input.get_shape()[2], input.get_shape()[1]);
    return StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH;
}
}  // namespace tensor_conversion
}  // namespace ovms
