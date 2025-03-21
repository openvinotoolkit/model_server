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
#pragma warning(push)
#pragma warning(disable : 6269 6294 6201)
#include "opencv2/opencv.hpp"
#pragma warning(pop)
#include "profiler.hpp"
#include "status.hpp"

namespace ovms {
namespace tensor_conversion {
Status resizeMat(const cv::Mat& src, cv::Mat& dst, const dimension_value_t height, const dimension_value_t width) {
    OVMS_PROFILE_FUNCTION();
    cv::resize(src, dst, cv::Size(width, height));
    return StatusCode::OK;
}
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

shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    shape_t dims;
    dims.push_back(images.size());
    if (tensorInfo.isInfluencedByDemultiplexer()) {
        dims.push_back(1);
    }
    dims.push_back(images[0].rows);
    dims.push_back(images[0].cols);
    dims.push_back(images[0].channels());
    return dims;
}

ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    ov::Shape shape = getShapeFromImages(images, tensorInfo);
    ov::element::Type precision = tensorInfo.getOvPrecision();
    ov::Tensor tensor(precision, shape);
    char* ptr = (char*)tensor.data();
    for (cv::Mat image : images) {
        memcpy(ptr, (char*)image.data, image.total() * image.elemSize());
        ptr += (image.total() * image.elemSize());
    }
    return tensor;
}

ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const TensorInfo& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    switch (tensorInfo.getPrecision()) {
    case ovms::Precision::FP32:
    case ovms::Precision::I32:
    case ovms::Precision::FP64:
    case ovms::Precision::I8:
    case ovms::Precision::U8:
    case ovms::Precision::FP16:
    case ovms::Precision::U16:
    case ovms::Precision::I16:
        return createTensorFromMats(images, tensorInfo);
    case ovms::Precision::MIXED:
    case ovms::Precision::Q78:
    case ovms::Precision::BIN:
    case ovms::Precision::BOOL:
    case ovms::Precision::CUSTOM:
    default:
        return ov::Tensor();
    }
}
Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage) {
    OVMS_PROFILE_FUNCTION();
    if (input.cols == firstBatchImage->cols && input.rows == firstBatchImage->rows) {
        return StatusCode::OK;
    }
    SPDLOG_DEBUG("Each binary image in request needs to have resolution matched. First cols: {}, rows: {}, current cols: {}, rows: {}",
        firstBatchImage->cols, firstBatchImage->rows, input.cols, input.rows);
    return StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH;
}
}  // namespace tensor_conversion
}  // namespace ovms
