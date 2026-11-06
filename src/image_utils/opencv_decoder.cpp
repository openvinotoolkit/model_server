//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "opencv_decoder_api.hpp"

#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>

#include <openvino/runtime/tensor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {
struct ImageAllocator {
    std::shared_ptr<cv::Mat> image;
    void* allocate(size_t bytes, size_t) const {
        if (bytes != image->total() * image->elemSize())
            throw std::runtime_error("Unexpected decoded image allocation size");
        return image->data;
    }
    void deallocate(void*, size_t, size_t) noexcept {}
    bool is_equal(const ImageAllocator& other) const { return image == other.image; }
};
}  // namespace

#ifdef _WIN32
#define OVMS_IMAGE_EXPORT __declspec(dllexport)
#else
#define OVMS_IMAGE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" OVMS_IMAGE_EXPORT int ovmsDecodeOpenCvV1(
    const unsigned char* bytes, size_t size, ov::Tensor* tensor, char* error, size_t errorSize) {
    try {
        if (size == 0 || size > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Invalid encoded image size");
        cv::Mat buffer(1, static_cast<int>(size), CV_8U, const_cast<unsigned char*>(bytes));
        auto image = std::make_shared<cv::Mat>(cv::imdecode(buffer, cv::IMREAD_UNCHANGED));
        if (image->empty())
            throw std::runtime_error("OpenCV could not decode the image");
        ov::element::Type precision;
        switch (image->depth()) {
        case CV_8U:
            precision = ov::element::u8;
            break;
        case CV_16U:
            precision = ov::element::u16;
            break;
        case CV_32F:
            precision = ov::element::f32;
            break;
        case CV_64F:
            precision = ov::element::f64;
            break;
        default:
            throw std::runtime_error("Unsupported OpenCV decoded image depth");
        }
        if (!image->isContinuous())
            *image = image->clone();
        *tensor = ov::Tensor(precision,
            ov::Shape{1, static_cast<size_t>(image->rows), static_cast<size_t>(image->cols), static_cast<size_t>(image->channels())},
            ImageAllocator{image});
        return 1;
    } catch (const std::exception& exception) {
        std::snprintf(error, errorSize, "%s", exception.what());
        return 0;
    }
}
