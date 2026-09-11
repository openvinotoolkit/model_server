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
#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <string>

#include "deps/opencv.hpp"
#include "tensorinfo.hpp"

namespace ovms {
class Status;
namespace tensor_conversion {
Status validateLayout(const TensorInfo& tensorInfo);
int getNumberOfInputs(const std::string* buffer);
bool checkBatchSizeMismatch(const TensorInfo& tensorInfo, const int batchSize);
int getMatTypeFromTensorPrecision(ovms::Precision tensorPrecision);
bool isPrecisionEqual(int matPrecision, ovms::Precision tensorPrecision);
cv::Mat convertStringToMat(const std::string& image);
Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const ovms::Precision requestedPrecision);
Status validateLayout(const TensorInfo& tensorInfo);
bool resizeNeeded(const cv::Mat& image, const dimension_value_t height, const dimension_value_t width);
Status validateInput(const TensorInfo& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage, bool enforceResolutionAlignment);
Status resizeMat(const cv::Mat& src, cv::Mat& dst, const dimension_value_t height, const dimension_value_t width);
Status validateNumberOfChannels(const TensorInfo& tensorInfo,
    const cv::Mat input,
    cv::Mat* firstBatchImage);
Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage);  // Rejects, before decoding, an encoded image whose header-estimated decoded size would blow the
// per-request byte budget (accounting for what previous images already reserved). Logs on
// rejection. Fails open when the format cannot be estimated. Arithmetic is overflow-safe.
Status checkEstimatedImageSize(std::string_view encodedImage, const std::string& inputName,
    size_t alreadyAllocatedBytes, size_t maxAllowedImageBytes);
// Adds a decoded image's byte cost (decoded pixels plus any precision-converted copy) to the
// running per-request total, rejecting if the budget is exceeded. Logs on rejection. Arithmetic
// is overflow-safe. On success totalAllocatedBytes is advanced by this image's cost.
Status accumulateAndCheckDecodedImageSize(const cv::Mat& image, const TensorInfo& tensorInfo,
    size_t& totalAllocatedBytes, size_t maxAllowedImageBytes);  /////////////////////////////////
}  // namespace tensor_conversion
/////////////////////////////////
Dimension getTensorInfoHeightDim(const TensorInfo& tensorInfo);
void updateTargetResolution(Dimension& height, Dimension& width, const cv::Mat& image);
bool isResizeSupported(const TensorInfo& tensorInfo);
shape_t getShapeFromImages(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo);
ov::Tensor createTensorFromMats(const std::vector<cv::Mat>& images, const TensorInfo& tensorInfo);
ov::Tensor convertMatsToTensor(std::vector<cv::Mat>& images, const TensorInfo& tensorInfo);
}  // namespace ovms
