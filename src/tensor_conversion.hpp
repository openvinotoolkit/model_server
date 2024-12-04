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
#include <opencv2/opencv.hpp>

#include "precision.hpp"
#include "tensorinfo.hpp"

namespace ovms {
class Status;
template <typename TensorType>
Status convertStringRequestToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const TensorInfo& tensorInfo, const std::string* buffer);

template <typename TensorType>
Status convertStringRequestToOVTensor2D(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, TensorType& dst);


namespace tensor_conversion {
int getMatTypeFromTensorPrecision(ovms::Precision tensorPrecision);
bool isPrecisionEqual(int matPrecision, ovms::Precision tensorPrecision);
Status validateInput(const TensorInfo& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage, bool enforceResolutionAlignment);
bool checkBatchSizeMismatch(const TensorInfo& tensorInfo, const int batchSize);
Status validateResolutionAgainstFirstBatchImage(const cv::Mat input, cv::Mat* firstBatchImage);
Status validateNumberOfChannels(const TensorInfo& tensorInfo, const cv::Mat input, cv::Mat* firstBatchImage);
Status resizeMat(const cv::Mat& src, cv::Mat& dst, const dimension_value_t height, const dimension_value_t width);
bool resizeNeeded(const cv::Mat& image, const dimension_value_t height, const dimension_value_t width);
Status validateLayout(const TensorInfo& tensorInfo);
Status convertPrecision(const cv::Mat& src, cv::Mat& dst, const ovms::Precision requestedPrecision);
cv::Mat convertStringToMat(const std::string& image);
int getNumberOfInputs(const std::string* buffer);
Dimension getTensorInfoHeightDim(const TensorInfo& tensorInfo);
Dimension getTensorInfoWidthDim(const TensorInfo& tensorInfo);
void updateTargetResolution(Dimension& height, Dimension& width, const cv::Mat& image);
bool isResizeSupported(const TensorInfo& tensorInfo);
Status getInputs(const std::string* buffer, std::vector<std::string>& inputs);
}  // namespace tensor_conversion
}  // namespace ovms
