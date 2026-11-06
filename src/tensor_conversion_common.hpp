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

#include "tensorinfo.hpp"

namespace ovms {
class Status;
namespace tensor_conversion {
Status validateLayout(const TensorInfo& tensorInfo);
int getNumberOfInputs(const std::string* buffer);
bool checkBatchSizeMismatch(const TensorInfo& tensorInfo, const int batchSize);
bool isSupportedImagePrecision(ovms::Precision tensorPrecision);
Status validateInput(const TensorInfo& tensorInfo, const ov::Tensor& input, const ov::Tensor* firstBatchImage, bool enforceResolutionAlignment);
Status validateNumberOfChannels(const TensorInfo& tensorInfo,
    const ov::Tensor& input,
    const ov::Tensor* firstBatchImage);
Status validateResolutionAgainstFirstBatchImage(const ov::Tensor& input, const ov::Tensor* firstBatchImage);
Status checkEstimatedImageSize(std::string_view encodedImage, const std::string& inputName,
    size_t alreadyAllocatedPixels, size_t maxAllowedImagePixels);
Status accumulateAndCheckDecodedImageSize(const ov::Tensor& image, const TensorInfo& tensorInfo,
    size_t& totalAllocatedPixels, size_t maxAllowedImagePixels);
Dimension getTensorInfoHeightDim(const TensorInfo& tensorInfo);
Dimension getTensorInfoWidthDim(const TensorInfo& tensorInfo);
void updateTargetResolution(Dimension& height, Dimension& width, const ov::Tensor& image);
bool isResizeSupported(const TensorInfo& tensorInfo);
Status getInputs(const std::string* buffer, std::vector<std::string>& inputs);
ov::Tensor createTensorFromImages(const std::vector<ov::Tensor>& images, const TensorInfo& tensorInfo);
}  // namespace tensor_conversion
}  // namespace ovms
