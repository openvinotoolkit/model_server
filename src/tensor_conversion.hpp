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

#include "tensorinfo.hpp"

namespace ovms {
class Status;
template <typename TensorType>
Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::shared_ptr<const TensorInfo>& tensorInfo, const std::string* buffer);

template <typename TensorType>
Status convertStringRequestToOVTensor2D(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertStringRequestToOVTensor1D(const TensorType& src, ov::Tensor& tensor, const std::string* buffer);

template <typename TensorType>
Status convertOVTensor2DToStringResponse(const ov::Tensor& tensor, TensorType& dst);

}  // namespace ovms
