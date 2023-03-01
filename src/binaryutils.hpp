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
#include "kfs_frontend/kfs_utils.hpp"

namespace tensorflow {
class TensorProto;
}


namespace ovms {
class Status;
template <typename TensorType>
Status convertNativeFileFormatRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo, const std::string* buffer);

Status convertStringProtoToOVTensor(const inference::ModelInferRequest::InferInputTensor& src, ov::Tensor& tensor, const std::string* buffer);
Status convertStringProtoToOVTensor(const tensorflow::TensorProto& src, ov::Tensor& tensor);

Status convertOVTensorToStringProto(const ov::Tensor& tensor, tensorflow::TensorProto& dst);
Status convertOVTensorToStringProto(const ov::Tensor& tensor, ::inference::ModelInferResponse::InferOutputTensor& dst);

}  // namespace ovms
