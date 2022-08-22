//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "../precision.hpp"

namespace ovms {
tensorflow::DataType getPrecisionAsDataType(Precision precision);
std::string getDataTypeAsString(tensorflow::DataType dataType);
std::string tensorShapeToString(const tensorflow::TensorShapeProto& tensorShape);
//   static std::string tensorShapeToString(const google::protobuf::RepeatedField<int64_t>& tensorShape);
}  // namespace ovms
