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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <google/protobuf/map.h>

#include "status.hpp"
#include "shapeinfo.hpp"
#include "tensorinfo.hpp"

namespace ovms {

// TODO: Separate namespace?

Status validateNumberOfInputs_New(
    const tensorflow::serving::PredictRequest& request,
    const size_t expectedNumberOfInputs);

Status validateAndGetInput_New(
    const tensorflow::serving::PredictRequest& request,
    const std::string& name,
    google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator& it);

Status checkIfShapeValuesNegative_New(
    const tensorflow::TensorProto& proto);

Status validateNumberOfBinaryInputShapeDimensions_New(
    const tensorflow::TensorProto& proto);

Status checkBatchSizeMismatch_New(
    const tensorflow::TensorProto& proto,
    const size_t networkBatchSize,
    Status& finalStatus,
    Mode batchingMode = Mode::FIXED,
    Mode shapeMode = Mode::FIXED);

Status checkBinaryBatchSizeMismatch_New(
    const tensorflow::TensorProto& proto,
    const size_t networkBatchSize,
    Status& finalStatus,
    Mode batchingMode = Mode::FIXED,
    Mode shapeMode = Mode::FIXED);

Status checkShapeMismatch_New(
    const tensorflow::TensorProto& proto,
    const ovms::TensorInfo& inputInfo,
    Status& finalStatus,
    Mode batchingMode = Mode::FIXED,
    Mode shapeMode = Mode::FIXED);

Status validateTensorContentSize_New(
    const tensorflow::TensorProto& proto,
    InferenceEngine::Precision expectedPrecision);

Status validateNumberOfShapeDimensions_New(
    const ovms::TensorInfo& inputInfo,
    const tensorflow::TensorProto& proto);

Status validatePrecision_New(
    const tensorflow::TensorProto& proto,
    const ovms::TensorInfo& inputInfo);

}  // namespace ovms
