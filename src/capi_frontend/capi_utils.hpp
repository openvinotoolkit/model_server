//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <vector>

#include "../predict_request_validation_utils.hpp"
#include "../ovms.h"  // NOLINT
#include "../precision.hpp"
#include "../shape.hpp"

namespace ovms {
class InferenceRequest;
class InferenceResponse;
class InferenceTensor;
class Status;

template <>
std::string tensorShapeToString(const signed_shape_t& tensorShape);

Precision getOVMSDataTypeAsPrecision(OVMS_DataType datatype);
OVMS_DataType getPrecisionAsOVMSDataType(Precision precision);

size_t DataTypeToByteSize(OVMS_DataType datatype);

const std::string& getRequestServableName(const ovms::InferenceRequest& request);

std::string& createOrGetString(InferenceTensor& proto, int index);

bool requiresPreProcessing(const InferenceTensor& tensor);

int getBinaryInputsSize(const InferenceTensor& tensor);
const std::string& getBinaryInput(const InferenceTensor& tensor, size_t i);
Status validateTensor(const TensorInfo& tensorInfo,
    const InferenceTensor& src,
    const std::string* buffer);

Status isNativeFileFormatUsed(const InferenceRequest& request, const std::string& name, bool& nativeFileFormatUsed);
}  // namespace ovms
