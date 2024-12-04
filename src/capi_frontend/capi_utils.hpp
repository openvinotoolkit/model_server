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
#include "../status.hpp" // TODO move impl @atobisze

namespace ovms {
class InferenceRequest;
class InferenceResponse;
class InferenceTensor;
class Status;

template<>
std::string tensorShapeToString(const signed_shape_t& tensorShape);

OVMS_DataType getPrecisionAsOVMSDataType(Precision precision);
Precision getOVMSDataTypeAsPrecision(OVMS_DataType datatype);
size_t DataTypeToByteSize(OVMS_DataType datatype);

Status isNativeFileFormatUsed(const InferenceRequest& request, const std::string& name, bool& nativeFileFormatUsed);
const std::string& getRequestServableName(const ovms::InferenceRequest& request);
bool requiresPreProcessing(const InferenceTensor& tensor);
std::string& createOrGetString(InferenceTensor& proto, int index);
}  // namespace ovms
