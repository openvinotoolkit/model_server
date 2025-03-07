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

#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011 4457 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../precision.hpp"
#include "../tensor_conversion_common.hpp"

using TFSDataType = tensorflow::DataType;
using TFSPredictRequest = tensorflow::serving::PredictRequest;
using TFSPredictResponse = tensorflow::serving::PredictResponse;
using TFSInputTensorType = tensorflow::TensorProto;

namespace ovms {
class Status;

Precision TFSPrecisionToOvmsPrecision(const TFSDataType& s);
TFSDataType getPrecisionAsDataType(Precision precision);
std::string getDataTypeAsString(TFSDataType dataType);

std::string tensorShapeToString(const tensorflow::TensorShapeProto& tensorShape);
Status prepareConsolidatedTensorImpl(TFSPredictResponse* response, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape, char*& bufferOut, size_t size);
const std::string& getRequestServableName(const TFSPredictRequest& request);
Status isNativeFileFormatUsed(const TFSPredictRequest& request, const std::string& name, bool& isNativeFileFormatUsed);
bool isNativeFileFormatUsed(const TFSInputTensorType& request);
bool requiresPreProcessing(const TFSInputTensorType& proto);
std::string& createOrGetString(TFSInputTensorType& proto, int index);
void setBatchSize(TFSInputTensorType& proto, int64_t batch);
void setStringPrecision(TFSInputTensorType& proto);
const std::string& getBinaryInput(const tensorflow::TensorProto& tensor, size_t i);
int getBinaryInputsSize(const tensorflow::TensorProto& tensor);
Status validateTensor(const TensorInfo& tensorInfo,
    const tensorflow::TensorProto& src,
    const std::string* buffer);
Status convertBinaryExtensionStringFromBufferToNativeOVTensor(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::string* buffer);
}  // namespace ovms
