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

#include "../modelversion.hpp"
#include "../precision.hpp"
#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"

using KFSServerMetadataRequest = inference::ServerMetadataRequest;
using KFSServerMetadataResponse = inference::ServerMetadataResponse;
using KFSModelMetadataRequest = inference::ModelMetadataRequest;
using KFSModelMetadataResponse = inference::ModelMetadataResponse;
using KFSRequest = inference::ModelInferRequest;
using KFSResponse = inference::ModelInferResponse;
using KFSStreamResponse = inference::ModelStreamInferResponse;
using KFSServerReaderWriter = ::grpc::ServerReaderWriterInterface<KFSStreamResponse, KFSRequest>;
using KFSTensorInputProto = inference::ModelInferRequest::InferInputTensor;
using KFSTensorOutputProto = inference::ModelInferResponse::InferOutputTensor;
using KFSShapeType = google::protobuf::RepeatedField<int64_t>;
using KFSGetModelStatusRequest = inference::ModelReadyRequest;
using KFSGetModelStatusResponse = inference::ModelReadyResponse;
using KFSDataType = std::string;
using KFSInputTensorIteratorType = google::protobuf::internal::RepeatedPtrIterator<const ::inference::ModelInferRequest_InferInputTensor>;
using KFSOutputTensorIteratorType = google::protobuf::internal::RepeatedPtrIterator<const ::inference::ModelInferResponse_InferOutputTensor>;
namespace ovms {
class Status;
std::string tensorShapeToString(const KFSShapeType& tensorShape);

Precision KFSPrecisionToOvmsPrecision(const KFSDataType& s);
const KFSDataType& ovmsPrecisionToKFSPrecision(Precision precision);

size_t KFSDataTypeSize(const KFSDataType& datatype);
Status prepareConsolidatedTensorImpl(KFSResponse* response, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape, char*& tensorOut, size_t size);
const std::string& getRequestServableName(const KFSRequest& request);
Status isNativeFileFormatUsed(const KFSRequest& request, const std::string& name, bool& nativeFileFormatUsed);
bool isNativeFileFormatUsed(const KFSTensorInputProto& proto);
bool requiresPreProcessing(const KFSTensorInputProto& proto);
std::string& createOrGetString(KFSTensorOutputProto& proto, int index);
void setBatchSize(KFSTensorOutputProto& proto, int64_t batch);
void setStringPrecision(KFSTensorOutputProto& proto);
Status getRawInputContentsBatchSizeAndWidth(const std::string& buffer, int32_t& batchSize, size_t& width);
/**
 * Check if request is using only one of:
 * - request.raw_input_content
 * - request.inputs[i].content
 */
Status validateRequestCoherencyKFS(const KFSRequest& request, const std::string servableName, model_version_t servableVersion);
}  // namespace ovms
