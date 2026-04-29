//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "src/capi_frontend/inferencerequest.hpp"
#include "src/capi_frontend/inferenceresponse.hpp"
#include "test_utils.hpp"

using CAPIInterface = std::pair<ovms::InferenceRequest, ovms::InferenceResponse>;

void prepareCAPIInferInputTensor(ovms::InferenceRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, OVMS_DataType>& inputInfo,
    const std::vector<float>& data, uint32_t decrementBufferSize = 0, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> deviceId = std::nullopt);
void prepareCAPIInferInputTensor(ovms::InferenceRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const ovms::Precision>& inputInfo,
    const std::vector<float>& data, uint32_t decrementBufferSize = 0, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> deviceId = std::nullopt);

void preparePredictRequest(ovms::InferenceRequest& request, inputs_info_t requestInputs, const std::vector<float>& data,
    uint32_t decrementBufferSize = 0, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> deviceId = std::nullopt);

void prepareInferStringTensor(ovms::InferenceTensor& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content);
void prepareInferStringRequest(ovms::InferenceRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent = true);  // CAPI binary not supported

void prepareBinaryPredictRequest(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize);          // CAPI binary not supported
void prepareBinaryPredictRequestNoShape(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize);   // CAPI binary not supported
void prepareBinary4x4PredictRequest(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize = 1);   // CAPI binary not supported

void assertStringOutputProto(const ovms::InferenceTensor& proto, const std::vector<std::string>& expectedStrings);
void assertStringResponse(const ovms::InferenceResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName);

static const std::vector<ovms::Precision> SUPPORTED_CAPI_INPUT_PRECISIONS{
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::U1,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::U32,
    ovms::Precision::U64,
    ovms::Precision::BOOL
};

static const std::vector<ovms::Precision> UNSUPPORTED_CAPI_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    ovms::Precision::Q78,
    ovms::Precision::BIN,
    ovms::Precision::CUSTOM};

static const std::vector<ovms::Precision> SUPPORTED_CAPI_INPUT_PRECISIONS_TENSORINPUTCONTENT{
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::U32,
    ovms::Precision::U64,
    ovms::Precision::BOOL
};

static const std::vector<ovms::Precision> UNSUPPORTED_CAPI_INPUT_PRECISIONS_TENSORINPUTCONTENT{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    ovms::Precision::FP16,
    ovms::Precision::Q78,
    ovms::Precision::BIN,
};
