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
#include "test_request_utils_capi.hpp"

#include <gtest/gtest.h>

#include "src/capi_frontend/capi_utils.hpp"
#include "src/capi_frontend/inferenceparameter.hpp"

void prepareBinaryPredictRequest(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize) { throw 42; }         // CAPI binary not supported
void prepareBinaryPredictRequestNoShape(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize) { throw 42; }  // CAPI binary not supported
void prepareBinary4x4PredictRequest(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize) { throw 42; }      // CAPI binary not supported

void prepareInferStringRequest(ovms::InferenceRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent) { throw 42; }                     // CAPI binary not supported
void prepareInferStringTensor(ovms::InferenceTensor& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content) { throw 42; }  // CAPI binary not supported

void preparePredictRequest(ovms::InferenceRequest& request, inputs_info_t requestInputs, const std::vector<float>& data, uint32_t decrementBufferSize, OVMS_BufferType bufferType, std::optional<uint32_t> deviceId) {
    request.removeAllInputs();
    for (auto const& it : requestInputs) {
        prepareCAPIInferInputTensor(request, it.first, it.second, data, decrementBufferSize, bufferType, deviceId);
    }
}

void prepareCAPIInferInputTensor(ovms::InferenceRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const ovms::Precision>& inputInfo,
    const std::vector<float>& data, uint32_t decrementBufferSize, OVMS_BufferType bufferType, std::optional<uint32_t> deviceId) {
    auto [shape, type] = inputInfo;
    prepareCAPIInferInputTensor(request, name,
        {shape, getPrecisionAsOVMSDataType(type)},
        data, decrementBufferSize, bufferType, deviceId);
}

void prepareCAPIInferInputTensor(ovms::InferenceRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, OVMS_DataType>& inputInfo,
    const std::vector<float>& data, uint32_t decrementBufferSize, OVMS_BufferType bufferType, std::optional<uint32_t> deviceId) {
    auto [shape, datatype] = inputInfo;
    size_t elementsCount = 1;

    bool isShapeNegative = false;
    for (auto const& dim : shape) {
        if (dim < 0) {
            isShapeNegative = true;
        }
        elementsCount *= dim;
    }

    request.addInput(name.c_str(), datatype, shape.data(), shape.size());

    size_t dataSize = 0;
    if (isShapeNegative) {
        dataSize = data.size() * ovms::DataTypeToByteSize(datatype);
    } else {
        dataSize = elementsCount * ovms::DataTypeToByteSize(datatype);
    }
    if (decrementBufferSize)
        dataSize -= decrementBufferSize;

    request.setInputBuffer(name.c_str(), data.data(), dataSize, bufferType, deviceId);
}

void assertStringOutputProto(const ovms::InferenceTensor& proto, const std::vector<std::string>& expectedStrings) {
    FAIL() << "not implemented";
}

void assertStringResponse(const ovms::InferenceResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName) {
    FAIL() << "not implemented";
}
