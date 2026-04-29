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

#include <algorithm>
#include <cstring>
#include <numeric>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include <gtest/gtest.h>

#include "src/kfs_frontend/kfs_grpc_inference_service.hpp"
#include "src/kfs_frontend/kfs_utils.hpp"
#include "test_utils.hpp"

using KFSInterface = std::pair<KFSRequest, KFSResponse>;

KFSTensorInputProto* findKFSInferInputTensor(::KFSRequest& request, const std::string& name);
std::string* findKFSInferInputTensorContentInRawInputs(::KFSRequest& request, const std::string& name);

template <typename T = float>
void prepareKFSInferInputTensor(::KFSRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const std::string>& inputInfo,
    const std::vector<T>& data = std::vector<float>{}, bool putBufferInInputTensorContent = false) {
    auto it = request.mutable_inputs()->begin();
    size_t bufferId = 0;
    while (it != request.mutable_inputs()->end()) {
        if (it->name() == name)
            break;
        ++it;
        ++bufferId;
    }
    KFSTensorInputProto* tensor;
    std::string* content = nullptr;
    if (it != request.mutable_inputs()->end()) {
        tensor = &*it;
        if (!putBufferInInputTensorContent) {
            content = request.mutable_raw_input_contents()->Mutable(bufferId);
        }
    } else {
        tensor = request.add_inputs();
        if (!putBufferInInputTensorContent) {
            content = request.add_raw_input_contents();
        }
    }
    auto [shape, datatype] = inputInfo;
    tensor->set_name(name);
    tensor->set_datatype(datatype);
    size_t elementsCount = 1;
    tensor->mutable_shape()->Clear();
    bool isNegativeShape = false;
    for (auto const& dim : shape) {
        tensor->add_shape(dim);
        if (dim < 0) {
            isNegativeShape = true;
        }
        elementsCount *= dim;
    }
    size_t dataSize = isNegativeShape ? data.size() : elementsCount;
    if (!putBufferInInputTensorContent) {
        if (data.size() == 0) {
            content->assign(dataSize * ovms::KFSDataTypeSize(datatype), '1');
        } else {
            content->resize(dataSize * ovms::KFSDataTypeSize(datatype));
            std::memcpy(content->data(), data.data(), content->size());
        }
    } else {
        switch (ovms::KFSPrecisionToOvmsPrecision(datatype)) {
        case ovms::Precision::FP64: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_fp64_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        case ovms::Precision::FP32: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_fp32_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        case ovms::Precision::U64: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_uint64_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        case ovms::Precision::U8:
        case ovms::Precision::U16:
        case ovms::Precision::U32: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_uint_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        case ovms::Precision::I64: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_int64_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        case ovms::Precision::BOOL: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_bool_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        case ovms::Precision::I8:
        case ovms::Precision::I16:
        case ovms::Precision::I32: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_int_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        case ovms::Precision::FP16:
        case ovms::Precision::U1:
        case ovms::Precision::CUSTOM:
        case ovms::Precision::UNDEFINED:
        case ovms::Precision::DYNAMIC:
        case ovms::Precision::MIXED:
        case ovms::Precision::Q78:
        case ovms::Precision::BIN:
        default: {
        }
        }
    }
}

template <>
inline void prepareKFSInferInputTensor<bool>(::KFSRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const std::string>& inputInfo,
    const std::vector<bool>& data, bool putBufferInInputTensorContent) {
    if (putBufferInInputTensorContent == 0) {
        throw std::string("Unsupported");
    }
    auto it = request.mutable_inputs()->begin();
    size_t bufferId = 0;
    while (it != request.mutable_inputs()->end()) {
        if (it->name() == name)
            break;
        ++it;
        ++bufferId;
    }
    KFSTensorInputProto* tensor;
    if (it != request.mutable_inputs()->end()) {
        tensor = &*it;
    } else {
        tensor = request.add_inputs();
    }
    auto [shape, datatype] = inputInfo;
    tensor->set_name(name);
    tensor->set_datatype(datatype);
    size_t elementsCount = 1;
    tensor->mutable_shape()->Clear();
    bool isNegativeShape = false;
    for (auto const& dim : shape) {
        tensor->add_shape(dim);
        if (dim < 0) {
            isNegativeShape = true;
        }
        elementsCount *= dim;
    }
    size_t dataSize = isNegativeShape ? data.size() : elementsCount;
    for (size_t i = 0; i < dataSize; ++i) {
        auto ptr = tensor->mutable_contents()->mutable_bool_contents()->Add();
        *ptr = (data.size() ? data[i] : 1);
    }
}

template <typename T = float>
void prepareKFSInferInputTensor(::KFSRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const ovms::Precision>& inputInfo,
    const std::vector<T>& data = std::vector<float>{}, bool putBufferInInputTensorContent = false) {
    auto [shape, type] = inputInfo;
    prepareKFSInferInputTensor(request, name,
        {shape, ovmsPrecisionToKFSPrecision(type)},
        data, putBufferInInputTensorContent);
}

template <typename T = float>
void preparePredictRequest(::KFSRequest& request, inputs_info_t requestInputs, const std::vector<T>& data = std::vector<float>{}, bool putBufferInInputTensorContent = false) {
    request.mutable_inputs()->Clear();
    request.mutable_raw_input_contents()->Clear();
    for (auto const& it : requestInputs) {
        prepareKFSInferInputTensor(request, it.first, it.second, data, putBufferInInputTensorContent);
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static std::string vectorTypeToKfsString(const std::type_info& vectorType) {
    if (vectorType == typeid(float))
        return std::string("FP32");
    if (vectorType == typeid(int32_t))
        return std::string("INT32");
    if (vectorType == typeid(double))
        return std::string("FP64");
    if (vectorType == typeid(int64_t))
        return std::string("INT64");
    if (vectorType == typeid(int16_t))
        return std::string("INT16");
    if (vectorType == typeid(int8_t))
        return std::string("INT8");
    if (vectorType == typeid(uint64_t))
        return std::string("UINT64");
    if (vectorType == typeid(uint32_t))
        return std::string("UINT32");
    if (vectorType == typeid(uint16_t))
        return std::string("UINT16");
    if (vectorType == typeid(uint8_t))
        return std::string("UINT8");
    if (vectorType == typeid(bool))
        return std::string("BOOL");
    return std::string("UNDEFINED");
}
#pragma GCC diagnostic pop

template <typename T = float>
void checkDummyResponse(const std::string outputName,
    const std::vector<T>& requestData,
    ::KFSRequest& request, ::KFSResponse& response, int seriesLength, int batchSize = 1, const std::string& servableName = "", size_t expectedOutputsCount = 1) {
    ASSERT_EQ(response.model_name(), servableName);
    ASSERT_EQ(response.outputs_size(), expectedOutputsCount);
    ASSERT_EQ(response.raw_output_contents_size(), expectedOutputsCount);
    auto it = std::find_if(response.outputs().begin(), response.outputs().end(), [&outputName](const ::KFSResponse::InferOutputTensor& tensor) {
        return tensor.name() == outputName;
    });
    ASSERT_NE(it, response.outputs().end());
    auto outputIndex = it - response.outputs().begin();
    const auto& output_proto = *it;
    std::string* content = response.mutable_raw_output_contents(outputIndex);

    ASSERT_EQ(content->size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(T));
    ASSERT_EQ(output_proto.datatype(), vectorTypeToKfsString(typeid(T)));
    ASSERT_EQ(output_proto.shape_size(), 2);
    ASSERT_EQ(output_proto.shape(0), batchSize);
    ASSERT_EQ(output_proto.shape(1), DUMMY_MODEL_OUTPUT_SIZE);

    std::vector<T> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [seriesLength](T& v) {
        v += 1.0 * seriesLength;
    });

    T* actual_output = (T*)content->data();
    T* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(T);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(T));
}

void checkScalarResponse(const std::string outputName,
    float inputScalar, ::KFSResponse& response, const std::string& servableName = "");

void checkStringResponse(const std::string outputName,
    const std::vector<std::string>& inputStrings, ::KFSResponse& response, const std::string& servableName = "");

void assertStringOutputProto(const KFSTensorOutputProto& proto, const std::vector<std::string>& expectedStrings);

void assertStringResponse(const ::KFSResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName);

void checkAddResponse(const std::string outputName,
    const std::vector<float>& requestData1,
    const std::vector<float>& requestData2,
    ::KFSRequest& request, const ::KFSResponse& response, int seriesLength, int batchSize, const std::string& servableName);

template <typename T>
void checkIncrement4DimResponse(const std::string outputName,
    const std::vector<T>& expectedData,
    ::KFSResponse& response,
    const std::vector<size_t>& expectedShape,
    bool checkRaw = true) {
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.mutable_outputs(0)->name(), outputName);
    ASSERT_EQ(response.outputs(0).shape_size(), expectedShape.size());
    for (size_t i = 0; i < expectedShape.size(); i++) {
        ASSERT_EQ(response.outputs(0).shape(i), expectedShape[i]);
    }

    if (checkRaw) {
        ASSERT_EQ(response.raw_output_contents_size(), 1);
        auto elementsCount = std::accumulate(expectedShape.begin(), expectedShape.end(), 1, std::multiplies<size_t>());
        ASSERT_EQ(response.raw_output_contents(0).size(), elementsCount * sizeof(T));
        T* actual_output = (T*)response.raw_output_contents(0).data();
        T* expected_output = (T*)expectedData.data();
        const int dataLengthToCheck = elementsCount * sizeof(T);
        EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
            << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(T));
    } else {
        ASSERT_EQ(response.outputs(0).datatype(), "UINT8") << "other precision testing currently not supported";
        ASSERT_EQ(sizeof(T), 1) << "other precision testing currently not supported";
        ASSERT_EQ(response.outputs(0).contents().uint_contents_size(), expectedData.size());
        for (size_t i = 0; i < expectedData.size(); i++) {
            ASSERT_EQ(response.outputs(0).contents().uint_contents(i), expectedData[i]);
        }
    }
}

bool isShapeTheSame(const KFSShapeType&, const std::vector<int64_t>&&);

void prepareInferStringTensor(::KFSRequest::InferInputTensor& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content);
void prepareInferStringRequest(::KFSRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent = true);

void prepareBinaryPredictRequest(::KFSRequest& request, const std::string& inputName, const int batchSize);
void prepareBinaryPredictRequestNoShape(::KFSRequest& request, const std::string& inputName, const int batchSize);
void prepareBinary4x4PredictRequest(::KFSRequest& request, const std::string& inputName, const int batchSize = 1);

static const std::vector<ovms::Precision> SUPPORTED_KFS_INPUT_PRECISIONS{
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
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

static const std::vector<ovms::Precision> UNSUPPORTED_KFS_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    ovms::Precision::Q78,
    ovms::Precision::BIN,
    ovms::Precision::CUSTOM};

static const std::vector<ovms::Precision> SUPPORTED_KFS_INPUT_PRECISIONS_TENSORINPUTCONTENT{
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

static const std::vector<ovms::Precision> UNSUPPORTED_KFS_INPUT_PRECISIONS_TENSORINPUTCONTENT{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    ovms::Precision::FP16,
    ovms::Precision::Q78,
    ovms::Precision::BIN,
};
