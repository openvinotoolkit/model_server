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

#include <cstring>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "src/test/test_utils.hpp"

using TFSRequestType = tensorflow::serving::PredictRequest;
using TFSResponseType = tensorflow::serving::PredictResponse;
using TFSInputTensorType = tensorflow::TensorProto;
using TFSOutputTensorType = tensorflow::TensorProto;
using TFSShapeType = tensorflow::TensorShapeProto;
using TFSInputTensorIteratorType = google::protobuf::Map<std::string, TFSInputTensorType>::const_iterator;
using TFSOutputTensorIteratorType = google::protobuf::Map<std::string, TFSOutputTensorType>::const_iterator;
using TFSInterface = std::pair<TFSRequestType, TFSResponseType>;

void preparePredictRequest(tensorflow::serving::PredictRequest& request, inputs_info_t requestInputs, const std::vector<float>& data = std::vector<float>{});

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    tensorflow::serving::PredictRequest& request, tensorflow::serving::PredictResponse& response, int seriesLength, int batchSize = 1, const std::string& servableName = "", size_t expectedOutputsCount = 1);

void checkScalarResponse(const std::string outputName,
    float inputScalar, tensorflow::serving::PredictResponse& response, const std::string& servableName = "");

void checkStringResponse(const std::string outputName,
    const std::vector<std::string>& inputStrings, tensorflow::serving::PredictResponse& response, const std::string& servableName = "");

void assertStringOutputProto(const tensorflow::TensorProto& proto, const std::vector<std::string>& expectedStrings);

void assertStringResponse(const tensorflow::serving::PredictResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName);

template <typename T>
void checkIncrement4DimResponse(const std::string outputName,
    const std::vector<T>& expectedData,
    tensorflow::serving::PredictResponse& response,
    const std::vector<size_t>& expectedShape,
    bool checkRaw = true) {
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    const auto& output_proto = response.outputs().at(outputName);

    auto elementsCount = std::accumulate(expectedShape.begin(), expectedShape.end(), 1, std::multiplies<size_t>());

    ASSERT_EQ(output_proto.tensor_content().size(), elementsCount * sizeof(T));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), expectedShape.size());
    for (size_t i = 0; i < expectedShape.size(); i++) {
        ASSERT_EQ(output_proto.tensor_shape().dim(i).size(), expectedShape[i]);
    }

    T* actual_output = (T*)output_proto.tensor_content().data();
    T* expected_output = (T*)expectedData.data();
    const int dataLengthToCheck = elementsCount * sizeof(T);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(T));
}

void checkIncrement4DimShape(const std::string outputName,
    tensorflow::serving::PredictResponse& response,
    const std::vector<size_t>& expectedShape);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static std::vector<int> asVector(const tensorflow::TensorShapeProto& proto) {
    std::vector<int> shape;
    for (int i = 0; i < proto.dim_size(); i++) {
        shape.push_back(proto.dim(i).size());
    }
    return shape;
}

static std::vector<google::protobuf::int32> asVector(google::protobuf::RepeatedField<google::protobuf::int32>* container) {
    std::vector<google::protobuf::int32> result(container->size(), 0);
    std::memcpy(result.data(), container->mutable_data(), result.size() * sizeof(google::protobuf::int32));
    return result;
}
#pragma GCC diagnostic pop

bool isShapeTheSame(const tensorflow::TensorShapeProto&, const std::vector<int64_t>&&);

void prepareInferStringTensor(tensorflow::TensorProto& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content);
void prepareInferStringRequest(tensorflow::serving::PredictRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent = true);

void prepareBinaryPredictRequest(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize);
void prepareBinaryPredictRequestNoShape(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize);
void prepareBinary4x4PredictRequest(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize = 1);

static const std::vector<ovms::Precision> SUPPORTED_TFS_INPUT_PRECISIONS{
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::U32,
    ovms::Precision::I32,
    ovms::Precision::I64,
};

static const std::vector<ovms::Precision> UNSUPPORTED_TFS_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    ovms::Precision::Q78,
    ovms::Precision::BIN,
    ovms::Precision::BOOL
};
