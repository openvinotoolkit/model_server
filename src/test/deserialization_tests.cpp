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

#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "../deserialization.hpp"
#include "test_utils.hpp"

#include <gmock/gmock-generated-function-mockers.h>

using TFTensorProto = tensorflow::TensorProto;
using KFSTensorProto = ::inference::ModelInferRequest::InferInputTensor;

using TFPredictRequest = tensorflow::serving::PredictRequest;
using TFPredictResponse = tensorflow::serving::PredictResponse;

using KFSPredictRequest = ::inference::ModelInferRequest;

using namespace ovms;

using testing::_;
using testing::NiceMock;
using testing::Throw;

std::vector<std::pair<ovms::Precision, bool>> cartesianProduct(const std::vector<ovms::Precision>& precisions, const std::vector<bool>& bufferInRawInputContents) {
    std::vector<std::pair<ovms::Precision, bool>> result;
    result.reserve(precisions.size() * bufferInRawInputContents.size());
    for (const auto& p : precisions) {
        for (const auto& b : bufferInRawInputContents) {
            result.emplace_back(p, b);
        }
    }
    return result;
}

class TensorflowGRPCPredict : public ::testing::TestWithParam<ovms::Precision> {
protected:
    void SetUp() override {
        auto precision = ovms::Precision::FP32;
        tensorMap[tensorName] = std::make_shared<ovms::TensorInfo>(
            tensorName,
            precision,
            shape_t{1, 3},
            Layout{"NC"});
        SetUpTensorProto(TensorInfo::getPrecisionAsDataType(precision));
    }
    void SetUpTensorProto(tensorflow::DataType dataType) {
        tensorProto.set_dtype(dataType);
        auto tensorShape = tensorProto.mutable_tensor_shape();
        tensorShape->Clear();
        tensorShape->add_dim()->set_size(1);
        tensorShape->add_dim()->set_size(DUMMY_MODEL_INPUT_SIZE);
        *(tensorProto.mutable_tensor_content()) = std::string(1 * DUMMY_MODEL_INPUT_SIZE, '1');
    }
    TFTensorProto tensorProto;
    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t tensorMap;
    bool isPipeline = false;
};

class DeserializeTFTensorProto : public TensorflowGRPCPredict {};
class DeserializeTFTensorProtoNegative : public TensorflowGRPCPredict {};

class GRPCPredictRequest : public TensorflowGRPCPredict {
public:
    void SetUp() {
        TensorflowGRPCPredict::SetUp();
        (*request.mutable_inputs())[tensorName] = tensorProto;
    }
    void TearDown() {
        request.mutable_inputs()->clear();
    }

public:
    TFPredictRequest request;
};

class GRPCPredictRequestNegative : public GRPCPredictRequest {};

TEST_P(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::InferRequest inferRequest;
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION)
        << "Unsupported OVMS precision:"
        << toString(testedPrecision)
        << " should return error";
}

TEST_P(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetTensorException) {
    ovms::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::InferRequest inferRequest;
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION) << status.string();
}

class MockTensorProtoDeserializatorThrowingInferenceEngine {
public:
    MOCK_METHOD(ov::Tensor,
        deserializeTensorProto,
        (const tensorflow::TensorProto&,
            const std::shared_ptr<ovms::TensorInfo>&));

    MOCK_METHOD(ov::Tensor,
        deserializeTensorProto,
        (const ::inference::ModelInferRequest::InferInputTensor&,
            const std::shared_ptr<ovms::TensorInfo>&,
            const std::string* buffer));
};

// Enables static method mock
class MockTensorProtoDeserializator {
public:
    static MockTensorProtoDeserializatorThrowingInferenceEngine* mock;
    static ov::Tensor deserializeTensorProto(
        const tensorflow::TensorProto& requestInput,
        const std::shared_ptr<ovms::TensorInfo>& tensorInfo) {
        return mock->deserializeTensorProto(requestInput, tensorInfo);
    }

    static ov::Tensor deserializeTensorProto(
        const ::inference::ModelInferRequest::InferInputTensor& requestInput,
        const std::shared_ptr<TensorInfo>& tensorInfo,
        const std::string* buffer) {
        return mock->deserializeTensorProto(requestInput, tensorInfo, buffer);
    }
};

MockTensorProtoDeserializatorThrowingInferenceEngine* MockTensorProtoDeserializator::mock = nullptr;

TEST_F(GRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetTensorException2) {
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    MockTensorProtoDeserializatorThrowingInferenceEngine mockTPobject;
    MockTensorProtoDeserializator::mock = &mockTPobject;
    EXPECT_CALL(mockTPobject, deserializeTensorProto(_, _))
        .Times(1)
        .WillRepeatedly(
            Throw(ov::Exception("")));
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    Status status;
    status = deserializePredictRequest<MockTensorProtoDeserializator>(
        request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR) << status.string();
}

TEST_F(GRPCPredictRequest, ShouldSuccessForSupportedPrecision) {
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_TRUE(status.ok());
}

TEST_P(DeserializeTFTensorProtoNegative, ShouldReturnNullptrForPrecision) {
    ovms::Precision testedPrecision = GetParam();
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::Tensor tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName]);
    EXPECT_FALSE((bool)tensor) << "Unsupported OVMS precision:"
                               << toString(testedPrecision)
                               << " should return nullptr";
}

TEST_P(DeserializeTFTensorProto, ShouldReturnValidTensor) {
    ovms::Precision testedPrecision = GetParam();
    SetUpTensorProto(TensorInfo::getPrecisionAsDataType(testedPrecision));
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::Tensor tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName]);
    EXPECT_TRUE((bool)tensor) << "Supported OVMS precision:"
                              << toString(testedPrecision)
                              << " should return valid tensor ptr";
}

class KserveGRPCPredict : public ::testing::TestWithParam<std::pair<ovms::Precision, bool>> {
protected:
    void SetUp() override {
        auto precision = ovms::Precision::FP32;
        tensorMap[tensorName] = std::make_shared<ovms::TensorInfo>(
            tensorName,
            precision,
            shape_t{1, 3},
            Layout{"NC"});
        SetUpTensorProto(TensorInfo::getPrecisionAsString(precision), true);
        float value = 1.0;
        auto bytes = static_cast<char*>(static_cast<void*>(&value));
        SetUpBuffer(bytes);
    }
    void SetUpTensorProto(std::string dataType, bool getInputFromRawInputContents) {
        ovms::Shape tensorShape{1, DUMMY_MODEL_INPUT_SIZE};
        tensorProto.set_name(tensorName);
        tensorProto.set_datatype(dataType);
        tensorProto.mutable_shape()->Clear();
        size_t elementsCount = 1;
        for (const auto& d : tensorShape) {
            tensorProto.add_shape(d.getStaticValue());
            ++elementsCount;
        }
        if (!getInputFromRawInputContents) {
            switch (KFSPrecisionToOvmsPrecision(dataType)) {
            case ovms::Precision::FP64: {
                for (size_t i = 0; i < elementsCount; ++i) {
                    auto ptr = tensorProto.mutable_contents()->mutable_fp64_contents()->Add();
                    *ptr = 1;
                }
                break;
            }
            case ovms::Precision::FP32: {
                for (size_t i = 0; i < elementsCount; ++i) {
                    auto ptr = tensorProto.mutable_contents()->mutable_fp32_contents()->Add();
                    *ptr = 1;
                }
                break;
            }
            // uint64_contents
            case ovms::Precision::U64: {
                for (size_t i = 0; i < elementsCount; ++i) {
                    auto ptr = tensorProto.mutable_contents()->mutable_uint64_contents()->Add();
                    *ptr = 1;
                }
                break;
            }
            // uint_contents
            case ovms::Precision::U8:
            case ovms::Precision::U16:
            case ovms::Precision::U32: {
                for (size_t i = 0; i < elementsCount; ++i) {
                    auto ptr = tensorProto.mutable_contents()->mutable_uint_contents()->Add();
                    *ptr = 1;
                }
                break;
            }
            // int64_contents
            case ovms::Precision::I64: {
                for (size_t i = 0; i < elementsCount; ++i) {
                    auto ptr = tensorProto.mutable_contents()->mutable_int64_contents()->Add();
                    *ptr = 1;
                }
                break;
            }
            // bool_contents
            case ovms::Precision::BOOL: {
                for (size_t i = 0; i < elementsCount; ++i) {
                    auto ptr = tensorProto.mutable_contents()->mutable_bool_contents()->Add();
                    *ptr = 1;
                }
                break;
            }
            // int_contents
            case ovms::Precision::I8:
            case ovms::Precision::I16:
            case ovms::Precision::I32: {
                for (size_t i = 0; i < elementsCount; ++i) {
                    auto ptr = tensorProto.mutable_contents()->mutable_int_contents()->Add();
                    *ptr = 1;
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
            default: {}
            }
        }
    }
    void SetUpBuffer(char* bytes) {
        buffer = "";
        for (int i = 0; i < DUMMY_MODEL_INPUT_SIZE; i++) {
            buffer += std::string(bytes, 4);
        }
    }

    KFSTensorProto tensorProto;
    std::string buffer;
    const char* tensorName = DUMMY_MODEL_INPUT_NAME;
    ovms::tensor_map_t tensorMap;
    bool isPipeline = false;
};

class DeserializeKFSTensorProto : public KserveGRPCPredict {};
class DeserializeKFSTensorProtoNegative : public KserveGRPCPredict {};

TEST_P(DeserializeKFSTensorProtoNegative, ShouldReturnNullptrForPrecision) {
    auto [testedPrecision, getInputFromRawInputContents] = GetParam();
    std::string* bufferPtr = (getInputFromRawInputContents ? &buffer : nullptr);
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::Tensor tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName], bufferPtr);
    EXPECT_FALSE((bool)tensor) << "Unsupported OVMS precision:"
                               << toString(testedPrecision)
                               << " should return nullptr";
}

TEST_P(DeserializeKFSTensorProto, ShouldReturnValidTensor) {
    auto [testedPrecision, getInputFromRawInputContents] = GetParam();
    std::string* bufferPtr = (getInputFromRawInputContents ? &buffer : nullptr);
    if (!getInputFromRawInputContents && (ovms::Precision::FP16 == testedPrecision)) {
        GTEST_SKIP() << "Not supported";
    }
    SetUpTensorProto(TensorInfo::getPrecisionAsString(testedPrecision), getInputFromRawInputContents);
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::Tensor tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName], bufferPtr);
    EXPECT_TRUE((bool)tensor) << "Supported OVMS precision:"
                              << toString(testedPrecision)
                              << " should return valid tensor ptr";
}

TEST_F(KserveGRPCPredict, ShouldReturnValidTensor) {
    ov::Tensor tensor = deserializeTensorProto<ConcreteTensorProtoDeserializator>(tensorProto, tensorMap[tensorName], &buffer);

    ASSERT_EQ(tensor.get_element_type(), ov::element::Type_t::f32);
    ASSERT_EQ(tensor.get_shape(), ov::Shape({1, DUMMY_MODEL_INPUT_SIZE}));
    float_t* data = (float_t*)tensor.data();
    for (int i = 0; i < DUMMY_MODEL_INPUT_SIZE; i++) {
        ASSERT_EQ(data[i], 1);
    }
}

class KserveGRPCPredictRequest : public KserveGRPCPredict {
public:
    void SetUp() {
        KserveGRPCPredict::SetUp();
        *request.add_inputs() = tensorProto;
        *request.add_raw_input_contents() = buffer;
    }
    void TearDown() {
        request.Clear();
    }

public:
    KFSPredictRequest request;
};

TEST_F(KserveGRPCPredictRequest, ShouldSuccessForSupportedPrecision) {
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_TRUE(status.ok());
    std::cout << status.string();
}

class KserveGRPCPredictRequestNegative : public KserveGRPCPredictRequest {};

TEST_P(KserveGRPCPredictRequestNegative, ShouldReturnDeserializationErrorForPrecision) {
    auto [testedPrecision, getInputFromRawInputContents] = GetParam();
    if (!getInputFromRawInputContents)
        GTEST_SKIP() << "test setup not implemented yet";
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::InferRequest inferRequest;
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION)
        << "Unsupported OVMS precision:"
        << toString(testedPrecision)
        << " should return error";
}

TEST_P(KserveGRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetTensorException) {
    auto [testedPrecision, getInputFromRawInputContents] = GetParam();
    if (!getInputFromRawInputContents)
        GTEST_SKIP() << "test setup not implemented yet";
    tensorMap[tensorName]->setPrecision(testedPrecision);
    ov::InferRequest inferRequest;
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    auto status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION) << status.string();
}

std::string toString(const std::pair<ovms::Precision, bool>& pair) {
    return toString(pair.first) + "_" + (pair.second ? "true" : "false");
}

TEST_F(KserveGRPCPredictRequestNegative, ShouldReturnDeserializationErrorForSetTensorException2) {
    ov::Core ieCore;
    std::shared_ptr<ov::Model> model = ieCore.read_model(std::filesystem::current_path().u8string() + "/src/test/dummy/1/dummy.xml");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "CPU");
    ov::InferRequest inferRequest = compiledModel.create_infer_request();
    MockTensorProtoDeserializatorThrowingInferenceEngine mockTPobject;
    MockTensorProtoDeserializator::mock = &mockTPobject;
    EXPECT_CALL(mockTPobject, deserializeTensorProto(_, _, _))
        .Times(1)
        .WillRepeatedly(
            Throw(ov::Exception("")));
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    Status status;
    status = deserializePredictRequest<MockTensorProtoDeserializator>(
        request, tensorMap, inputSink, isPipeline);
    EXPECT_EQ(status, ovms::StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR) << status.string();
}

std::vector<std::pair<ovms::Precision, bool>> KserveGRPCPredictRequestNegativeParams = cartesianProduct(UNSUPPORTED_KFS_INPUT_PRECISIONS, {true, false});

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    KserveGRPCPredictRequestNegative,
    ::testing::ValuesIn(KserveGRPCPredictRequestNegativeParams),
    [](const ::testing::TestParamInfo<KserveGRPCPredictRequestNegative::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    GRPCPredictRequestNegative,
    ::testing::ValuesIn(UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<GRPCPredictRequestNegative::ParamType>& info) {
        return toString(info.param);
    });

std::vector<std::pair<ovms::Precision, bool>> KserveGRPCPredictRequestParams = cartesianProduct(SUPPORTED_KFS_INPUT_PRECISIONS, {true, false});

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    KserveGRPCPredictRequest,
    ::testing::ValuesIn(KserveGRPCPredictRequestParams),
    [](const ::testing::TestParamInfo<KserveGRPCPredictRequest::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    GRPCPredictRequest,
    ::testing::ValuesIn(SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<GRPCPredictRequest::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeTFTensorProtoNegative,
    ::testing::ValuesIn(UNSUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<DeserializeTFTensorProtoNegative::ParamType>& info) {
        return toString(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeTFTensorProto,
    ::testing::ValuesIn(SUPPORTED_INPUT_PRECISIONS),
    [](const ::testing::TestParamInfo<DeserializeTFTensorProto::ParamType>& info) {
        return toString(info.param);
    });

std::vector<std::pair<ovms::Precision, bool>> DeserializeKFSTensorProtoNegativeParams = cartesianProduct(UNSUPPORTED_KFS_INPUT_PRECISIONS, {true, false});

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeKFSTensorProtoNegative,
    ::testing::ValuesIn(DeserializeKFSTensorProtoNegativeParams),
    [](const ::testing::TestParamInfo<DeserializeKFSTensorProtoNegative::ParamType>& info) {
        return toString(info.param);
    });

std::vector<std::pair<ovms::Precision, bool>> DeserializeKFSTensorProtoParams = cartesianProduct(SUPPORTED_KFS_INPUT_PRECISIONS, {true, false});

INSTANTIATE_TEST_SUITE_P(
    Test,
    DeserializeKFSTensorProto,
    ::testing::ValuesIn(DeserializeKFSTensorProtoParams),
    [](const ::testing::TestParamInfo<DeserializeKFSTensorProto::ParamType>& info) {
        return toString(info.param);
    });
