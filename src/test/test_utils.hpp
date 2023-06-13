//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include "../capi_frontend/inferencerequest.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
#include "../config.hpp"
#include "../dags/node_library.hpp"
#include "../execution_context.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../metric_registry.hpp"
#include "../modelinstance.hpp"
#include "../modelmanager.hpp"
#include "../shape.hpp"
#include "../tensorinfo.hpp"

using inputs_info_t = std::map<std::string, std::tuple<ovms::signed_shape_t, ovms::Precision>>;

const std::string dummy_model_location = std::filesystem::current_path().u8string() + "/src/test/dummy";
const std::string dummy_fp64_model_location = std::filesystem::current_path().u8string() + "/src/test/dummy_fp64";
const std::string sum_model_location = std::filesystem::current_path().u8string() + "/src/test/add_two_inputs_model";
const std::string increment_1x3x4x5_model_location = std::filesystem::current_path().u8string() + "/src/test/increment_1x3x4x5";
const std::string passthrough_model_location = std::filesystem::current_path().u8string() + "/src/test/passthrough";
const std::string dummy_saved_model_location = std::filesystem::current_path().u8string() + "/src/test/dummy_saved_model";

const ovms::ModelConfig DUMMY_MODEL_CONFIG{
    "dummy",
    dummy_model_location,  // base path
    "CPU",                 // target device
    "1",                   // batchsize
    1,                     // NIREQ
    false,                 // is stateful
    true,                  // idle sequence cleanup enabled
    false,                 // low latency transformation enabled
    500,                   // stateful sequence max number
    "",                    // cache directory
    1,                     // model_version unused since version are read from path
    dummy_model_location,  // local path
};

const ovms::ModelConfig DUMMY_FP64_MODEL_CONFIG{
    "dummy_fp64",
    dummy_fp64_model_location,  // base path
    "CPU",                      // target device
    "1",                        // batchsize
    1,                          // NIREQ
    false,                      // is stateful
    true,                       // idle sequence cleanup enabled
    false,                      // low latency transformation enabled
    500,                        // stateful sequence max number
    "",                         // cache directory
    1,                          // model_version unused since version are read from path
    dummy_fp64_model_location,  // local path
};

const ovms::ModelConfig SUM_MODEL_CONFIG{
    "sum",
    sum_model_location,  // base path
    "CPU",               // target device
    "1",                 // batchsize
    1,                   // NIREQ
    false,               // is stateful
    true,                // idle sequence cleanup enabled
    false,               // low latency transformation enabled
    500,                 // stateful sequence max number
    "",                  // cache directory
    1,                   // model_version unused since version are read from path
    sum_model_location,  // local path
};

const ovms::ModelConfig INCREMENT_1x3x4x5_MODEL_CONFIG{
    "increment_1x3x4x5",
    increment_1x3x4x5_model_location,  // base path
    "CPU",                             // target device
    "1",                               // batchsize
    1,                                 // NIREQ
    false,                             // is stateful
    true,                              // idle sequence cleanup enabled
    false,                             // low latency transformation enabled
    500,                               // stateful sequence max number
    "",                                // cache directory
    1,                                 // model_version unused since version are read from path
    increment_1x3x4x5_model_location,  // local path
};

const ovms::ModelConfig PASSTHROUGH_MODEL_CONFIG{
    "passthrough",
    passthrough_model_location,  // base path
    "CPU",                       // target device
    "1",                         // batchsize
    1,                           // NIREQ
    false,                       // is stateful
    true,                        // idle sequence cleanup enabled
    false,                       // low latency transformation enabled
    500,                         // stateful sequence max number
    "",                          // cache directory
    1,                           // model_version unused since version are read from path
    passthrough_model_location,  // local path
};

const ovms::ModelConfig DUMMY_SAVED_MODEL_CONFIG{
    "dummy_saved_model",
    dummy_saved_model_location,  // base path
    "CPU",                       // target device
    "1",                         // batchsize
    1,                           // NIREQ
    false,                       // is stateful
    true,                        // idle sequence cleanup enabled
    false,                       // low latency transformation enabled
    500,                         // stateful sequence max number
    "",                          // cache directory
    1,                           // model_version unused since version are read from path
    dummy_saved_model_location,  // local path
};

constexpr const char* DUMMY_MODEL_INPUT_NAME = "b";
constexpr const char* DUMMY_MODEL_OUTPUT_NAME = "a";
constexpr const int DUMMY_MODEL_INPUT_SIZE = 10;
constexpr const int DUMMY_MODEL_OUTPUT_SIZE = 10;
constexpr const float DUMMY_ADDITION_VALUE = 1.0;
const ovms::signed_shape_t DUMMY_MODEL_SHAPE{1, 10};
const ovms::Shape DUMMY_MODEL_SHAPE_META{1, 10};

constexpr const char* DUMMY_FP64_MODEL_INPUT_NAME = "input:0";
constexpr const char* DUMMY_FP64_MODEL_OUTPUT_NAME = "output:0";

constexpr const char* SUM_MODEL_INPUT_NAME_1 = "input1";
constexpr const char* SUM_MODEL_INPUT_NAME_2 = "input2";
constexpr const char* SUM_MODEL_OUTPUT_NAME = "sum";
constexpr const int SUM_MODEL_INPUT_SIZE = 10;
constexpr const int SUM_MODEL_OUTPUT_SIZE = 10;

constexpr const char* INCREMENT_1x3x4x5_MODEL_INPUT_NAME = "input";
constexpr const char* INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME = "output";
constexpr const float INCREMENT_1x3x4x5_ADDITION_VALUE = 1.0;

constexpr const char* PASSTHROUGH_MODEL_INPUT_NAME = "input";
constexpr const char* PASSTHROUGH_MODEL_OUTPUT_NAME = "copy:0";

const std::string UNUSED_SERVABLE_NAME = "UNUSED_SERVABLE_NAME";
constexpr const ovms::model_version_t UNUSED_MODEL_VERSION = 42;  // Answer to the Ultimate Question of Life

static const ovms::ExecutionContext DEFAULT_TEST_CONTEXT{ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::Predict};

using TFSRequestType = tensorflow::serving::PredictRequest;
using TFSResponseType = tensorflow::serving::PredictResponse;
using TFSInputTensorType = tensorflow::TensorProto;
using TFSOutputTensorType = tensorflow::TensorProto;
using TFSShapeType = tensorflow::TensorShapeProto;
using TFSInputTensorIteratorType = google::protobuf::Map<std::string, TFSInputTensorType>::const_iterator;
using TFSOutputTensorIteratorType = google::protobuf::Map<std::string, TFSOutputTensorType>::const_iterator;
using TFSInterface = std::pair<TFSRequestType, TFSResponseType>;
using KFSInterface = std::pair<KFSRequest, KFSResponse>;
using CAPIInterface = std::pair<ovms::InferenceRequest, ovms::InferenceResponse>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
ovms::tensor_map_t prepareTensors(
    const std::unordered_map<std::string, ovms::Shape>&& tensors,
    ovms::Precision precision = ovms::Precision::FP32);

void preparePredictRequest(tensorflow::serving::PredictRequest& request, inputs_info_t requestInputs, const std::vector<float>& data = {});

KFSTensorInputProto* findKFSInferInputTensor(::KFSRequest& request, const std::string& name);
std::string* findKFSInferInputTensorContentInRawInputs(::KFSRequest& request, const std::string& name);

void prepareKFSInferInputTensor(::KFSRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const std::string>& inputInfo,
    const std::vector<float>& data = {}, bool putBufferInInputTensorContent = false);
void prepareKFSInferInputTensor(::KFSRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const ovms::Precision>& inputInfo,
    const std::vector<float>& data = {}, bool putBufferInInputTensorContent = false);

void prepareCAPIInferInputTensor(ovms::InferenceRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, OVMS_DataType>& inputInfo,
    const std::vector<float>& data, uint32_t decrementBufferSize = 0, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> deviceId = std::nullopt);
void prepareCAPIInferInputTensor(ovms::InferenceRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const ovms::Precision>& inputInfo,
    const std::vector<float>& data, uint32_t decrementBufferSize = 0, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> deviceId = std::nullopt);

void preparePredictRequest(::KFSRequest& request, inputs_info_t requestInputs, const std::vector<float>& data = {}, bool putBufferInInputTensorContent = false);

void preparePredictRequest(ovms::InferenceRequest& request, inputs_info_t requestInputs, const std::vector<float>& data,
    uint32_t decrementBufferSize = 0, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> deviceId = std::nullopt);

void prepareInferStringTensor(::KFSRequest::InferInputTensor& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content);
void prepareInferStringTensor(tensorflow::TensorProto& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content);
void prepareInferStringTensor(ovms::InferenceTensor& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content);

void prepareInferStringRequest(::KFSRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent = true);
void prepareInferStringRequest(tensorflow::serving::PredictRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent = true);
void prepareInferStringRequest(ovms::InferenceRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent = true);  // CAPI binary not supported

void assertOutputTensorMatchExpectations(const ov::Tensor& tensor, std::vector<std::string> expectedStrings);

void prepareBinaryPredictRequest(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize);
void prepareBinaryPredictRequest(::KFSRequest& request, const std::string& inputName, const int batchSize);
void prepareBinaryPredictRequest(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize);  // CAPI binary not supported

void prepareBinaryPredictRequestNoShape(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize);
void prepareBinaryPredictRequestNoShape(::KFSRequest& request, const std::string& inputName, const int batchSize);
void prepareBinaryPredictRequestNoShape(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize);  // CAPI binary not supported
void prepareBinary4x4PredictRequest(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize = 1);
void prepareBinary4x4PredictRequest(::KFSRequest& request, const std::string& inputName, const int batchSize = 1);
void prepareBinary4x4PredictRequest(ovms::InferenceRequest& request, const std::string& inputName, const int batchSize = 1);  // CAPI binary not supported

template <typename TensorType>
void prepareInvalidImageBinaryTensor(TensorType& tensor);

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    tensorflow::serving::PredictRequest& request, tensorflow::serving::PredictResponse& response, int seriesLength, int batchSize = 1, const std::string& servableName = "");

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    ::KFSRequest& request, ::KFSResponse& response, int seriesLength, int batchSize = 1, const std::string& servableName = "");

template <typename T>
std::string readableError(const T* expected_output, const T* actual_output, const size_t size) {
    std::stringstream ss;
    for (size_t i = 0; i < size; ++i) {
        if (actual_output[i] != expected_output[i]) {
            ss << "Expected:" << expected_output[i] << ", actual:" << actual_output[i] << " at place:" << i << std::endl;
            break;
        }
    }
    return ss.str();
}

void assertStringOutputProto(const tensorflow::TensorProto& proto, const std::vector<std::string>& expectedStrings);
void assertStringOutputProto(const KFSTensorOutputProto& proto, const std::vector<std::string>& expectedStrings);
void assertStringOutputProto(const ovms::InferenceTensor& proto, const std::vector<std::string>& expectedStrings);

void assertStringResponse(const tensorflow::serving::PredictResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName);
void assertStringResponse(const ::KFSResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName);
void assertStringResponse(const ovms::InferenceResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName);

void checkAddResponse(const std::string outputName,
    const std::vector<float>& requestData1,
    const std::vector<float>& requestData2,
    ::KFSRequest& request, ::KFSResponse& response, int seriesLength, int batchSize, const std::string& servableName);

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

void checkIncrement4DimShape(const std::string outputName,
    tensorflow::serving::PredictResponse& response,
    const std::vector<size_t>& expectedShape);

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

// returns path to a file.
std::string createConfigFileWithContent(const std::string& content, std::string filename = "/tmp/ovms_config_file.json");
#pragma GCC diagnostic pop

template <typename T>
static std::vector<T> asVector(const std::string& tensor_content) {
    std::vector<T> v(tensor_content.size() / sizeof(T) + 1);
    std::memcpy(
        reinterpret_cast<char*>(v.data()),
        reinterpret_cast<const char*>(tensor_content.data()),
        tensor_content.size());
    v.resize(tensor_content.size() / sizeof(T));
    return v;
}

class ConstructorEnabledModelManager : public ovms::ModelManager {
    ovms::MetricRegistry registry;

public:
    ConstructorEnabledModelManager(const std::string& modelCacheDirectory = "") :
        ovms::ModelManager(modelCacheDirectory, &registry) {}
    ~ConstructorEnabledModelManager() {
        join();
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
        models.clear();
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
    }
    ovms::Status loadConfig(const std::string& jsonFilename) {
        return ModelManager::loadConfig(jsonFilename);
    }

    /**
     * @brief Updates OVMS configuration with cached configuration file. Will check for newly added model versions
     */
    void updateConfigurationWithoutConfigFile() {
        ModelManager::updateConfigurationWithoutConfigFile();
    }
};

class MockedMetadataModelIns : public ovms::ModelInstance {
public:
    MockedMetadataModelIns(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", 42, ieCore) {}
    MOCK_METHOD(const ovms::tensor_map_t&, getInputsInfo, (), (const, override));
    MOCK_METHOD(ovms::Dimension, getBatchSize, (), (const, override));
    MOCK_METHOD(const ovms::ModelConfig&, getModelConfig, (), (const, override));
    const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request) {
        return validate(request);
    }
    const ovms::Status mockValidate(const ::KFSRequest* request) {
        return validate(request);
    }
    const ovms::Status mockValidate(const ovms::InferenceRequest* request) {
        return validate(request);
    }
};

class ResourcesAccessModelManager : public ConstructorEnabledModelManager {
public:
    int getResourcesSize() {
        return resources.size();
    }
};

class TestWithTempDir : public ::testing::Test {
protected:
    void SetUp() override {
        const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss;
        ss << std::string(test_info->test_suite_name())
           << "/"
           << std::string(test_info->name());
        const std::string directoryName = ss.str();
        directoryPath = "/tmp/" + directoryName;
        std::filesystem::remove_all(directoryPath);
        std::filesystem::create_directories(directoryPath);
    }

    void TearDown() override {
        std::filesystem::remove_all(directoryPath);
    }

    std::string directoryPath;
};

void waitForOVMSConfigReload(ovms::ModelManager& manager);
void waitForOVMSResourcesCleanup(ovms::ModelManager& manager);

template <typename T>
static ovms::NodeLibrary createLibraryMock() {
    return ovms::NodeLibrary{
        T::initialize,
        T::deinitialize,
        T::execute,
        T::getInputsInfo,
        T::getOutputsInfo,
        T::release};
}

bool isShapeTheSame(const tensorflow::TensorShapeProto&, const std::vector<int64_t>&&);
bool isShapeTheSame(const KFSShapeType&, const std::vector<int64_t>&&);

void readRgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes);
void read4x4RgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes);
void readImage(const std::string& path, size_t& filesize, std::unique_ptr<char[]>& image_bytes);

static const std::vector<ovms::Precision> SUPPORTED_INPUT_PRECISIONS{
    // ovms::Precision::UNDEFINED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    // ovms::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    // ovms::Precision::BIN,
    // ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

static const std::vector<ovms::Precision> UNSUPPORTED_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    // ovms::Precision::FP16,
    ovms::Precision::Q78,
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    // ovms::Precision::I64,
    ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

static const std::vector<ovms::Precision> SUPPORTED_CAPI_INPUT_PRECISIONS{
    // ovms::Precision::UNDEFINED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    // ovms::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::U1,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::U32,
    ovms::Precision::U64,
    // ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};
static const std::vector<ovms::Precision> UNSUPPORTED_CAPI_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    // ovms::Precision::FP16,
    ovms::Precision::Q78,
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::U1,
    // vms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    // ovms::Precision::I64,
    // ovms::Precision::U32,
    // ovms::Precision::U64,
    ovms::Precision::BIN,
    // ovms::Precision::BOOL
    ovms::Precision::CUSTOM};
static const std::vector<ovms::Precision> SUPPORTED_KFS_INPUT_PRECISIONS{
    // ovms::Precision::UNDEFINED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    ovms::Precision::FP16,
    // ovms::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::U32,
    ovms::Precision::U64,
    // ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

static const std::vector<ovms::Precision> UNSUPPORTED_KFS_INPUT_PRECISIONS{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    // ovms::Precision::FP16,
    ovms::Precision::Q78,
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    // ovms::Precision::I64,
    // ovms::Precision::U32,
    // ovms::Precision::U64,
    ovms::Precision::BIN,
    // ovms::Precision::BOOL
    ovms::Precision::CUSTOM};

static const std::vector<ovms::Precision> SUPPORTED_KFS_INPUT_PRECISIONS_TENSORINPUTCONTENT{
    // ovms::Precision::UNDEFINED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    // ovms::Precision::FP16,
    // ovms::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::U32,
    ovms::Precision::U64,
    // ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

static const std::vector<ovms::Precision> UNSUPPORTED_KFS_INPUT_PRECISIONS_TENSORINPUTCONTENT{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    ovms::Precision::FP16,
    ovms::Precision::Q78,
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    // ovms::Precision::I64,
    // ovms::Precision::U32,
    // ovms::Precision::U64,
    ovms::Precision::BIN,
    // ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

static const std::vector<ovms::Precision> SUPPORTED_CAPI_INPUT_PRECISIONS_TENSORINPUTCONTENT{
    // ovms::Precision::UNDEFINED,
    // ovms::Precision::MIXED,
    ovms::Precision::FP64,
    ovms::Precision::FP32,
    // ovms::Precision::FP16,
    // ovms::Precision::Q78,
    ovms::Precision::I16,
    ovms::Precision::U8,
    ovms::Precision::I8,
    ovms::Precision::U16,
    ovms::Precision::I32,
    ovms::Precision::I64,
    ovms::Precision::U32,
    ovms::Precision::U64,
    // ovms::Precision::BIN,
    ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

static const std::vector<ovms::Precision> UNSUPPORTED_CAPI_INPUT_PRECISIONS_TENSORINPUTCONTENT{
    ovms::Precision::UNDEFINED,
    ovms::Precision::MIXED,
    // ovms::Precision::FP64,
    // ovms::Precision::FP32,
    ovms::Precision::FP16,
    ovms::Precision::Q78,
    // ovms::Precision::I16,
    // ovms::Precision::U8,
    // ovms::Precision::I8,
    // ovms::Precision::U16,
    // ovms::Precision::I32,
    // ovms::Precision::I64,
    // ovms::Precision::U32,
    // ovms::Precision::U64,
    ovms::Precision::BIN,
    // ovms::Precision::BOOL
    // ovms::Precision::CUSTOM)
};

void randomizePort(std::string& port);
void randomizePorts(std::string& port1, std::string& port2);

class ConstructorEnabledConfig : public ovms::Config {
public:
    ConstructorEnabledConfig() {}
};

std::shared_ptr<const ovms::TensorInfo> createTensorInfoCopyWithPrecision(std::shared_ptr<const ovms::TensorInfo> src, ovms::Precision precision);
