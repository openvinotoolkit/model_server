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
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <inference_engine.hpp>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "../modelmanager.hpp"
#include "../node_library.hpp"
#include "../tensorinfo.hpp"

using inputs_info_t = std::map<std::string, std::tuple<ovms::shape_t, tensorflow::DataType>>;

const std::string dummy_model_location = std::filesystem::current_path().u8string() + "/src/test/dummy";
const std::string sum_model_location = std::filesystem::current_path().u8string() + "/src/test/add_two_inputs_model";
const std::string increment_1x3x4x5_model_location = std::filesystem::current_path().u8string() + "/src/test/increment_1x3x4x5";

const ovms::ModelConfig DUMMY_MODEL_CONFIG{
    "dummy",
    dummy_model_location,  // base path
    "CPU",                 // target device
    "1",                   // batchsize
    1,                     // NIREQ
    false,                 // is stateful
    true,                  // idle sequence cleanup enabled
    false,                 // low latency transformation enabled
    500,                   // steteful sequence max number
    1,                     // model_version unused since version are read from path
    dummy_model_location,  // local path
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
    500,                 // steteful sequence max number
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
    500,                               // steteful sequence max number
    1,                                 // model_version unused since version are read from path
    increment_1x3x4x5_model_location,  // local path
};

constexpr const char* DUMMY_MODEL_INPUT_NAME = "b";
constexpr const char* DUMMY_MODEL_OUTPUT_NAME = "a";
constexpr const int DUMMY_MODEL_INPUT_SIZE = 10;
constexpr const int DUMMY_MODEL_OUTPUT_SIZE = 10;
constexpr const float DUMMY_ADDITION_VALUE = 1.0;

constexpr const char* SUM_MODEL_INPUT_NAME_1 = "input1";
constexpr const char* SUM_MODEL_INPUT_NAME_2 = "input2";
constexpr const char* SUM_MODEL_OUTPUT_NAME = "sum";
constexpr const int SUM_MODEL_INPUT_SIZE = 10;
constexpr const int SUM_MODEL_OUTPUT_SIZE = 10;

constexpr const char* INCREMENT_1x3x4x5_MODEL_INPUT_NAME = "input";
constexpr const char* INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME = "output";
constexpr const float INCREMENT_1x3x4x5_ADDITION_VALUE = 1.0;

constexpr const ovms::model_version_t UNUSED_MODEL_VERSION = 42;  // Answer to the Ultimate Question of Life

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
ovms::tensor_map_t prepareTensors(
    const std::unordered_map<std::string, ovms::shape_t>&& tensors,
    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32);

static tensorflow::serving::PredictRequest preparePredictRequest(inputs_info_t requestInputs, const std::vector<float>& data = {}) {
    tensorflow::serving::PredictRequest request;
    for (auto const& it : requestInputs) {
        auto& name = it.first;
        auto [shape, dtype] = it.second;

        auto& input = (*request.mutable_inputs())[name];
        input.set_dtype(dtype);
        size_t numberOfElements = 1;
        for (auto const& dim : shape) {
            input.mutable_tensor_shape()->add_dim()->set_size(dim);
            numberOfElements *= dim;
        }
        if (data.size() == 0) {
            *input.mutable_tensor_content() = std::string(numberOfElements * tensorflow::DataTypeSize(dtype), '1');
        } else {
            std::string content;
            content.resize(numberOfElements * tensorflow::DataTypeSize(dtype));
            std::memcpy(content.data(), data.data(), content.size());
            *input.mutable_tensor_content() = content;
        }
    }
    return request;
}

tensorflow::serving::PredictRequest prepareBinaryPredictRequest(const std::string& inputName, const int batchSize = 1);

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    tensorflow::serving::PredictRequest& request, tensorflow::serving::PredictResponse& response, int seriesLength, int batchSize = 1);

void checkIncrement4DimResponse(const std::string outputName,
    const std::vector<float>& expectedData,
    tensorflow::serving::PredictRequest& request,
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

static std::string readableError(const float* expected_output, const float* actual_output, const size_t size) {
    std::stringstream ss;
    for (size_t i = 0; i < size; ++i) {
        if (actual_output[i] != expected_output[i]) {
            ss << "Expected:" << expected_output[i] << ", actual:" << actual_output[i] << " at place:" << i << std::endl;
            break;
        }
    }
    return ss.str();
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
public:
    ConstructorEnabledModelManager() :
        ovms::ModelManager() {}
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

template <typename T>
static ovms::NodeLibrary createLibraryMock() {
    return ovms::NodeLibrary{
        T::execute,
        T::getInputsInfo,
        T::getOutputsInfo,
        T::release};
}

extern bool isShapeTheSame(const tensorflow::TensorShapeProto&, const std::vector<int64_t>&&);

void readRgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes);
