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
#include "test_utils.hpp"

#include "../prediction_service_utils.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

void waitForOVMSConfigReload(ovms::ModelManager& manager) {
    // This is effectively multiplying by 1.2 to have 1 config reload in between
    // two test steps
    const float WAIT_MULTIPLIER_FACTOR = 1.2;
    const uint waitTime = WAIT_MULTIPLIER_FACTOR * manager.getWatcherIntervalSec() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
}

std::string createConfigFileWithContent(const std::string& content, std::string filename) {
    std::ofstream configFile{filename};
    spdlog::info("Creating config file: {}\n with content:\n{}", filename, content);
    configFile << content << std::endl;
    configFile.close();
    if (configFile.fail()) {
        spdlog::info("Closing configFile failed");
    } else {
        spdlog::info("Closing configFile succeed");
    }
    return filename;
}

ovms::tensor_map_t prepareTensors(
    const std::unordered_map<std::string, ovms::shape_t>&& tensors,
    InferenceEngine::Precision precision) {
    ovms::tensor_map_t result;
    for (const auto& kv : tensors) {
        result[kv.first] = std::make_shared<ovms::TensorInfo>(
            kv.first,
            precision,
            kv.second);
    }
    return result;
}

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    PredictRequest& request, PredictResponse& response, int seriesLength, int batchSize) {
    ASSERT_EQ(response.outputs().count(outputName), 1);
    const auto& output_proto = response.outputs().at(outputName);

    ASSERT_EQ(output_proto.tensor_content().size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), batchSize);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), DUMMY_MODEL_OUTPUT_SIZE);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [seriesLength](float& v) { v += 1.0 * seriesLength; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));
}

std::string readableError(const float* expected_output, const float* actual_output, const size_t size) {
    std::stringstream ss;
    for (size_t i = 0; i < size; ++i) {
        if (actual_output[i] != expected_output[i]) {
            ss << "Expected:" << expected_output[i] << ", actual:" << actual_output[i] << " at place:" << i << std::endl;
        }
    }
    return ss.str();
}
