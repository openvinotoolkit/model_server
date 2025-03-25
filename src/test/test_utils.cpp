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

#include <algorithm>
#include <chrono>
#include <functional>
#include <optional>
#include <unordered_set>

#include "../capi_frontend/capi_utils.hpp"
#include "../capi_frontend/inferenceparameter.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../prediction_service_utils.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../tensorinfo.hpp"
#include "../tfs_frontend/tfs_utils.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using ovms::TensorInfo;

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

void preparePredictRequest(tensorflow::serving::PredictRequest& request, inputs_info_t requestInputs, const std::vector<float>& data) {
    request.mutable_inputs()->clear();
    for (auto const& it : requestInputs) {
        auto& name = it.first;
        auto [shape, precision] = it.second;

        auto& input = (*request.mutable_inputs())[name];
        auto datatype = getPrecisionAsDataType(precision);
        input.set_dtype(datatype);
        size_t numberOfElements = 1;
        for (auto const& dim : shape) {
            input.mutable_tensor_shape()->add_dim()->set_size(dim);
            numberOfElements *= dim;
        }
        switch (datatype) {
        case tensorflow::DataType::DT_HALF: {
            if (data.size() == 0) {
                for (size_t i = 0; i < numberOfElements; i++) {
                    input.add_half_val('1');
                }
            } else {
                for (size_t i = 0; i < data.size(); i++) {
                    input.add_half_val(data[i]);
                }
            }
            break;
        }
        case tensorflow::DataType::DT_UINT16: {
            if (data.size() == 0) {
                for (size_t i = 0; i < numberOfElements; i++) {
                    input.add_int_val('1');
                }
            } else {
                for (size_t i = 0; i < data.size(); i++) {
                    input.add_int_val(data[i]);
                }
            }
            break;
        }
        default: {
            if (data.size() == 0) {
                *input.mutable_tensor_content() = std::string(numberOfElements * tensorflow::DataTypeSize(datatype), '1');
            } else {
                std::string content;
                content.resize(data.size() * tensorflow::DataTypeSize(datatype));
                std::memcpy(content.data(), data.data(), content.size());
                *input.mutable_tensor_content() = content;
            }
        }
        }
    }
}

void waitForOVMSConfigReload(ovms::ModelManager& manager) {
    // This is effectively multiplying by 5 to have at least 1 config reload in between
    // two test steps, but we check if config files changed to exit earlier if changes are already applied
    const float WAIT_MULTIPLIER_FACTOR = 5;
    const uint32_t waitTime = WAIT_MULTIPLIER_FACTOR * manager.getWatcherIntervalMillisec() * 1000;
    bool reloadIsNeeded = true;
    int timestepMs = 10;

    auto start = std::chrono::high_resolution_clock::now();
    while (reloadIsNeeded &&
           (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() < waitTime)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(timestepMs));
        manager.configFileReloadNeeded(reloadIsNeeded);
    }
}

void waitForOVMSResourcesCleanup(ovms::ModelManager& manager) {
    // This is effectively multiplying by 1.8 to have 1 config reload in between
    // two test steps
    const float WAIT_MULTIPLIER_FACTOR = 1.8;
    const uint32_t waitTime = WAIT_MULTIPLIER_FACTOR * manager.getResourcesCleanupIntervalMillisec();
    SPDLOG_DEBUG("waitForOVMSResourcesCleanup {} ms", waitTime);
    std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
}

bool createConfigFileWithContent(const std::string& content, std::string filename) {
    std::ofstream configFile{filename};
    // Check if the file was successfully opened
    if (!configFile.is_open()) {
        SPDLOG_ERROR("Failed to open file: {}", filename);
        throw std::runtime_error("Failed to open file: " + filename);
    }
    SPDLOG_INFO("Creating config file: {}\n with content:\n{}", filename, content);
    configFile << content << std::endl;
    configFile.close();
    if (configFile.fail()) {
        SPDLOG_INFO("Closing configFile failed");
        return false;
    } else {
        SPDLOG_INFO("Closing configFile succeed");
    }
    return true;
}

ovms::tensor_map_t prepareTensors(
    const std::unordered_map<std::string, ovms::Shape>&& tensors,
    ovms::Precision precision) {
    ovms::tensor_map_t result;
    for (const auto& kv : tensors) {
        result[kv.first] = std::make_shared<ovms::TensorInfo>(
            kv.first,
            precision,
            kv.second);
    }
    return result;
}

std::string readableSetError(std::unordered_set<std::string> actual, std::unordered_set<std::string> expected) {
    std::stringstream ss;
    std::unordered_set<std::string>::const_iterator it;
    if (actual.size() >= expected.size()) {
        for (auto iter = actual.begin(); iter != actual.end(); ++iter) {
            it = expected.find(*iter);
            if (it == expected.end()) {
                ss << "Missing element in expected set: " << *iter << std::endl;
            }
        }
    } else {
        for (auto iter = expected.begin(); iter != expected.end(); ++iter) {
            it = actual.find(*iter);
            if (it == actual.end()) {
                ss << "Missing element in actual set: " << *iter << std::endl;
            }
        }
    }
    return ss.str();
}

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    PredictRequest& request, PredictResponse& response, int seriesLength, int batchSize, const std::string& servableName, size_t expectedOutputsCount) {
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
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
    checkBuffers(actual_output, expected_output, dataLengthToCheck);
}

void checkScalarResponse(const std::string outputName,
    float inputScalar, PredictResponse& response, const std::string& servableName) {
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    const auto& output_proto = response.outputs().at(outputName);

    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 0);

    ASSERT_EQ(output_proto.tensor_content().size(), sizeof(float));
    ASSERT_EQ(*((float*)output_proto.tensor_content().data()), inputScalar);
}

void checkScalarResponse(const std::string outputName,
    float inputScalar, ::KFSResponse& response, const std::string& servableName) {
    ASSERT_EQ(response.model_name(), servableName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), outputName) << "Did not find:" << outputName;
    const auto& output_proto = *response.outputs().begin();
    std::string* content = response.mutable_raw_output_contents(0);

    ASSERT_EQ(output_proto.shape_size(), 0);
    ASSERT_EQ(content->size(), sizeof(float));

    ASSERT_EQ(*((float*)content->data()), inputScalar);
}

void checkStringResponse(const std::string outputName,
    const std::vector<std::string>& inputStrings, PredictResponse& response, const std::string& servableName) {
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    const auto& output_proto = response.outputs().at(outputName);

    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 1);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), inputStrings.size());
    ASSERT_EQ(output_proto.dtype(), tensorflow::DT_STRING);

    ASSERT_EQ(output_proto.string_val_size(), inputStrings.size());
    for (size_t i = 0; i < inputStrings.size(); i++) {
        ASSERT_EQ(output_proto.string_val(i), inputStrings[i]);
    }
}

void checkStringResponse(const std::string outputName,
    const std::vector<std::string>& inputStrings, ::KFSResponse& response, const std::string& servableName) {
    ASSERT_EQ(response.model_name(), servableName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), outputName) << "Did not find:" << outputName;
    const auto& output_proto = *response.outputs().begin();
    std::string* content = response.mutable_raw_output_contents(0);

    ASSERT_EQ(output_proto.shape_size(), 1);
    ASSERT_EQ(output_proto.shape(0), inputStrings.size());

    size_t offset = 0;
    for (size_t i = 0; i < inputStrings.size(); i++) {
        ASSERT_GE(content->size(), offset + 4);
        uint32_t batchLength = *((uint32_t*)(content->data() + offset));
        ASSERT_EQ(batchLength, inputStrings[i].size());
        offset += 4;
        ASSERT_GE(content->size(), offset + batchLength);
        ASSERT_EQ(std::string(content->data() + offset, batchLength), inputStrings[i]);
        offset += batchLength;
    }
    ASSERT_EQ(offset, content->size());
}

void checkAddResponse(const std::string outputName,
    const std::vector<float>& requestData1,
    const std::vector<float>& requestData2,
    ::KFSRequest& request, const ::KFSResponse& response, int seriesLength, int batchSize, const std::string& servableName) {
    ASSERT_EQ(response.model_name(), servableName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), outputName) << "Did not find:" << outputName;
    const auto& output_proto = *response.outputs().begin();
    const std::string& content = response.raw_output_contents(0);

    ASSERT_EQ(content.size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto.shape_size(), 2);
    ASSERT_EQ(output_proto.shape(0), batchSize);
    ASSERT_EQ(output_proto.shape(1), DUMMY_MODEL_OUTPUT_SIZE);

    std::vector<float> responseData = requestData1;
    for (size_t i = 0; i < requestData1.size(); ++i) {
        responseData[i] += requestData2[i];
    }

    const float* actual_output = (const float*)content.data();
    float* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(float);
    checkBuffers(actual_output, expected_output, dataLengthToCheck);
}

void checkIncrement4DimShape(const std::string outputName,
    PredictResponse& response,
    const std::vector<size_t>& expectedShape) {
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    const auto& output_proto = response.outputs().at(outputName);

    ASSERT_EQ(output_proto.tensor_shape().dim_size(), expectedShape.size());
    for (size_t i = 0; i < expectedShape.size(); i++) {
        ASSERT_EQ(output_proto.tensor_shape().dim(i).size(), expectedShape[i]);
    }
}

bool isShapeTheSame(const tensorflow::TensorShapeProto& actual, const std::vector<int64_t>&& expected) {
    bool same = true;
    if (static_cast<unsigned int>(actual.dim_size()) != expected.size()) {
        SPDLOG_ERROR("Unexpected dim_size. Got: {}, Expect: {}", actual.dim_size(), expected.size());
        return false;
    }
    for (int i = 0; i < actual.dim_size(); i++) {
        if (actual.dim(i).size() != expected[i]) {
            SPDLOG_ERROR("Unexpected dim[{}]. Got: {}, Expect: {}", i, actual.dim(i).size(), expected[i]);
            same = false;
        }
    }
    if (same == false) {
        std::stringstream ss;
        for (int i = 0; i < actual.dim_size(); i++) {
            ss << "dim["
               << i
               << "] got:"
               << actual.dim(i).size()
               << " expect:" << expected[i];
        }
        SPDLOG_ERROR("Shape mismatch: {}", ss.str());
    }
    return same;
}

bool isShapeTheSame(const KFSShapeType& actual, const std::vector<int64_t>&& expected) {
    bool same = true;
    int a_size = actual.size();
    if (a_size != int(expected.size())) {
        SPDLOG_ERROR("Unexpected dim_size. Got: {}, Expect: {}", a_size, expected.size());
        return false;
    }
    for (int i = 0; i < a_size; i++) {
        if (actual.at(i) != expected[i]) {
            SPDLOG_ERROR("Unexpected dim[{}]. Got: {}, Expect: {}", i, actual.at(i), expected[i]);
            same = false;
            break;
        }
    }
    if (same == false) {
        std::stringstream ss;
        for (int i = 0; i < a_size; i++) {
            ss << "dim["
               << i
               << "] got:"
               << actual.at(i)
               << " expect:" << expected[i];
        }
        SPDLOG_ERROR("Shape mismatch: {}", ss.str());
    }
    return same;
}

void readImage(const std::string& path, size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    std::ifstream DataFile;
    DataFile.open(path, std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    filesize = DataFile.tellg();
    DataFile.seekg(0);
    image_bytes = std::make_unique<char[]>(filesize);
    DataFile.read(image_bytes.get(), filesize);
}

void readRgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    return readImage(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb.jpg"), filesize, image_bytes);
}

void read4x4RgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    return readImage(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb4x4.jpg"), filesize, image_bytes);
}

void prepareInferStringTensor(::KFSRequest::InferInputTensor& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content) {
    if (!putBufferInInputTensorContent && content == nullptr) {
        throw std::runtime_error("Preparation of infer string tensor failed");
        return;
    }
    tensor.set_name(name);
    tensor.set_datatype("BYTES");
    tensor.mutable_shape()->Clear();
    tensor.add_shape(data.size());
    if (!putBufferInInputTensorContent) {
        size_t dataSize = 0;
        for (auto input : data) {
            dataSize += input.size() + 4;
        }
        content->resize(dataSize);
        size_t offset = 0;
        for (auto input : data) {
            uint32_t inputSize = input.size();
            std::memcpy(content->data() + offset, reinterpret_cast<const unsigned char*>(&inputSize), sizeof(uint32_t));
            offset += sizeof(uint32_t);
            std::memcpy(content->data() + offset, input.data(), input.length());
            offset += input.length();
        }
    } else {
        for (auto inputData : data) {
            auto bytes_val = tensor.mutable_contents()->mutable_bytes_contents()->Add();
            bytes_val->append(inputData.data(), inputData.size());
        }
    }
}

void prepareInferStringRequest(::KFSRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent) {
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
    prepareInferStringTensor(*tensor, name, data, putBufferInInputTensorContent, content);
}

void prepareInferStringTensor(tensorflow::TensorProto& tensor, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent, std::string* content) {
    tensor.set_dtype(tensorflow::DataType::DT_STRING);
    tensor.mutable_tensor_shape()->add_dim()->set_size(data.size());
    for (auto inputData : data) {
        tensor.add_string_val(inputData);
    }
}

void prepareInferStringRequest(tensorflow::serving::PredictRequest& request, const std::string& name, const std::vector<std::string>& data, bool putBufferInInputTensorContent) {
    request.mutable_inputs()->clear();
    auto& input = (*request.mutable_inputs())[name];
    prepareInferStringTensor(input, name, data, putBufferInInputTensorContent, nullptr);
}

void assertOutputTensorMatchExpectations(const ov::Tensor& tensor, std::vector<std::string> expectedStrings) {
    size_t maxStringLength = 0;
    for (const auto& input : expectedStrings) {
        maxStringLength = std::max(maxStringLength, input.size());
    }
    size_t width = maxStringLength + 1;
    size_t i = 0;
    ASSERT_EQ(tensor.get_shape().size(), 2);
    ASSERT_EQ(tensor.get_shape()[0], expectedStrings.size());
    ASSERT_EQ(tensor.get_shape()[1], width);
    ASSERT_EQ(tensor.get_size(), (width * expectedStrings.size()));
    for (const auto& input : expectedStrings) {
        for (size_t j = 0; j < input.size(); j++) {
            ASSERT_EQ(
                tensor.data<uint8_t>()[i * width + j],
                reinterpret_cast<const uint8_t*>(input.data())[j])
                << "Tensor data does not match expectations for input: " << input << " at index: " << i << " and position: " << j;
        }
        for (size_t j = input.size(); j < width; j++) {
            ASSERT_EQ(tensor.data<uint8_t>()[i * width + j], 0);
        }
        i++;
    }
}

void assertStringOutputProto(const tensorflow::TensorProto& proto, const std::vector<std::string>& expectedStrings) {
    ASSERT_EQ(proto.string_val_size(), expectedStrings.size());
    for (size_t i = 0; i < expectedStrings.size(); i++) {
        ASSERT_EQ(proto.string_val(i), expectedStrings[i]);
    }
}
void assertStringOutputProto(const KFSTensorOutputProto& proto, const std::vector<std::string>& expectedStrings) {
    ASSERT_EQ(proto.contents().bytes_contents_size(), expectedStrings.size());
    for (size_t i = 0; i < expectedStrings.size(); i++) {
        ASSERT_EQ(proto.contents().bytes_contents(i), expectedStrings[i]);
    }
}
void assertStringOutputProto(const ovms::InferenceTensor& proto, const std::vector<std::string>& expectedStrings) {
    FAIL() << "not implemented";
}

void assertStringResponse(const tensorflow::serving::PredictResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName) {
    ASSERT_EQ(proto.outputs().count(outputName), 1);
    ASSERT_EQ(proto.outputs().at(outputName).dtype(), tensorflow::DataType::DT_STRING);
    ASSERT_EQ(proto.outputs().at(outputName).tensor_shape().dim_size(), 1);
    ASSERT_EQ(proto.outputs().at(outputName).tensor_shape().dim(0).size(), expectedStrings.size());
    assertStringOutputProto(proto.outputs().at(outputName), expectedStrings);
}
void assertStringResponse(const ::KFSResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName) {
    ASSERT_EQ(proto.outputs_size(), 1);
    ASSERT_EQ(proto.outputs(0).name(), outputName);
    ASSERT_EQ(proto.outputs(0).datatype(), "BYTES");
    ASSERT_EQ(proto.outputs(0).shape_size(), 1);
    ASSERT_EQ(proto.outputs(0).shape(0), expectedStrings.size());
    std::string expectedString;
    for (auto str : expectedStrings) {
        int size = str.size();
        for (int k = 0; k < 4; k++, size >>= 8) {
            expectedString += static_cast<char>(size & 0xff);
        }
        expectedString.append(str);
    }
    ASSERT_EQ(memcmp(proto.raw_output_contents(0).data(), expectedString.data(), expectedString.size()), 0);
}
void assertStringResponse(const ovms::InferenceResponse& proto, const std::vector<std::string>& expectedStrings, const std::string& outputName) {
    FAIL() << "not implemented";
}

void prepareBinaryPredictRequest(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize) {
    auto& tensor = (*request.mutable_inputs())[inputName];
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    readRgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor.add_string_val(image_bytes.get(), filesize);
    }
    tensor.set_dtype(tensorflow::DataType::DT_STRING);
    tensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
}

void prepareBinaryPredictRequest(::KFSRequest& request, const std::string& inputName, const int batchSize) {
    request.add_inputs();
    auto tensor = request.mutable_inputs()->Mutable(0);
    tensor->set_name(inputName);
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    readRgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor->mutable_contents()->add_bytes_contents(image_bytes.get(), filesize);
    }
    tensor->set_datatype("BYTES");
    tensor->mutable_shape()->Add(batchSize);
}

void prepareBinaryPredictRequestNoShape(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize) {
    auto& tensor = (*request.mutable_inputs())[inputName];
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    readRgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor.add_string_val(image_bytes.get(), filesize);
    }
    tensor.set_dtype(tensorflow::DataType::DT_STRING);
}

void prepareBinaryPredictRequestNoShape(::KFSRequest& request, const std::string& inputName, const int batchSize) {
    request.add_inputs();
    auto tensor = request.mutable_inputs()->Mutable(0);
    tensor->set_name(inputName);
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    readRgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor->mutable_contents()->add_bytes_contents(image_bytes.get(), filesize);
    }
    tensor->set_datatype("BYTES");
}

void prepareBinary4x4PredictRequest(tensorflow::serving::PredictRequest& request, const std::string& inputName, const int batchSize) {
    auto& tensor = (*request.mutable_inputs())[inputName];
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    read4x4RgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor.add_string_val(image_bytes.get(), filesize);
    }
    tensor.set_dtype(tensorflow::DataType::DT_STRING);
    tensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
}

void prepareBinary4x4PredictRequest(::KFSRequest& request, const std::string& inputName, const int batchSize) {
    request.add_inputs();
    auto tensor = request.mutable_inputs()->Mutable(0);
    tensor->set_name(inputName);
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    read4x4RgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor->mutable_contents()->add_bytes_contents(image_bytes.get(), filesize);
    }
    tensor->set_datatype("BYTES");
    tensor->mutable_shape()->Add(batchSize);
}

::KFSTensorInputProto* findKFSInferInputTensor(::KFSRequest& request, const std::string& name) {
    auto it = request.mutable_inputs()->begin();
    while (it != request.mutable_inputs()->end()) {
        if (it->name() == name)
            break;
        ++it;
    }
    return it == request.mutable_inputs()->end() ? nullptr : &(*it);
}

std::string* findKFSInferInputTensorContentInRawInputs(::KFSRequest& request, const std::string& name) {
    auto it = request.mutable_inputs()->begin();
    size_t bufferId = 0;
    std::string* content = nullptr;
    while (it != request.mutable_inputs()->end()) {
        if (it->name() == name)
            break;
        ++it;
        ++bufferId;
    }
    if (it != request.mutable_inputs()->end()) {
        content = request.mutable_raw_input_contents()->Mutable(bufferId);
    }
    return content;
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

    // In case shape is negative, deduce size from provided data size
    // Otherwise calculate from shape
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

void randomizePort(std::string& port) {
    std::mt19937_64 eng{std::random_device{}()};
    std::uniform_int_distribution<> dist{0, 9};
    for (auto j : {1, 2, 3}) {
        char* digitToRandomize = (char*)port.c_str() + j;
        *digitToRandomize = '0' + dist(eng);
    }
}
void randomizePorts(std::string& port1, std::string& port2) {
    randomizePort(port1);
    randomizePort(port2);
    while (port2 == port1) {
        randomizePort(port2);
    }
}

const int64_t SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS = 5;

void SetUpServer(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& port, const char* configPath) {
    server.setShutdownRequest(0);
    randomizePort(port);
    char* argv[] = {(char*)"ovms",
        (char*)"--config_path",
        (char*)configPath,
        (char*)"--port",
        (char*)port.c_str()};
    int argc = 5;
    t.reset(new std::thread([&argc, &argv, &server]() {
        EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
    }));
    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (!server.isReady()) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS)) {
    }
}
void EnsureSetUpServer(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& port, const char* configPath, int timeoutSeconds) {
    SetUpServer(t, server, port, configPath);
    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < timeoutSeconds)) {
    }
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED) << "OVMS did not fully load until allowed time. Check machine load";
}

void SetUpServer(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& port, const char* modelPath, const char* modelName) {
    server.setShutdownRequest(0);
    randomizePort(port);
    char* argv[] = {(char*)"ovms",
        (char*)"--model_name",
        (char*)modelName,
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest(modelPath).c_str(),
        (char*)"--port",
        (char*)port.c_str()};
    int argc = 7;
    t.reset(new std::thread([&argc, &argv, &server]() {
        EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
    }));
    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (!server.isReady()) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS)) {
    }
}

std::shared_ptr<const TensorInfo> createTensorInfoCopyWithPrecision(std::shared_ptr<const TensorInfo> src, ovms::Precision newPrecision) {
    return std::make_shared<TensorInfo>(
        src->getName(),
        src->getMappedName(),
        newPrecision,
        src->getShape(),
        src->getLayout());
}

// Static map workaround for char* pointers as paths
const std::string& getPathFromMap(std::string inputPath, std::string outputPath) {
    static std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);
    static std::unordered_map<std::string, std::string> inputMap = {};
    auto it = inputMap.find(inputPath);
    if (it != inputMap.end()) {
        // element exists
        return inputMap.at(inputPath);
    } else {
        // element does not exist
        inputMap.emplace(inputPath, outputPath);
        return inputMap.at(inputPath);
    }
}

// Function changes linux docker container path /ovms/src/test/dummy to windows workspace "C:\git\model_server\src\test\dummy"
// Depending on the ovms_test.exe location after build
const std::string& getGenericFullPathForSrcTest(const std::string& linuxPath, bool logChange) {
#ifdef __linux__
    return getPathFromMap(linuxPath, linuxPath);
#elif _WIN32
    // For ovms_test cwd = C:\git\model_server\bazel-out\x64_windows-opt\bin\src
    std::filesystem::path cwd = std::filesystem::current_path();
    std::size_t bazelOutIndex = cwd.string().find("bazel-out");

    // Example linuxPath "/ovms/src/test/dummy"
    std::size_t postOvmsIndex = linuxPath.find("/src/test");
    if (postOvmsIndex != std::string::npos) {
        // Setting winPath to "/src/test/dummy"
        std::string winPath = linuxPath.substr(postOvmsIndex);
        // Set basePath to "C:\git\model_server\"
        std::string basePath = bazelOutIndex != std::string::npos ? cwd.string().substr(0, bazelOutIndex) : cwd.string();
        // Combine "C:\git\model_server\" + "/src/test/dummy"
        std::string finalWinPath = basePath + winPath;
        // Change paths to linux separator for JSON parser compatybility in configs
        std::replace(finalWinPath.begin(), finalWinPath.end(), '\\', '/');

        if (logChange) {
            std::cout << "[WINDOWS DEBUG] Changed path: " << linuxPath << " to path: " << finalWinPath << " for Windows" << std::endl;
        }
        return getPathFromMap(linuxPath, finalWinPath);
    }
#endif
    return getPathFromMap(linuxPath, linuxPath);
}

// Function changes linux docker container path /ovms/bazel-out/src/lib_node_mock.so to windows workspace "C:\git\model_server\bazel-bin\src\lib_node_mock.so"
// Depending on the ovms_test.exe location after build
const std::string& getGenericFullPathForBazelOut(const std::string& linuxPath, bool logChange) {
#ifdef __linux__
    return getPathFromMap(linuxPath, linuxPath);
#elif _WIN32
    // For ovms_test cwd = C:\git\model_server\bazel-out\x64_windows-opt\bin\src
    std::filesystem::path cwd = std::filesystem::current_path();
    std::size_t bazelOutIndex = cwd.string().find("bazel-out");

    // Example linuxPath "/ovms/bazel-bin/src/lib_node_mock.so"
    std::size_t postOvmsIndex = linuxPath.find("/bazel-bin/src");
    if (postOvmsIndex != std::string::npos) {
        // Setting winPath to "/bazel-bin/src"
        std::string winPath = linuxPath.substr(postOvmsIndex);
        // Set basePath to "C:\git\model_server\"
        std::string basePath = bazelOutIndex != std::string::npos ? cwd.string().substr(0, bazelOutIndex) : cwd.string();
        // Combine "C:\git\model_server\" + "/bazel-bin/src"
        std::string finalWinPath = basePath + winPath;
        // Change paths to linux separator for JSON parser compatybility in configs
        std::replace(finalWinPath.begin(), finalWinPath.end(), '\\', '/');

        if (logChange) {
            std::cout << "[WINDOWS DEBUG] Changed path: " << linuxPath << " to path: " << finalWinPath << " for Windows" << std::endl;
        }
        return getPathFromMap(linuxPath, finalWinPath);
    }
#endif
    return getPathFromMap(linuxPath, linuxPath);
}

const std::string& getGenericFullPathForSrcTest(const char* linuxPath, bool logChange) {
    return getGenericFullPathForSrcTest(std::string(linuxPath, strlen(linuxPath)), logChange);
}

// Function changes docker linux paths starting with /tmp: "/tmp/dummy" to windows C:\git\model_server\tmp\dummy
const std::string& getGenericFullPathForTmp(const std::string& linuxPath, bool logChange) {
#ifdef __linux__
    return getPathFromMap(linuxPath, linuxPath);
#elif _WIN32
    // For ovms_test cwd = C:\git\model_server\bazel-out\x64_windows-opt\bin\src
    std::filesystem::path cwd = std::filesystem::current_path();
    size_t bazelOutIndex = cwd.string().find("bazel-out");

    // Example linuxPath "/tmp/dummy"
    const std::string tmpString = "/tmp";
    const size_t tmpStringSize = 4;

    size_t postTmpIndex = linuxPath.find(tmpString) + tmpStringSize;
    if (postTmpIndex != std::string::npos) {
        std::string winPath = linuxPath.substr(postTmpIndex);
        // Set basePath to "C:\git\model_server\"
        std::string basePath = bazelOutIndex != std::string::npos ? cwd.string().substr(0, bazelOutIndex) : cwd.string();
        // Combine "C:\git\model_server\" + "tmp" "\dummy"
        std::string finalWinPath = basePath + tmpString + winPath;
        // Change paths to linux separator for JSON parser compatybility in configs
        std::replace(finalWinPath.begin(), finalWinPath.end(), '\\', '/');

        if (logChange) {
            std::cout << "[WINDOWS DEBUG] Changed path: " << linuxPath << " to path: " << finalWinPath << " for Windows" << std::endl;
        }
        return getPathFromMap(linuxPath, finalWinPath);
    }
#endif
    return getPathFromMap(linuxPath, linuxPath);
}

const std::string& getGenericFullPathForTmp(const char* linuxPath, bool logChange) {
    return getGenericFullPathForTmp(std::string(linuxPath, strlen(linuxPath)), logChange);
}

#ifdef _WIN32
const std::string getWindowsRepoRootPath() {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::size_t bazelOutIndex = cwd.string().find("bazel-out");
    std::string rootPath = cwd.string().substr(0, bazelOutIndex);
    std::replace(rootPath.begin(), rootPath.end(), '\\', '/');
    return rootPath;
}
#endif
// Apply necessary changes so the graph config will comply with the platform
// that tests are run on
void adjustConfigForTargetPlatform(std::string& input) {
#ifdef _WIN32
    std::string repoTestPath = getWindowsRepoRootPath() + "/src/test";
    std::string searchString = "\"/ovms/src/test";
    std::string replaceString = "\"" + repoTestPath;
    size_t pos = 0;
    while ((pos = input.find(searchString, pos)) != std::string::npos) {
        input.replace(pos, searchString.length(), replaceString);
        pos += replaceString.length();
    }

    repoTestPath = getWindowsRepoRootPath() + "/tmp";
    searchString = "\"/tmp";
    replaceString = "\"" + repoTestPath;
    pos = 0;
    while ((pos = input.find(searchString, pos)) != std::string::npos) {
        input.replace(pos, searchString.length(), replaceString);
        pos += replaceString.length();
    }

    repoTestPath = getWindowsRepoRootPath() + "/bazel-bin/src";
    searchString = "\"/ovms/bazel-bin/src";
    replaceString = "\"" + repoTestPath;
    pos = 0;
    while ((pos = input.find(searchString, pos)) != std::string::npos) {
        input.replace(pos, searchString.length(), replaceString);
        pos += replaceString.length();
    }
#elif __linux__
    // No changes needed for linux now, but keeping it as a placeholder
#endif
}

// Apply necessary changes so the graph config will comply with the platform
// that tests are run on
const std::string& adjustConfigForTargetPlatformReturn(std::string& input) {
    adjustConfigForTargetPlatform(input);
    return input;
}

std::string adjustConfigForTargetPlatformCStr(const char* input) {
    std::string inputString(input);
    adjustConfigForTargetPlatform(inputString);
    return inputString;
}

void adjustConfigToAllowModelFileRemovalWhenLoaded(ovms::ModelConfig& modelConfig) {
#ifdef _WIN32
    modelConfig.setPluginConfig(ovms::plugin_config_t({{"ENABLE_MMAP", "NO"}}));
#endif
    // on linux we can remove files from disk even if mmap is enabled
}
