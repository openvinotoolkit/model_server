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
#include <functional>

#include "../capi_frontend/capi_utils.hpp"
#include "../capi_frontend/inferenceparameter.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../prediction_service_utils.hpp"
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

void preparePredictRequest(::KFSRequest& request, inputs_info_t requestInputs, const std::vector<float>& data, bool putBufferInInputTensorContent) {
    request.mutable_inputs()->Clear();
    request.mutable_raw_input_contents()->Clear();
    for (auto const& it : requestInputs) {
        prepareKFSInferInputTensor(request, it.first, it.second, data, putBufferInInputTensorContent);
    }
}

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
    // This is effectively multiplying by 1.8 to have 1 config reload in between
    // two test steps
    const float WAIT_MULTIPLIER_FACTOR = 1.8;
    const uint waitTime = WAIT_MULTIPLIER_FACTOR * manager.getWatcherIntervalSec() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
}

void waitForOVMSResourcesCleanup(ovms::ModelManager& manager) {
    // This is effectively multiplying by 1.8 to have 1 config reload in between
    // two test steps
    const float WAIT_MULTIPLIER_FACTOR = 1.8;
    const uint waitTime = WAIT_MULTIPLIER_FACTOR * manager.getResourcesCleanupIntervalSec() * 1000;
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

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    PredictRequest& request, PredictResponse& response, int seriesLength, int batchSize, const std::string& servableName) {
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
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));
}

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    ::KFSRequest& request, ::KFSResponse& response, int seriesLength, int batchSize, const std::string& servableName) {
    ASSERT_EQ(response.model_name(), servableName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), outputName) << "Did not find:" << outputName;
    const auto& output_proto = *response.outputs().begin();
    std::string* content = response.mutable_raw_output_contents(0);

    ASSERT_EQ(content->size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto.shape_size(), 2);
    ASSERT_EQ(output_proto.shape(0), batchSize);
    ASSERT_EQ(output_proto.shape(1), DUMMY_MODEL_OUTPUT_SIZE);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [seriesLength](float& v) { v += 1.0 * seriesLength; });

    float* actual_output = (float*)content->data();
    float* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));
}

void checkAddResponse(const std::string outputName,
    const std::vector<float>& requestData1,
    const std::vector<float>& requestData2,
    ::KFSRequest& request, ::KFSResponse& response, int seriesLength, int batchSize, const std::string& servableName) {
    ASSERT_EQ(response.model_name(), servableName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), outputName) << "Did not find:" << outputName;
    const auto& output_proto = *response.outputs().begin();
    std::string* content = response.mutable_raw_output_contents(0);

    ASSERT_EQ(content->size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto.shape_size(), 2);
    ASSERT_EQ(output_proto.shape(0), batchSize);
    ASSERT_EQ(output_proto.shape(1), DUMMY_MODEL_OUTPUT_SIZE);

    std::vector<float> responseData = requestData1;
    for (size_t i = 0; i < requestData1.size(); ++i) {
        responseData[i] += requestData2[i];
    }

    float* actual_output = (float*)content->data();
    float* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(float);
    EXPECT_EQ(actual_output[0], expected_output[0]);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));
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
    return readImage("/ovms/src/test/binaryutils/rgb.jpg", filesize, image_bytes);
}

void read4x4RgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    return readImage("/ovms/src/test/binaryutils/rgb4x4.jpg", filesize, image_bytes);
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
void prepareKFSInferInputTensor(::KFSRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const ovms::Precision>& inputInfo,
    const std::vector<float>& data, bool putBufferInInputTensorContent) {
    auto [shape, type] = inputInfo;
    prepareKFSInferInputTensor(request, name,
        {shape, ovmsPrecisionToKFSPrecision(type)},
        data, putBufferInInputTensorContent);
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

void prepareKFSInferInputTensor(::KFSRequest& request, const std::string& name, const std::tuple<ovms::signed_shape_t, const std::string>& inputInfo,
    const std::vector<float>& data, bool putBufferInInputTensorContent) {
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
        // uint64_contents
        case ovms::Precision::U64: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_uint64_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        // uint_contents
        case ovms::Precision::U8:
        case ovms::Precision::U16:
        case ovms::Precision::U32: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_uint_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        // int64_contents
        case ovms::Precision::I64: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_int64_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        // bool_contents
        case ovms::Precision::BOOL: {
            for (size_t i = 0; i < dataSize; ++i) {
                auto ptr = tensor->mutable_contents()->mutable_bool_contents()->Add();
                *ptr = (data.size() ? data[i] : 1);
            }
            break;
        }
        // int_contents
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

std::shared_ptr<const TensorInfo> createTensorInfoCopyWithPrecision(std::shared_ptr<const TensorInfo> src, ovms::Precision newPrecision) {
    return std::make_shared<TensorInfo>(
        src->getName(),
        src->getMappedName(),
        newPrecision,
        src->getShape(),
        src->getLayout());
}
