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
#include "test_request_utils_kfs.hpp"

#include <fstream>
#include <sstream>

#include <spdlog/spdlog.h>

#include "platform_utils.hpp"

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

void assertStringOutputProto(const KFSTensorOutputProto& proto, const std::vector<std::string>& expectedStrings) {
    ASSERT_EQ(proto.contents().bytes_contents_size(), expectedStrings.size());
    for (size_t i = 0; i < expectedStrings.size(); i++) {
        ASSERT_EQ(proto.contents().bytes_contents(i), expectedStrings[i]);
    }
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
