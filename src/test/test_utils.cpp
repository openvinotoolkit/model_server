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
#include <fstream>
#include <unordered_set>

#include "platform_utils.hpp"

#include "src/tensorinfo.hpp"

using ovms::TensorInfo;


void printTensor(const ov::Tensor& tensor) {
    const auto& elementType = tensor.get_element_type();
    const void* dataPtr = tensor.data();

    size_t limit = 20;
    if (tensor.get_size() < limit) {
        limit = tensor.get_size();
    }
    if (elementType == ov::element::f32) {
        const float* data = static_cast<const float*>(dataPtr);
        std::cout << "Tensor data (f32): ";
        for (size_t i = 0; i < limit; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        return;
    } else if (elementType == ov::element::i32) {
        const int32_t* data = static_cast<const int32_t*>(dataPtr);
        std::cout << "Tensor data (i32): ";
        for (size_t i = 0; i < limit; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        return;
    } else if (elementType == ov::element::i64) {
        const int64_t* data = static_cast<const int64_t*>(dataPtr);
        std::cout << "Tensor data (i64): ";
        for (size_t i = 0; i < limit; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        return;
    } else if (elementType == ov::element::f64) {
        const double* data = static_cast<const double*>(dataPtr);
        std::cout << "Tensor data (f64): ";
        for (size_t i = 0; i < limit; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        return;
    }

    std::cout << "[ERROR] Unsupported data type: " << elementType << std::endl;
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

void RemoveReadonlyFileAttributeFromDir(std::string& directoryPath) {
    for (const std::filesystem::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(directoryPath)) {
        std::filesystem::permissions(dir_entry, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec | std::filesystem::perms::group_read | std::filesystem::perms::group_write | std::filesystem::perms::others_read, std::filesystem::perm_options::add);
    }
}

void SetReadonlyFileAttributeFromDir(std::string& directoryPath) {
    for (const std::filesystem::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(directoryPath)) {
        std::filesystem::permissions(dir_entry, std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec | std::filesystem::perms::group_write, std::filesystem::perm_options::remove);
        std::filesystem::permissions(dir_entry, std::filesystem::perms::owner_read | std::filesystem::perms::group_read | std::filesystem::perms::others_read, std::filesystem::perm_options::add);
    }
}

void readFile(const std::string& path, size_t& filesize, std::unique_ptr<char[]>& bytes) {
    std::ifstream DataFile;
    DataFile.open(path, std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    filesize = DataFile.tellg();
    DataFile.seekg(0);
    bytes = std::make_unique<char[]>(filesize);
    DataFile.read(bytes.get(), filesize);
}

void readRgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    return readFile(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb.jpg"), filesize, image_bytes);
}

void read4x4RgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    return readFile(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb4x4.jpg"), filesize, image_bytes);
}

std::shared_ptr<const TensorInfo> createTensorInfoCopyWithPrecision(std::shared_ptr<const TensorInfo> src, ovms::Precision newPrecision) {
    return std::make_shared<TensorInfo>(
        src->getName(),
        src->getMappedName(),
        newPrecision,
        src->getShape(),
        src->getLayout());
}

void adjustConfigToAllowModelFileRemovalWhenLoaded(ovms::ModelConfig& modelConfig) {
#ifdef _WIN32
    modelConfig.setPluginConfig(ovms::plugin_config_t({{"ENABLE_MMAP", "NO"}}));
#endif
    // on linux we can remove files from disk even if mmap is enabled
}
