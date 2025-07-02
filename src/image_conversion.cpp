//****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "image_conversion.hpp"

#include <iostream>
#include <variant>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "logging.hpp"
#include "profiler.hpp"
// FIXME check compilation warnings
#pragma warning(push)
#pragma warning(disable : 6262 6386 6385)
#include "stb_image.h"        // NOLINT
#include "stb_image_write.h"  // NOLINT
#pragma warning(default : 6262)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#pragma warning(pop)

namespace ovms {

ov::Tensor loadImageStbi(unsigned char* image, const int x, const int y, const int desiredChannels) {
    if (!image) {
        std::stringstream errorMessage;
        errorMessage << stbi_failure_reason();
        throw std::runtime_error{errorMessage.str()};
    }
    struct SharedImageAllocator {
        unsigned char* image;
        int channels, height, width;
        void* allocate(size_t bytes, size_t) const {
            if (image && channels * height * width == bytes) {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void*, size_t bytes, size_t) {
            if (channels * height * width != bytes) {
                throw std::runtime_error{"Unexpected number of bytes was requested to deallocate."};
            }
            if (image != nullptr) {
                stbi_image_free(image);
                image = nullptr;
            }
        }
        bool is_equal(const SharedImageAllocator& other) const noexcept { return this == &other; }
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desiredChannels)},
        SharedImageAllocator{image, desiredChannels, y, x});
}

ov::Tensor loadImageStbiFromMemory(const std::string& imageBytes) {
    int x = 0, y = 0, channelsInFile = 0;
    constexpr int desiredChannels = 3;
    unsigned char* image = stbi_load_from_memory(
        (const unsigned char*)imageBytes.data(), imageBytes.size(),
        &x, &y, &channelsInFile, desiredChannels);
    return loadImageStbi(image, x, y, desiredChannels);
}

ov::Tensor loadImageStbiFromFile(char const* filename) {
    int x = 0, y = 0, channelsInFile = 0;
    constexpr int desiredChannels = 3;
    unsigned char* image = stbi_load(
        filename,
        &x, &y, &channelsInFile, desiredChannels);
    return loadImageStbi(image, x, y, desiredChannels);
}

std::vector<std::string> saveImagesStbi(const ov::Tensor& tensor) {
    // Validate tensor properties
    if (tensor.get_element_type() != ov::element::u8) {
        throw std::runtime_error{"Only U8 tensor element type is supported for image saving"};
    }
    if (tensor.get_shape().size() != 4) {
        throw std::runtime_error{"Tensor must be in NHWC format with batch size 1"};
    }
    size_t batchSize = tensor.get_shape()[0];
    size_t height = tensor.get_shape()[1];
    size_t width = tensor.get_shape()[2];
    size_t channels = tensor.get_shape()[3];
    size_t imageSize = height * width * channels;

    if (channels != 3 && channels != 1) {
        throw std::runtime_error{"Only 1 or 3 channel images are supported for saving"};
    }
    if (batchSize == 0) {
        throw std::runtime_error{"Tensor batch size cannot be zero"};
    }

    unsigned char* imageData = tensor.data<unsigned char>();

    // Create a memory buffer to hold the PNG data
    std::vector<std::vector<unsigned char>> pngBuffers(batchSize);

    // Define the write function that will store data in our buffer
    auto writeFunc = [](void* context, void* data, int size) {
        std::vector<unsigned char>* buffer = static_cast<std::vector<unsigned char>*>(context);
        unsigned char* bytes = static_cast<unsigned char*>(data);
        buffer->insert(buffer->end(), bytes, bytes + size);
    };

    // Write PNG to memory using our buffer
    for (size_t i = 0; i < batchSize; ++i) {
        int success = stbi_write_png_to_func(
            writeFunc,                    // Our write function
            &pngBuffers.at(i),            // Context (our buffer)
            width,                        // Image width
            height,                       // Image height
            channels,                     // Number of channels
            imageData + (i * imageSize),  // Image data
            width * channels);            // Stride (bytes per row)
        if (!success) {
            throw std::runtime_error{"Failed to encode image to PNG format"};
        }
    }

    // Convert the buffer to a string
    std::vector<std::string> result;
    for (const auto& png_buffer : pngBuffers) {
        result.emplace_back(png_buffer.begin(), png_buffer.end());
    }
    return result;
}
}  // namespace ovms
