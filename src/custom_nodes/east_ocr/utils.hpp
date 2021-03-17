//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <string>
#include <utility>
#include <vector>

#include "../../custom_node_interface.h"
#include "opencv2/opencv.hpp"

#define NODE_ASSERT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
        return 1;                                                         \
    }

template <typename T>
std::vector<T> reorder_to_nhwc(const T* nchwVector, int rows, int cols, int channels) {
    std::vector<T> nhwcVector(rows * cols * channels);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                nhwcVector[y * channels * cols + x * channels + c] = nchwVector[c * (rows * cols) + y * cols + x];
            }
        }
    }
    return std::move(nhwcVector);
}

template <typename T>
std::vector<T> reorder_to_nchw(const T* nhwcVector, int rows, int cols, int channels) {
    std::vector<T> nchwVector(rows * cols * channels);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                nchwVector[c * (rows * cols) + y * cols + x] = reinterpret_cast<const T*>(nhwcVector)[y * channels * cols + x * channels + c];
            }
        }
    }
    return std::move(nchwVector);
}

const cv::Mat nhwc_to_mat(const CustomNodeTensor* input) {
    uint64_t height = input->dims[1];
    uint64_t width = input->dims[2];
    return cv::Mat(height, width, CV_32FC3, input->data);
}

const cv::Mat nchw_to_mat(const CustomNodeTensor* input) {
    uint64_t channels = input->dims[1];
    uint64_t rows = input->dims[2];
    uint64_t cols = input->dims[3];
    auto nhwcVector = reorder_to_nhwc<float>((float*)input->data, rows, cols, channels);

    cv::Mat image(rows, cols, CV_32FC3);
    std::memcpy(image.data, nhwcVector.data(), nhwcVector.size() * sizeof(float));
    return image;
}

bool crop_and_resize(cv::Mat originalImage, cv::Mat& targetImage, cv::Rect roi, cv::Size targetShape) {
    try {
        cv::resize(originalImage(roi), targetImage, targetShape);
    } catch (const cv::Exception& e) {
        std::cout << e.what() << std::endl;
        return false;
    }
    return true;
}

cv::Mat apply_grayscale(cv::Mat image) {
    cv::Mat grayscaled;
    cv::cvtColor(image, grayscaled, cv::COLOR_BGR2GRAY);
    return grayscaled;
}

float get_float_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, float defaultValue = 0.0f) {
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                return std::stof(params[i].value);
            } catch (std::invalid_argument& e) {
                return defaultValue;
            } catch (std::out_of_range& e) {
                return defaultValue;
            }
        }
    }
    return defaultValue;
}

int get_int_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, int defaultValue = 0) {
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                return std::stoi(params[i].value);
            } catch (std::invalid_argument& e) {
                return defaultValue;
            } catch (std::out_of_range& e) {
                return defaultValue;
            }
        }
    }
    return defaultValue;
}

std::string get_string_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, const std::string& defaultValue = "") {
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            return params[i].value;
        }
    }
    return defaultValue;
}
