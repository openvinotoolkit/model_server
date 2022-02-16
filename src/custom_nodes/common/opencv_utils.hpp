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

#include <vector>

#include "../../custom_node_interface.h"
#include "opencv2/opencv.hpp"

template <typename T>
void reorder_to_nhwc_2(const T* sourceNchwBuffer, T* destNhwcBuffer, int rows, int cols, int channels) {
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                destNhwcBuffer[y * channels * cols + x * channels + c] = reinterpret_cast<const T*>(sourceNchwBuffer)[c * (rows * cols) + y * cols + x];
            }
        }
    }
}

template <typename T>
std::vector<T> reorder_to_nhwc(const T* nchwVector, int rows, int cols, int channels) {
    std::vector<T> nhwcVector(rows * cols * channels);
    reorder_to_nhwc_2(nchwVector, nhwcVector.data(), rows, cols, channels);
    return nhwcVector;
}

template <typename T>
void reorder_to_nchw_2(const T* sourceNhwcBuffer, T* destNchwBuffer, int rows, int cols, int channels) {
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                destNchwBuffer[c * (rows * cols) + y * cols + x] = reinterpret_cast<const T*>(sourceNhwcBuffer)[y * channels * cols + x * channels + c];
            }
        }
    }
}

template <typename T>
std::vector<T> reorder_to_nchw(const T* nhwcVector, int rows, int cols, int channels) {
    std::vector<T> nchwVector(rows * cols * channels);
    reorder_to_nchw_2(nhwcVector, nchwVector.data(), rows, cols, channels);
    return nchwVector;
}

const cv::Mat nhwc_to_mat(const CustomNodeTensor* input);

const cv::Mat nchw_to_mat(const CustomNodeTensor* input);

bool crop_rotate_resize(cv::Mat originalImage, cv::Mat& targetImage, cv::Rect roi, float angle, float originalTextWidth, float originalTextHeight, cv::Size targetShape);

cv::Mat apply_grayscale(cv::Mat image);

bool scale_image(
    bool isScaleDefined,
    const float scale,
    const std::vector<float>& meanValues,
    const std::vector<float>& scaleValues,
    cv::Mat& image);
