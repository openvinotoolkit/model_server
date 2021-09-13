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

#include <cmath>
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

#define NODE_EXPECT(cond, msg)                                            \
    if (!(cond)) {                                                        \
        std::cout << "[" << __LINE__ << "] Assert: " << msg << std::endl; \
    }

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

bool crop_rotate_resize(cv::Mat originalImage, cv::Mat& targetImage, cv::Rect roi, float angle, float originalTextWidth, float originalTextHeight, cv::Size targetShape) {
    cv::Mat cropped = originalImage(roi);
    cv::Mat rotated;
    if (angle != 0.0) {
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(cropped.size().width / 2, cropped.size().height / 2), angle, 1.0);
        cv::warpAffine(cropped, rotated, rotationMatrix, cropped.size());
    } else {
        rotated = cropped;
    }

    try {
        cv::Mat rotatedSlicedImage;
        if (angle != 0.0) {
            int sliceOffset = (rotated.size().height - originalTextHeight) / 2;
            rotatedSlicedImage = rotated(cv::Rect(0, sliceOffset, rotated.size().width, originalTextHeight));
        } else {
            rotatedSlicedImage = rotated;
        }
        cv::resize(rotatedSlicedImage, targetImage, targetShape);
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

bool scale_image(
    bool isScaleDefined,
    const float scale,
    const std::vector<float>& meanValues,
    const std::vector<float>& scaleValues,
    cv::Mat& image) {
    if (!isScaleDefined && scaleValues.size() == 0 && meanValues.size() == 0) {
        return true;
    }

    size_t colorChannels = static_cast<size_t>(image.channels());
    if (meanValues.size() > 0 && meanValues.size() != colorChannels) {
        return false;
    }
    if (scaleValues.size() > 0 && scaleValues.size() != colorChannels) {
        return false;
    }

    std::vector<cv::Mat> channels;
    if (meanValues.size() > 0 || scaleValues.size() > 0) {
        cv::split(image, channels);
        if (channels.size() != colorChannels) {
            return false;
        }
    } else {
        channels.emplace_back(image);
    }

    for (size_t i = 0; i < meanValues.size(); i++) {
        channels[i] -= meanValues[i];
    }

    if (scaleValues.size() > 0) {
        for (size_t i = 0; i < channels.size(); i++) {
            channels[i] /= scaleValues[i];
        }
    } else if (isScaleDefined) {
        for (size_t i = 0; i < channels.size(); i++) {
            channels[i] /= scale;
        }
    }

    if (channels.size() == 1) {
        image = channels[0];
    } else {
        cv::merge(channels, image);
    }

    return true;
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

float get_float_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount, bool& isDefined, float defaultValue = 0.0f) {
    isDefined = true;
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            try {
                return std::stof(params[i].value);
            } catch (std::invalid_argument& e) {
                isDefined = false;
                return defaultValue;
            } catch (std::out_of_range& e) {
                isDefined = false;
                return defaultValue;
            }
        }
    }
    isDefined = false;
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

std::vector<float> get_float_list_parameter(const std::string& name, const struct CustomNodeParam* params, int paramsCount) {
    std::string listStr;
    for (int i = 0; i < paramsCount; i++) {
        if (name == params[i].key) {
            listStr = params[i].value;
            break;
        }
    }

    if (listStr.length() < 2 || listStr.front() != '[' || listStr.back() != ']') {
        return {};
    }

    listStr = listStr.substr(1, listStr.size() - 2);

    std::vector<float> result;

    std::stringstream lineStream(listStr);
    std::string element;
    while (std::getline(lineStream, element, ',')) {
        try {
            float e = std::stof(element.c_str());
            result.push_back(e);
        } catch (std::invalid_argument& e) {
            NODE_EXPECT(false, "error parsing list parameter");
            return {};
        } catch (std::out_of_range& e) {
            NODE_EXPECT(false, "error parsing list parameter");
            return {};
        }
    }

    return result;
}

std::string floatListToString(const std::vector<float>& values) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            ss << ",";
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}
