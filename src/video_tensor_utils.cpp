//*****************************************************************************
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
#include "video_tensor_utils.hpp"

#include <vector>
#include <cstring>
#include "deps/opencv.hpp"

#include "profiler.hpp"
#include "logging.hpp"

namespace ovms {

ov::Tensor makeVideoTensorFromPath(const std::string& filePath) {
    OVMS_PROFILE_FUNCTION();
    
    cv::VideoCapture cap(filePath);
    if (!cap.isOpened()) {
        SPDLOG_DEBUG("Error opening video file: {}", filePath);
        // Return a properly initialized empty tensor
        return ov::Tensor(ov::element::f32, ov::Shape{0});
    }
    
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    
    try {
        // Read all frames from the video
        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }
            // Clone the frame to ensure we have our own copy
            frames.push_back(frame.clone());
        }
    } catch (const cv::Exception& e) {
        SPDLOG_DEBUG("Error during video frame reading: {}", e.what());
        cap.release();
        // Return a properly initialized empty tensor
        return ov::Tensor(ov::element::f32, ov::Shape{0});
    }
    
    cap.release();
    
    if (frames.empty()) {
        SPDLOG_DEBUG("No frames found in video file: {}", filePath);
        // Return a properly initialized empty tensor
        return ov::Tensor(ov::element::f32, ov::Shape{0});
    }
    
    // Create tensor shape [N, H, W, C]
    ov::Shape shape = {
        frames.size(),                        // N - number of frames
        static_cast<size_t>(frames[0].rows),  // H - height
        static_cast<size_t>(frames[0].cols),  // W - width
        static_cast<size_t>(frames[0].channels()) // C - channels
    };
    
    // Use FP32 precision as default (can be modified based on requirements)
    ov::element::Type precision = ov::element::f32;
    ov::Tensor tensor(precision, shape);
    
    // Copy frame data to tensor
    char* ptr = (char*)tensor.data();
    for (const cv::Mat& img : frames) {
        // Convert to float if needed
        cv::Mat floatImg;
        if (img.type() != CV_32FC3 && img.channels() == 3) {
            img.convertTo(floatImg, CV_32FC3, 1.0/255.0); // Normalize to [0,1] range
        } else if (img.type() != CV_32FC1 && img.channels() == 1) {
            img.convertTo(floatImg, CV_32FC1, 1.0/255.0); // Normalize to [0,1] range
        } else {
            floatImg = img;
        }
        
        memcpy(ptr, (char*)floatImg.data, floatImg.total() * floatImg.elemSize());
        ptr += (floatImg.total() * floatImg.elemSize());
    }
    
    return tensor;
}

}  // namespace ovms