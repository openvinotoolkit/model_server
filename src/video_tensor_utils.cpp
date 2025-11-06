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
#include <unistd.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "profiler.hpp"
#include "logging.hpp"
#include "status.hpp"

namespace ovms {

Status makeVideoTensorFromPath(const std::string& filePath, ov::Tensor& outputTensor) {
    OVMS_PROFILE_FUNCTION();
    
    if (filePath.empty()) {
        SPDLOG_DEBUG("Empty file path provided");
        return Status(StatusCode::FILE_INVALID, "Video file path is empty");
    }
    
    cv::VideoCapture cap(filePath);
    if (!cap.isOpened()) {
        SPDLOG_DEBUG("Error opening video file: {}", filePath);
        return Status(StatusCode::FILE_INVALID, "Cannot open video file: " + filePath);
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
        return Status(StatusCode::INTERNAL_ERROR, "OpenCV error during video reading: " + std::string(e.what()));
    }
    
    cap.release();
    
    if (frames.empty()) {
        SPDLOG_DEBUG("No frames found in video file: {}", filePath);
        return Status(StatusCode::FILE_INVALID, "Video file contains no frames: " + filePath);
    }
    
    try {
        // Create tensor shape [N, H, W, C]
        ov::Shape shape = {
            frames.size(),                        // N - number of frames
            static_cast<size_t>(frames[0].rows),  // H - height
            static_cast<size_t>(frames[0].cols),  // W - width
            static_cast<size_t>(frames[0].channels()) // C - channels
        };
        
        // Use FP32 precision as default (can be modified based on requirements)
        ov::element::Type precision = ov::element::f32;
        outputTensor = ov::Tensor(precision, shape);
        
        // Copy frame data to tensor
        char* ptr = (char*)outputTensor.data();
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
        
        return Status(StatusCode::OK);
        
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("Error creating tensor from video frames: {}", e.what());
        return Status(StatusCode::INTERNAL_ERROR, "Failed to create tensor from video frames: " + std::string(e.what()));
    }
}

Status makeVideoTensorFromMemory(const std::string& videoData, ov::Tensor& outputTensor) {
    OVMS_PROFILE_FUNCTION();
    
    if (videoData.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Empty video data provided");
        return Status(StatusCode::FILE_INVALID, "Empty video data provided");
    }

    try {
        // Create a temporary file to write the video data
        // OpenCV doesn't support reading video directly from memory buffer
        // so we need to create a temporary file
        char tempFileName[] = "/tmp/ovms_video_XXXXXX.mp4";
        int fd = mkstemps(tempFileName, 4);  // 4 for ".mp4" extension
        if (fd == -1) {
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to create temporary file for video data");
            return Status(StatusCode::INTERNAL_ERROR, "Failed to create temporary file");
        }
        
        // Write video data to temporary file
        ssize_t bytesWritten = write(fd, videoData.data(), videoData.size());
        close(fd);
        
        if (bytesWritten != static_cast<ssize_t>(videoData.size())) {
            unlink(tempFileName);  // Clean up temporary file
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to write video data to temporary file");
            return Status(StatusCode::INTERNAL_ERROR, "Failed to write video data to temporary file");
        }
        
        // Use the existing function to process the temporary file
        Status status = makeVideoTensorFromPath(std::string(tempFileName), outputTensor);
        
        // Clean up temporary file
        unlink(tempFileName);
        
        return status;
        
    } catch (const cv::Exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "OpenCV exception in makeVideoTensorFromMemory: {}", e.what());
        return Status(StatusCode::INTERNAL_ERROR, "OpenCV error during video processing");
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Exception in makeVideoTensorFromMemory: {}", e.what());
        return Status(StatusCode::INTERNAL_ERROR, "Error during video processing");
    }
}

}  // namespace ovms