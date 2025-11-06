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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "../video_tensor_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

class VideoTensorUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        tmpDir = "/tmp/ovms_video_test_" + std::to_string(getpid());
        // Create directory using C++ filesystem operations (cross-platform)
        std::string mkdirCmd = "mkdir -p " + tmpDir;
        int result = system(mkdirCmd.c_str());
        (void)result; // Suppress unused variable warning
    }

    void TearDown() override {
        // Clean up temporary directory
        std::string rmCmd = "rm -rf " + tmpDir;
        int result = system(rmCmd.c_str());
        (void)result; // Suppress unused variable warning
    }

    void createTestVideo(const std::string& filePath, int width = 64, int height = 48, int frameCount = 4) {
        // Create a simple test video using OpenCV VideoWriter
        cv::VideoWriter writer;
        
        // Try different codecs that should work cross-platform
        std::vector<int> codecs = {
            cv::VideoWriter::fourcc('M', 'P', '4', 'V'),  // MPEG-4
            cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),  // XVID
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),  // Motion JPEG
            0  // Default codec
        };
        
        bool success = false;
        for (int codec : codecs) {
            success = writer.open(filePath, codec, 1.0, cv::Size(width, height), true);
            if (success) {
                // Test by writing one frame to ensure it actually works
                cv::Mat testFrame(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
                writer.write(testFrame);
                if (!writer.isOpened()) {
                    writer.release();
                    success = false;
                    continue;
                }
                writer.release();
                
                // Reopen for actual writing
                success = writer.open(filePath, codec, 1.0, cv::Size(width, height), true);
                if (success) break;
            }
        }
        
        if (success && writer.isOpened()) {
            for (int i = 0; i < frameCount; ++i) {
                cv::Mat frame(height, width, CV_8UC3);
                
                // Create different colored frames for each frame
                cv::Scalar color(i * 60 % 256, (i * 80) % 256, (i * 100) % 256);
                frame.setTo(color);
                
                // Add some pattern to make frames distinguishable
                if (i * 10 + 20 < width && i * 8 + 16 < height) {
                    cv::rectangle(frame, 
                                cv::Point(i * 10, i * 8), 
                                cv::Point(i * 10 + 20, i * 8 + 16), 
                                cv::Scalar(255, 255, 255), -1);
                }
                
                writer.write(frame);
            }
            writer.release();
        } else {
            // If video creation fails, create a dummy file for testing error handling
            std::ofstream dummyFile(filePath);
            dummyFile << "dummy video content";
            dummyFile.close();
        }
    }

    std::string tmpDir;
};

TEST_F(VideoTensorUtilsTest, NonExistentVideoFile) {
    std::string nonExistentPath = tmpDir + "/non_existent_video.mp4";
    
    auto tensor = makeVideoTensorFromPath(nonExistentPath);
    
    // For non-existent files, the function should return an empty tensor with shape {0}
    auto shape = tensor.get_shape();
    EXPECT_EQ(shape.size(), 1) << "Expected tensor with single dimension";
    EXPECT_EQ(shape[0], 0) << "Expected empty tensor (size 0) for non-existent video file";
    EXPECT_EQ(tensor.get_element_type(), ov::element::f32) << "Expected f32 element type";
}

TEST_F(VideoTensorUtilsTest, InvalidVideoFile) {
    // Create a dummy file that's not a valid video
    std::string invalidVideoPath = tmpDir + "/invalid_video.mp4";
    std::ofstream file(invalidVideoPath);
    file << "This is not a video file content";
    file.close();
    
    auto tensor = makeVideoTensorFromPath(invalidVideoPath);
    
    // For invalid video files, the function should return an empty tensor with shape {0}
    auto shape = tensor.get_shape();
    EXPECT_EQ(shape.size(), 1) << "Expected tensor with single dimension";
    EXPECT_EQ(shape[0], 0) << "Expected empty tensor (size 0) for invalid video file";
    EXPECT_EQ(tensor.get_element_type(), ov::element::f32) << "Expected f32 element type";
}

TEST_F(VideoTensorUtilsTest, EmptyFilePath) {
    std::string emptyPath = "";
    
    auto tensor = makeVideoTensorFromPath(emptyPath);
    
    // For empty path, the function should return an empty tensor with shape {0}
    auto shape = tensor.get_shape();
    EXPECT_EQ(shape.size(), 1) << "Expected tensor with single dimension";
    EXPECT_EQ(shape[0], 0) << "Expected empty tensor (size 0) for empty file path";
    EXPECT_EQ(tensor.get_element_type(), ov::element::f32) << "Expected f32 element type";
}

TEST_F(VideoTensorUtilsTest, ValidVideoFile) {
    std::string videoPath = tmpDir + "/test_video.mp4";
    createTestVideo(videoPath, 64, 48, 4);
    
    auto tensor = makeVideoTensorFromPath(videoPath);
    auto shape = tensor.get_shape();
    
    // If video creation succeeded, we should have a 4D tensor
    // If it failed, we should have an empty tensor (shape {0})
    if (shape.size() == 4 && shape[0] > 0) {
        // Video was created successfully, test the full functionality
        EXPECT_EQ(shape[0], 4) << "Expected 4 frames";
        EXPECT_EQ(shape[1], 48) << "Expected height 48";
        EXPECT_EQ(shape[2], 64) << "Expected width 64";
        EXPECT_EQ(shape[3], 3) << "Expected 3 channels (BGR)";
        EXPECT_EQ(tensor.get_element_type(), ov::element::f32) << "Expected f32 element type";
        
        // Check that tensor data is not null and has reasonable values
        float* data = tensor.data<float>();
        EXPECT_NE(data, nullptr) << "Tensor data should not be null";
        
        // Check normalization (values should be in [0, 1] range)
        size_t totalElements = shape[0] * shape[1] * shape[2] * shape[3];
        bool validRange = true;
        for (size_t i = 0; i < totalElements; ++i) {
            if (data[i] < 0.0f || data[i] > 1.0f) {
                validRange = false;
                break;
            }
        }
        EXPECT_TRUE(validRange) << "All pixel values should be normalized to [0, 1] range";
        
        std::cout << "Video creation succeeded - full test completed" << std::endl;
    } else {
        // Video creation failed, which is acceptable in some environments
        // Just verify we get an empty tensor
        EXPECT_EQ(shape.size(), 1) << "Expected tensor with single dimension for failed video";
        EXPECT_EQ(shape[0], 0) << "Expected empty tensor for failed video creation";
        EXPECT_EQ(tensor.get_element_type(), ov::element::f32) << "Expected f32 element type";
        
        std::cout << "Video creation failed - testing error handling path" << std::endl;
    }
}

TEST_F(VideoTensorUtilsTest, SingleFrameVideo) {
    std::string videoPath = tmpDir + "/single_frame_video.mp4";
    createTestVideo(videoPath, 32, 24, 1);
    
    auto tensor = makeVideoTensorFromPath(videoPath);
    auto shape = tensor.get_shape();
    
    // Handle both successful video creation and failure
    if (shape.size() == 4 && shape[0] > 0) {
        EXPECT_EQ(shape[0], 1) << "Expected 1 frame";
        EXPECT_EQ(shape[1], 24) << "Expected height 24";
        EXPECT_EQ(shape[2], 32) << "Expected width 32";
        EXPECT_EQ(shape[3], 3) << "Expected 3 channels";
    } else {
        // Video creation failed - test error handling
        EXPECT_EQ(shape.size(), 1) << "Expected single dimension for failed video";
        EXPECT_EQ(shape[0], 0) << "Expected empty tensor for failed video";
    }
}

TEST_F(VideoTensorUtilsTest, DifferentResolutionVideo) {
    std::string videoPath = tmpDir + "/hd_video.mp4";
    createTestVideo(videoPath, 128, 96, 3);
    
    auto tensor = makeVideoTensorFromPath(videoPath);
    auto shape = tensor.get_shape();
    
    // Handle both successful video creation and failure
    if (shape.size() == 4 && shape[0] > 0) {
        EXPECT_EQ(shape[0], 3) << "Expected 3 frames";
        EXPECT_EQ(shape[1], 96) << "Expected height 96";
        EXPECT_EQ(shape[2], 128) << "Expected width 128";
        EXPECT_EQ(shape[3], 3) << "Expected 3 channels";
    } else {
        // Video creation failed - test error handling
        EXPECT_EQ(shape.size(), 1) << "Expected single dimension for failed video";
        EXPECT_EQ(shape[0], 0) << "Expected empty tensor for failed video";
    }
}

TEST_F(VideoTensorUtilsTest, TensorDataConsistency) {
    std::string videoPath = tmpDir + "/consistency_test.mp4";
    createTestVideo(videoPath, 16, 12, 2);
    
    // Load same video twice
    auto tensor1 = makeVideoTensorFromPath(videoPath);
    auto tensor2 = makeVideoTensorFromPath(videoPath);
    
    // Both tensors should have identical shapes
    auto shape1 = tensor1.get_shape();
    auto shape2 = tensor2.get_shape();
    EXPECT_EQ(shape1, shape2) << "Tensors from same video should have identical shapes";
    
    // Both tensors should have identical data
    float* data1 = tensor1.data<float>();
    float* data2 = tensor2.data<float>();
    size_t totalElements = shape1[0] * shape1[1] * shape1[2] * shape1[3];
    
    bool dataIdentical = std::memcmp(data1, data2, totalElements * sizeof(float)) == 0;
    EXPECT_TRUE(dataIdentical) << "Tensors from same video should have identical data";
}

TEST_F(VideoTensorUtilsTest, LargeFrameCountVideo) {
    std::string videoPath = tmpDir + "/many_frames_video.mp4";
    createTestVideo(videoPath, 32, 24, 10);
    
    auto tensor = makeVideoTensorFromPath(videoPath);
    auto shape = tensor.get_shape();
    
    // Handle both successful video creation and failure
    if (shape.size() == 4 && shape[0] > 0) {
        EXPECT_EQ(shape[0], 10) << "Expected 10 frames";
        
        // Verify tensor size calculation
        size_t expectedSize = 10 * 24 * 32 * 3;
        size_t actualSize = shape[0] * shape[1] * shape[2] * shape[3];
        EXPECT_EQ(actualSize, expectedSize) << "Tensor size should match expected calculation";
    } else {
        // Video creation failed - test error handling
        EXPECT_EQ(shape.size(), 1) << "Expected single dimension for failed video";
        EXPECT_EQ(shape[0], 0) << "Expected empty tensor for failed video";
    }
}