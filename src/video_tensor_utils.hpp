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
#pragma once

#include <string>
#include <openvino/openvino.hpp>

namespace ovms {

class Status;

/**
 * @brief Creates an OpenVINO tensor from a video file path
 * 
 * This function reads all frames from a video file and creates a tensor
 * with shape [N, H, W, C] where N is the number of frames.
 * 
 * @param filePath Path to the video file
 * @param outputTensor Reference to the output tensor that will be populated
 * @return Status indicating success or failure
 */
Status makeVideoTensorFromPath(const std::string& filePath, ov::Tensor& outputTensor);

/**
 * @brief Creates an OpenVINO tensor from video data in memory
 * 
 * This function reads all frames from video data stored in memory and creates a tensor
 * with shape [N, H, W, C] where N is the number of frames.
 * 
 * @param videoData Video file content stored in memory as a string
 * @param outputTensor Reference to the output tensor that will be populated
 * @return Status indicating success or failure
 */
Status makeVideoTensorFromMemory(const std::string& videoData, ov::Tensor& outputTensor);

}  // namespace ovms