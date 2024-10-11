//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "ocl_utils.hpp"

#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/remote_tensor.hpp>

#include "logging.hpp"

namespace ovms {
cl_context getOCLContext() {
    cl_int err;

    // Step 1: Querying Platforms
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of platforms\n";
        throw 1;
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Detected {} openCL platforms.", numPlatforms);

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // Step 2: Querying Devices
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of devices\n";
        throw 1;
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Detected {} openCL GPU devices.", numDevices);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting GPU device\n";
        throw 1;
    }

    // Step 3: Creating a Context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context\n";
        throw 1;
    }
    return context;
}
cl_context get_cl_context(cl_platform_id& platformId, cl_device_id& deviceId) {
    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of platforms\n";
        throw 1;
    }
    // extract 1st platform from numPlatforms
    clGetPlatformIDs(1, &platformId, nullptr);
    cl_uint numDevices = 0;
    // query how many devices there are
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of devices\n";
        throw 1;
    }
    if (0 == numDevices) {
        std::cerr << "There is no available devices\n";
        throw 1;
    }
    cl_uint numberOfDevicesInContext = 1;
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numberOfDevicesInContext, &deviceId, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting GPU deviceId\n";
        throw 1;
    }
    // since we only use 1 device we can use address of deviceId
    cl_context openCLCContext = clCreateContext(nullptr, numberOfDevicesInContext, &deviceId, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context\n";
        throw 1;
    }
    return openCLCContext;
}

}  // namespace ovms
