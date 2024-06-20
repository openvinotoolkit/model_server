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
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <CL/cl2.hpp>
//#include <CL/opencl.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>

#include "../ov_utils.hpp"
#include "../status.hpp"
#include "c_api_test_utils.hpp"
#include "ocl_utils.hpp"
#include "test_utils.hpp"

using namespace ov;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

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

TEST(OpenVINO, ExtractContextFromModel) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    ov::AnyMap config = {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
        ov::auto_batch_timeout(0)};
    cl_platform_id platformId;
    cl_device_id deviceId;
    auto nonused = get_cl_context(platformId, deviceId);
    auto compiledModel = core.compile_model(model, "GPU", config);
    auto gpu_context = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctxFromModel = gpu_context.get();
    cl::Context openCLCppContext(ctxFromModel);
    cl_int err;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err);
    auto inputOVOCLBufferTensor = gpu_context.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = gpu_context.create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(10, 0.1);
    void* inputBufferData = in.data();
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    cl::Device device(deviceId);
    auto queue = cl::CommandQueue(openCLCppContext, device);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.set_tensor(input, inputOVOCLBufferTensor);
    inferRequest.set_tensor(output, outputOVOCLBufferTensor);
    inferRequest.infer();
    std::vector<float> out(10);
    void* buffer_out = out.data();
    queue.enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, outputByteSize, buffer_out);
    for (size_t i = 0; i < inputByteSize / sizeof(float); ++i) {
        // different precision on GPU vs CPU
        EXPECT_NEAR(in[i] + 1, out[i], 0.0004) << "i:" << i;
    }
}

TEST(OpenVINO, LoadModelWithPrecreatedContext) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    // we need byte size not no of elements
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Device device(deviceId);
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
    auto compiledModel = core.compile_model(model, remote_context);
    // now we create buffers
    cl::Context openCLCppContext(openCLCContext);
    cl_int err;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err);
    // create tensors and perform inference
    // wrap in and out buffers into RemoteTensor and set them to infer request
    auto inputOVOCLBufferTensor = remote_context.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = remote_context.create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(10, 0.1);
    void* inputBufferData = in.data();
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.set_tensor(input, inputOVOCLBufferTensor);
    inferRequest.set_tensor(output, outputOVOCLBufferTensor);
    inferRequest.infer();
    std::vector<float> out(10);
    void* buffer_out = out.data();
    queue.enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, outputByteSize, buffer_out);
    for (size_t i = 0; i < inputByteSize / sizeof(float); ++i) {
        // different precision on GPU vs CPU
        EXPECT_NEAR(in[i] + 1, out[i], 0.0004) << "i:" << i;
    }
}

struct CallbackUnblockingStruct {
    std::promise<uint32_t> signal;
    void* bufferAddr = nullptr;
    cl::CommandQueue* queue = nullptr;
};

TEST(OpenVINO, SetTensorTest) {
    size_t tSize = 10;
    int iterations = 10;
    iterations = 1'000;
    //std::vector<size_t> sizeSet{10, 10 * 10, 10 * 100, 10 * 1'000, 10 * 10'000, 10 * 100'000, 10 * 1'000'000};
    std::vector<size_t> sizeSet{1'000'000};
    // load model
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    const std::string inputName{"b"};
    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto outputByteSize = ov::shape_size(output->get_shape());
    // we need byte size not no of elements
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    enum {
        // DEV_CONTEXT_SCENARIO
        CPU_COPY,                         // regular OVMS scenario
        CPU_SET,                          // set output tensors to avoid copy
        GPU_OV_COPY_OV,                   // regular GPU OVMS scenario
        GPU_OV_SET_OV,                    // set regular ov tensors and use gpu for inference
        GPU_OCL_COPY,                     // model loadded with OCL use OV tensors on input and still copy output
        GPU_OCL_SET_OV,                   // set regular ov tensors and use gpu with passed in context for inference
        GPU_OCL_SET_OCL_IN_AND_OV_OUT,    // set ocl tensor on input and  ov tensors on output and use gpu with passed in context for inference
        GPU_OCL_SET_OCL,                  // set OCL tensors and use gpu with passed in context for inference
        GPU_OCL_DIFF_CONTEXT_INPUT_COPY,  // use OCL tensors on input but use different context
        GPU_OV_SET_OCL_DIFF_CONTEXT,      // set OCL tensors and use gpu for inference but model with default OV context
        GPU_OV_SET_OCL_SAME_CONTEXT,      // set OCL tensors with the default OV context with gpu
        GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME,
        GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL,
        GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS,
        GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS_CONCUR,
	GPU_OV_SET_VAA_BUF,
    };
    std::unordered_map<int, std::unordered_map<int, double>> times;
    for (auto tSize : sizeSet) {
        SPDLOG_ERROR("Performing tests for dummy shape (1,{}) ....", tSize);
        auto sizeStart = std::chrono::high_resolution_clock::now();
        ov::element::Type_t dtype = ov::element::Type_t::f32;
        ov::Shape ovShape;
        ovShape.emplace_back(1);
        ovShape.emplace_back(tSize);
        std::map<std::string, ov::PartialShape> inputShapes;
        inputShapes[inputName] = ovShape;
        model->reshape(inputShapes);
        auto gpuCompiledModel = core.compile_model(model, "GPU");
        auto gpuInferRequest = gpuCompiledModel.create_infer_request();
        std::vector<ov::InferRequest> gpuInferRequests;
        gpuInferRequests.emplace_back(gpuCompiledModel.create_infer_request());
        gpuInferRequests.emplace_back(gpuCompiledModel.create_infer_request());
        auto cpuCompiledModel = core.compile_model(model, "CPU");
        auto cpuInferRequest = cpuCompiledModel.create_infer_request();
        // prepare ov::Tensor data
        std::vector<ov::Tensor> inputOvTensors, outputOvTensors;
        inputOvTensors.emplace_back(dtype, ovShape);
        inputOvTensors.emplace_back(dtype, ovShape);
        outputOvTensors.emplace_back(dtype, ovShape);
        outputOvTensors.emplace_back(dtype, ovShape);
        auto inputByteSize = tSize;
        auto outputByteSize = tSize;
        inputByteSize *= sizeof(float);
        outputByteSize *= sizeof(float);

        cl_int err;
        std::vector<float> outputData(tSize, 0);
        {  // GPU_OCL_COPY model loaded with OCL context, using ov::Tensors on input & output (copy)
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                oclInferRequest.infer();
                const auto& outTensor = oclInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            SPDLOG_ERROR("finished GPU_OV_COPY_OV");
            times[GPU_OCL_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        }
        if (false) {  // GPU_OCL_DIFF_CONTEXT_INPUT_COPY model loaded with OCL context using OCL tensors on input from different context, copying output
            // not working
            // illegal [GPU] trying to reinterpret buffer allocated by a different engine
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl_context openCLCContextDifferent = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            cl::Context openCLCppContextDifferent(openCLCContextDifferent);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto ovWrappedOCLContextDifferent = ov::intel_gpu::ocl::ClContext(core, openCLCContextDifferent, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextDifferent, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextDifferent, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextDifferent.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextDifferent.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));

            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputs[i % 2]);
                oclInferRequest.infer();
                const auto& outTensor = oclInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OCL_DIFF_CONTEXT_INPUT_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_DIFF_CONTEXT_INPUT_COPY");
        }
        {  // GPU_OCL_SET_OCL using model loaded with OCL & tensor from the same context on both input & output
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));

            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputs[i % 2]);
                oclInferRequest.set_tensor(output, outputs[i % 2]);
                oclInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OCL_SET_OCL][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_SET_OCL");
        }
        // TODO FIXME
        {  // GPU_OCL_SET_OCL_IN_AND_OV_OUT using model loaded with OCL & tensor on input from the same context.Output using ov;:Tensor & copy
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputs[i % 2]);
                oclInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                oclInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OCL_SET_OCL_IN_AND_OV_OUT][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_SET_OCL_IN_AND_OV_OUT");
        }
        {  // GPU_OCL_SET_OV model loaded on gpu with both outpu & input being ov::Tensor
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                oclInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                oclInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OCL_SET_OV][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_SET_OV");
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                gpuInferRequest.infer();
                const auto& outTensor = gpuInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OV_COPY_OV][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OV_COPY_OV");
            // GPU set input & output
        }
        {  // GPU_OV_SET_OV inference with ov::Tensors but output is set as well
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                gpuInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                gpuInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OV][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_TRACE("finished GPU_OV_SET_OV");
        }
        {  // CPU_COPY inference with ov::Tensors - current (2024.1) OVMS flow with cpu
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                cpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                cpuInferRequest.infer();
                const auto& outTensor = cpuInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[CPU_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished CPU_COPY");
        }
        {  // CPU_SET inference with ov::Tensors but output is set as well
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                cpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                cpuInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                cpuInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[CPU_SET][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished CPU_SET");
        }
        if (false) {  // GPU_OV_SET_OCL_DIFF_CONTEXT model loaded with ov context and different ocl context used to create ocl tensors
            // illegal [GPU] trying to reinterpret buffer allocated by a different engine
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputs[i % 2]);
                gpuInferRequest.set_tensor(output, outputs[i % 2]);
                gpuInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OCL_DIFF_CONTEXT][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_DIFF_CONTEXT");
        }
        if (true) {  // GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME model loaded with ov context and different ocl context used to create ocl tensors
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContextFromModel = gpuCompiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputs[i % 2]);
                gpuInferRequest.set_tensor(output, outputs[i % 2]);
                gpuInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME");
        }
        {  // GPU_OV_SET_OCL_SAME_CONTEXT load model with target device and use context from model to create tensors
            auto ovWrappedOCLContextFromModel = gpuCompiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
            cl_context openCLCContextFromModel = ovWrappedOCLContextFromModel.get();
            bool retainObject = true;  // we need to retain here since its OV that will clean up
            cl::Context openCLCppContextFromModel(openCLCContextFromModel, retainObject);
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputs[i % 2]);
                gpuInferRequest.set_tensor(output, outputs[i % 2]);
                gpuInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OCL_SAME_CONTEXT][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_SAME_CONTEXT");
        }
        {  // GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL model loaded with ov context and different ocl context used to create ocl tensors
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContextFromModel = gpuCompiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
                inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[i % 2]));
                outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[i % 2]));
                gpuInferRequest.set_tensor(input, inputs[0]);
                gpuInferRequest.set_tensor(output, outputs[0]);
                gpuInferRequest.infer();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;  // ms
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL");
        }
        if (true) {  // GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS model loaded with ov context and different ocl context used to create ocl tensors
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContextFromModel = gpuCompiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            CallbackUnblockingStruct callbackStruct;
            auto unblockSignal = callbackStruct.signal.get_future();
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                SPDLOG_INFO("iter start");
                //std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
                //inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[i % 2]));
                //outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[i % 2]));
                ov::Tensor inputOVTensor = inputs[i % 2];
                ov::Tensor outputOVTensor = outputs[i % 2];
                //gpuInferRequest.set_tensor(input,inputOVTensor);
                //gpuInferRequest.set_tensor(output, outputOVTensor);
                gpuInferRequest.set_tensor(input, inputs[i % 2]);
                gpuInferRequest.set_tensor(output, outputs[i % 2]);
                gpuInferRequest.set_callback([&gpuInferRequest, &callbackStruct](std::exception_ptr exception) {
                    SPDLOG_INFO("entered callback");
                    gpuInferRequest.set_callback([](std::exception_ptr exception) {});
                    SPDLOG_INFO("callback start");
                    callbackStruct.signal.set_value(42);
                    SPDLOG_INFO("callback end");
                });
                SPDLOG_INFO("callback end");
                gpuInferRequest.start_async();
                SPDLOG_INFO("waiting to unblock");
                unblockSignal.get();
                SPDLOG_INFO("Unblocked thread");
                callbackStruct.signal = std::promise<uint32_t>();
                SPDLOG_INFO("reset promise");
                unblockSignal = callbackStruct.signal.get_future();
                SPDLOG_INFO("reset future");
                // gpuInferRequest.wait(); // TODO probably not required
                SPDLOG_INFO("iter end");
            }
            auto stop = std::chrono::high_resolution_clock::now();
            SPDLOG_ERROR("Log plugin");
            ovms::logOVPluginConfig([&gpuCompiledModel](const std::string& key) { return gpuCompiledModel.get_property(key); }, " {someAuthor} ", " {some details} ");
            SPDLOG_ERROR("Log plugin end");
            times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;  // ms
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS:{}", times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS][tSize]);
        }
	#ifdef TEST_VAAPI
        // TODO:
	// * no get_va_display function
	// * no allocate_image
        {  // GPU_OV_SET_VAA_BUF model loaded with ov context and vaapi tensors used
	    VADisplay display = get_va_display();
	    ov::intel_gpu::ocl::VAContext va_gpu_context(core, display);
            cl::Image2D y_plane_surface = allocate_image(y_plane_size);
            cl::Image2D uv_plane_surface = allocate_image(uv_plane_size);
            auto remote_tensor = gpu_context.create_tensor_nv12(y_plane_surface, uv_plane_surface);
            SPDLOG_ERROR("finished GPU_OV_SET_VAA_BUF:{}", times[GPU_OV_SET_VAA_BUF][tSize]);
	}
	#endif
        {  // GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS model loaded with ov context and different ocl context used to create ocl tensors
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContextFromModel = gpuCompiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            CallbackUnblockingStruct callbackStruct[2];
            std::vector<std::future<uint32_t>> unblockSignal;
            unblockSignal.emplace_back(callbackStruct[0].signal.get_future());
            unblockSignal.emplace_back(callbackStruct[1].signal.get_future());
            auto start = std::chrono::high_resolution_clock::now();
            auto j = 0;
            ;
            ov::Tensor inputOVTensor = inputs[j];
            ov::Tensor outputOVTensor = outputs[j];
            gpuInferRequest.set_tensor(input, inputs[j]);
            gpuInferRequest.set_tensor(output, outputs[j]);
            SPDLOG_INFO("set_callback");
            gpuInferRequest.set_callback([&gpuInferRequest, &callbackStruct, j](std::exception_ptr exception) {
                SPDLOG_INFO("entered callback");
                gpuInferRequest.set_callback([](std::exception_ptr exception) {});
                SPDLOG_INFO("callback start");
                callbackStruct[j].signal.set_value(42);
                SPDLOG_INFO("callback end");
            });
            SPDLOG_INFO("start async");
            gpuInferRequest.start_async();
            for (auto i = 0; i < iterations; ++i) {
                SPDLOG_INFO("iter start");
                auto j = (i + 1) % 2;
                auto& gpuInferRequest = gpuInferRequests[j];

                ov::Tensor inputOVTensor = inputs[j];
                ov::Tensor outputOVTensor = outputs[j];
                gpuInferRequest.set_tensor(input, inputs[j]);
                gpuInferRequest.set_tensor(output, outputs[j]);
                SPDLOG_INFO("set_callback");
                gpuInferRequest.set_callback([&gpuInferRequest, &callbackStruct, j](std::exception_ptr exception) {
                    SPDLOG_INFO("entered callback");
                    gpuInferRequest.set_callback([](std::exception_ptr exception) {});
                    SPDLOG_INFO("callback start");
                    callbackStruct[j].signal.set_value(42);
                    SPDLOG_INFO("callback end");
                });
                SPDLOG_INFO("start async");
                gpuInferRequest.start_async();
                // as we scheduled next infer we receive results from previous
                j = i % 2;
                SPDLOG_INFO("waiting to unblock");
                auto callbackReturnValue = unblockSignal[j].get();
                SPDLOG_INFO("Unblocked thread");
                callbackStruct[j].signal = std::promise<uint32_t>();
                SPDLOG_INFO("reset promise");
                unblockSignal[j] = callbackStruct[j].signal.get_future();
                SPDLOG_INFO("reset future");
                SPDLOG_INFO("iter end");
            }
            SPDLOG_ERROR("ER");
            auto callbackReturnValue = unblockSignal[iterations % 2].get();
            SPDLOG_ERROR("ER");
            auto stop = std::chrono::high_resolution_clock::now();
            SPDLOG_ERROR("Log plugin");
            ovms::logOVPluginConfig([&gpuCompiledModel](const std::string& key) { return gpuCompiledModel.get_property(key); }, " {someAuthor} ", " {some details} ");
            SPDLOG_ERROR("Log plugin end");
            times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS_CONCUR][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;  // ms
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS_CONCUR:{}", times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS_CONCUR][tSize]);
        }
        auto sizeStop = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(sizeStop - sizeStart).count() / 1000000.0;
        SPDLOG_ERROR("For size: {:8d} inferences all took {:03.5f} seconds. Next inferences will take probably ~x10 longer ...", tSize, totalTime);
    }
    std::cout << std::right;
    for (auto s : {"CPU_COPY", "CPU_SET", "GPU_OV_COPY_OV", "GPU_OV_SET_OV", "GPU_OCL_COPY", "GPU_OCL_SET_OV", "GPU_OCL_SET_OCL_IN_AND_OV_OUT", "GPU_OCL_SET_OCL", /*"GPU_OCL_DIFF_CONTEXT_INPUT_COPY", "GPU_OV_SET_OCL_DIFF_CONTEXT",*/ "GPU_OV_SET_OCL_SAME_CONTEXT", "GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME", "GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL", "GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS", "GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS_CONCUR"}) {
        std::cout << s << "[MePS]"
                  << "\t\t";
    }
    std::cout << std::endl;
    for (auto s : sizeSet) {
        for (auto t : {CPU_COPY, CPU_SET, GPU_OV_COPY_OV, GPU_OV_SET_OV, GPU_OCL_COPY, GPU_OCL_SET_OV, GPU_OCL_SET_OCL_IN_AND_OV_OUT, GPU_OCL_SET_OCL, /*GPU_OCL_DIFF_CONTEXT_INPUT_COPY, GPU_OV_SET_OCL_DIFF_CONTEXT,*/ GPU_OV_SET_OCL_SAME_CONTEXT, GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME, GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL, GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS, GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS_CONCUR}) {
            // times[ms] so we diide by 1000 to have per second
            double fps = iterations / (times[t][s] / 1000.);  //FPS[Frame/second]
            std::cout << "" << fps * s << " \t\t ";
        }
        std::cout << std::endl;
    }
}

#include "../ocl_utils.hpp"

TEST(CAPINonCopy, SetOpenCLBufferAsInputTensor) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, 42);
    void* inputBufferData = in.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    // start CAPI server
    // TODO load model with passed in context
    std::string port = "9000";
    randomizePort(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_gpu_dummy.json"));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t capiDeviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &capiDeviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);  // TODO ?
    EXPECT_EQ(capiDeviceId, 0);                  // TODO?
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(in[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
    OVMS_ServerDelete(cserver);
}

TEST(OpenCL, UseDifferentContextWhenReadingAndWritingToBuffer) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl_context openCLCContext2 = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Context openCLCppContext2(openCLCContext2);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    auto queue2 = cl::CommandQueue(openCLCppContext2, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, 42);
    void* inputBufferData = in.data();
    std::vector<float> out(10, 13.1);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    queue2.enqueueReadBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, outputBufferData);
    const float* outputData = reinterpret_cast<const float*>(outputBufferData);
    for (size_t i = 0; i < 10; ++i) {
        SPDLOG_ERROR("ER:{}", *(outputData + i));
    }
}


TEST(CAPINonCopy, SetOpenCLBufferAsInputAndOutputTensor) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, 42);
    void* inputBufferData = in.data();
    std::vector<float> out(10, 13.1);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    // start CAPI server
    // TODO load model with passed in context
    std::string port = "9000";
    randomizePort(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_gpu_dummy.json"));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));     // device id ?? TODO
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    OVMS_InferenceResponse* response = nullptr;
    SPDLOG_ERROR("ER");
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    SPDLOG_ERROR("ER");
    cl::vector<cl::Event> readEvents;
    SPDLOG_ERROR("ER");
    auto oclError = queue.enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, inputByteSize, outputBufferData, &readEvents);
    SPDLOG_ERROR("ER");
    readEvents[0].wait();
    SPDLOG_ERROR("ER:{}", oclError);
    SPDLOG_ERROR("ER");
    //TODO what to do if output set was not enough?
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t capiDeviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &capiDeviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_OPENCL);  // TODO ?
    EXPECT_EQ(capiDeviceId, 0);                  // TODO?
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    /*const float* outputData = reinterpret_cast<const float*>(outputBufferData);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(in[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }*/
    // TODO FIXME add output checking
    // TODO cleanup settings
    SPDLOG_ERROR("ER");
    OVMS_ServerDelete(cserver);
    SPDLOG_ERROR("ER");
}
static void callbackMarkingItWasUsedWith42(OVMS_InferenceResponse*, uint32_t flag, void* userstruct);
static void callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPICorrectness(OVMS_InferenceResponse*, uint32_t flag, void* userstruct);

const float INITIAL_VALUE{0.13666};
const float GARBAGE_VALUE = 42.66613;
const float FLOAT_TOLLERANCE{0.001};

TEST(CAPISyncWithCalback, DummyCallback) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, INITIAL_VALUE);
    void* inputBufferData = in.data();
    std::vector<float> out(10, GARBAGE_VALUE);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    // start CAPI server
    // TODO load model with passed in context
    std::string port = "9000";
    randomizePort(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_gpu_dummy.json"));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));     // device id ?? TODO
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    OVMS_InferenceResponse* response = nullptr;
    // set callback
    uint32_t callbackUsed = 31;

    CallbackUnblockingStruct callbackStruct;
    auto unblockSignal = callbackStruct.signal.get_future();
    callbackStruct.bufferAddr = &openCLCppOutputBuffer;
    callbackStruct.queue = &queue;

    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompleteCallback(request, callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPICorrectness, reinterpret_cast<void*>(&callbackStruct)));
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // check is done in callback
    auto callbackReturnValue = unblockSignal.get();
    SPDLOG_INFO("Using callbacks!");
    OVMS_ServerDelete(cserver);
}

//static void callbackMarkingItWasUsedWith42AndUnblocking(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct);
static void callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPICorrectness(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct);
static void callbackUnblockingAndFreeingRequest(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct);

cl::CommandQueue* globalQueue = nullptr;

TEST(CAPIAsyncWithCallback, DummyCallback) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, INITIAL_VALUE);
    void* inputBufferData = in.data();
    std::vector<float> out(10, GARBAGE_VALUE);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    // start CAPI server
    // TODO load model with passed in context
    std::string port = "9000";
    randomizePort(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_gpu_dummy.json"));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::vector<float> data(DUMMY_MODEL_INPUT_SIZE, INITIAL_VALUE);
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    SPDLOG_DEBUG("openCLCppOutputBuffer:{}", (void*)openCLCppOutputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    // set callback
    CallbackUnblockingStruct callbackStruct;
    auto unblockSignal = callbackStruct.signal.get_future();
    callbackStruct.bufferAddr = &openCLCppOutputBuffer;
    callbackStruct.queue = &queue;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompleteCallback(request, callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPICorrectness, reinterpret_cast<void*>(&callbackStruct)));
    // infer
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request));
    // check
    auto callbackReturnValue = unblockSignal.get();

    queue.enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, inputByteSize, outputBufferData);
    const float* outputData = reinterpret_cast<const float*>(outputBufferData);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(in[i] + 1, outputData[i], FLOAT_TOLLERANCE) << "Different at:" << i << " place.";
    }
    ASSERT_EQ(42, callbackReturnValue);
    SPDLOG_INFO("Using callbacks!");
    // TODO cleanup settings
    OVMS_ServerDelete(cserver);
}

OVMS_Server* startCAPIServerFromConfig(const std::string configPath) {
    std::string port = "9000";
    randomizePort(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    EXPECT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    EXPECT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    EXPECT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    EXPECT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, configPath.c_str()));
    OVMS_Server* cserver = nullptr;
    EXPECT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    EXPECT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    OVMS_ServerSettingsDelete(serverSettings);
    OVMS_ModelsSettingsDelete(modelsSettings);
    return cserver;
}

class CAPIGPUPerfComparison : public TestWithTempDir {
protected:
    const uint afterConfigChangeLoadTimeMs = 50;
    const int stressIterationsLimit = 5000;

    std::string configFilePath;
    std::string ovmsConfig;
    std::string modelPath;
    OVMS_Server* cserver;

public:
    void SetUpConfig(const std::string& configContent, size_t elementsCount) {
        ovmsConfig = configContent;
        const std::string STRING_TO_REPLACE{"SECOND_DIM_TO_REPLACE"};
        auto it = ovmsConfig.find("SECOND_DIM_TO_REPLACE");
        if (it != std::string::npos) {
            ovmsConfig.replace(it, STRING_TO_REPLACE.size(), std::to_string(elementsCount));
        }
        configFilePath = directoryPath + "/ovms_config.json";
        SPDLOG_ERROR("ConfigConternt:{}", ovmsConfig);
        SPDLOG_ERROR("ConfigConternt:{}", configFilePath);
    }
};

static const char* dummyConfigContentWithReplacableShape = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "GPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 2,
                "shape": {"b": "(1,SECOND_DIM_TO_REPLACE) "}
            }
        }
    ]
}
)";

TEST_F(CAPIGPUPerfComparison, Dummy) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    const size_t elementsCount = 1'000'000;
    std::vector<float> in(elementsCount, INITIAL_VALUE);
    void* inputBufferData = in.data();
    std::vector<float> out(elementsCount, GARBAGE_VALUE);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    std::vector<cl::Buffer> openCLCppInputBuffer;
    openCLCppInputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
    openCLCppInputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
    std::vector<cl::Buffer> openCLCppOutputBuffer;
    openCLCppOutputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
    openCLCppOutputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
    queue.enqueueWriteBuffer(openCLCppInputBuffer[0], /*blocking*/ true, 0, inputByteSize, inputBufferData);
    queue.enqueueWriteBuffer(openCLCppInputBuffer[1], /*blocking*/ true, 0, inputByteSize, inputBufferData);
    // start CAPI server
    SetUpConfig(dummyConfigContentWithReplacableShape, elementsCount);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    OVMS_Server* cserver = startCAPIServerFromConfig(configFilePath);
    ASSERT_NE(nullptr, cserver);
    // prepare request
    std::vector<OVMS_InferenceRequest*> request(2, nullptr);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request[0], cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request[1], cserver, "dummy", 1));
    const std::vector<int64_t> modelShape{1, elementsCount};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request[0], DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, modelShape.data(), modelShape.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request[0], DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, modelShape.data(), modelShape.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request[1], DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, modelShape.data(), modelShape.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request[1], DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, modelShape.data(), modelShape.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request[0], DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer[0]), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));     // device id ?? TODO
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request[0], DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer[0]), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request[1], DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer[1]), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));     // device id ?? TODO
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request[1], DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer[1]), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    // set callback
    CallbackUnblockingStruct callbackStruct[2];
    std::vector<std::future<uint32_t>> unblockSignal;
    unblockSignal.emplace_back(callbackStruct[0].signal.get_future());
    unblockSignal.emplace_back(callbackStruct[1].signal.get_future());
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompleteCallback(request[0], callbackUnblockingAndFreeingRequest, reinterpret_cast<void*>(&callbackStruct[0])));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompleteCallback(request[1], callbackUnblockingAndFreeingRequest, reinterpret_cast<void*>(&callbackStruct[1])));
    std::unordered_map<int, double> times;
    size_t iterations = 10;
    iterations = 1'000;
    auto start = std::chrono::high_resolution_clock::now();
    /*    for(size_t i = 0; i < iterations; ++i) {
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request[i %2 ]));
        auto callbackReturnValue = unblockSignal.get();
        // we need to reset promise to be able to reuse signal
        callbackStruct.signal = std::promise<uint32_t>();
        unblockSignal = callbackStruct.signal.get_future();
    }*/
    size_t i = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request[i % 2]));
    for (i = 0; i < iterations; ++i) {
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request[(i + 1) % 2]));
        auto callbackReturnValue = unblockSignal[i % 2].get();
        // we need to reset promise to be able to reuse signal
        callbackStruct[i % 2].signal = std::promise<uint32_t>();
        unblockSignal[i % 2] = callbackStruct[i % 2].signal.get_future();
    }
    SPDLOG_ERROR("ER");
    auto callbackReturnValue = unblockSignal[iterations % 2].get();
    SPDLOG_ERROR("ER");
    auto stop = std::chrono::high_resolution_clock::now();
    times[1] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    // TODO check non-remote tensors async
    // TODO check sync non-remote tensors
    // TODO check sync remote tensors

    OVMS_ServerDelete(cserver);
    double fps = iterations / (times[1] / 1'000.);  //FPS[Frame/second]
    std::cout << "" << fps * elementsCount << " \t\t ";
}

TEST(OpenVINO, CallbacksTest) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    const std::string inputName{"b"};
    auto input = model->get_parameters().at(0);
    ov::element::Type_t dtype = ov::element::Type_t::f32;
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(10000);
    std::map<std::string, ov::PartialShape> inputShapes;
    inputShapes[inputName] = ovShape;
    model->reshape(inputShapes);
    auto cpuCompiledModel = core.compile_model(model, "CPU");
    auto cpuInferRequest = cpuCompiledModel.create_infer_request();
    // prepare ov::Tensor data
    std::vector<ov::Tensor> inputOvTensors, outputOvTensors;
    inputOvTensors.emplace_back(dtype, ovShape);
    outputOvTensors.emplace_back(dtype, ovShape);
    cpuInferRequest.set_tensor(inputName, inputOvTensors[0]);

    uint32_t callbackUsed = 31;
    OVMS_InferenceResponse* response{nullptr};
    cpuInferRequest.set_callback([&response, &callbackUsed](std::exception_ptr exception) {
        if (exception) {
            try {
                std::rethrow_exception(exception);
            } catch (const std::exception& e) {
                std::cout << "Caught exception: '" << e.what() << "'\n";
            } catch (...) {
                return;
            }
        }
        SPDLOG_INFO("Using OV callback");
        // here will go OVMS C-API serialization code
        callbackMarkingItWasUsedWith42(response, 1, reinterpret_cast<void*>(&callbackUsed));
    });
    bool callbackCalled{false};
    cpuInferRequest.start_async();
    EXPECT_FALSE(callbackCalled);
    cpuInferRequest.wait();
    ov::Tensor outOvTensor = cpuInferRequest.get_tensor("a");
    auto outAutoTensor = cpuInferRequest.get_tensor("a");
    EXPECT_TRUE(outOvTensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
    EXPECT_TRUE(outOvTensor.is<ov::Tensor>());
    EXPECT_TRUE(outAutoTensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
    EXPECT_TRUE(outAutoTensor.is<ov::Tensor>());
    // TODO check what happens if output tensor is not correct
}

class OpenVINO2 : public ::testing::Test {
protected:
    Core core;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::CompiledModel> compiledModel;
    std::shared_ptr<ov::InferRequest> inferRequest;
    std::shared_ptr<ov::intel_gpu::ocl::ClContext> gpu_context;
    std::shared_ptr<cl::CommandQueue> queue;
    cl_context ctxFromModel;
    uint32_t inputSecondDim = 100;
    void SetUp() {
        Core core;
        this->model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
        auto input = model->get_parameters().at(0);
        auto inputByteSize = ov::shape_size(input->get_shape());
        auto output = model->get_results().at(0);
        auto outputByteSize = ov::shape_size(output->get_shape());
        inputByteSize *= sizeof(float);
        outputByteSize *= sizeof(float);
        ov::AnyMap config = {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
            ov::auto_batch_timeout(0)};
        cl_platform_id platformId;
        cl_device_id deviceId;
        auto nonused = get_cl_context(platformId, deviceId);
        ov::Shape ovShape;
        ovShape.emplace_back(1);
        ovShape.emplace_back(inputSecondDim);
        std::map<std::string, ov::PartialShape> inputShapes;
        inputShapes[DUMMY_MODEL_INPUT_NAME] = ovShape;
        model->reshape(inputShapes);
        this->compiledModel = std::make_shared<ov::CompiledModel>(core.compile_model(model, "GPU", config));
        this->gpu_context = std::make_shared<ov::intel_gpu::ocl::ClContext>(compiledModel->get_context().as<ov::intel_gpu::ocl::ClContext>());
        this->ctxFromModel = gpu_context->get();
        this->inferRequest = std::make_shared<ov::InferRequest>(compiledModel->create_infer_request());
        cl::Context openCLCppContext(ctxFromModel);
        cl::Device device(deviceId);
        this->queue = std::make_shared<cl::CommandQueue>(openCLCppContext, device);
    }
    void TearDown() {}
};
TEST_F(OpenVINO2, UseCLContextForBuffersOVContextForInference) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, 42);
    void* inputBufferData = in.data();
    std::vector<float> out(10, 13.1);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    // here perform inf with OV
    ov::Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    using plugin_config_t = std::map<std::string, ov::Any>;
    plugin_config_t pluginConfig;
    pluginConfig["PERFORMANCE_HINT"] = "LATENCY";
    auto compiledModel = core.compile_model(model, "GPU", pluginConfig);
    auto request = compiledModel.create_infer_request();
    ov::element::Type_t type = ov::element::Type_t::f32;
    ov::Shape shape;
    shape.emplace_back(1);
    shape.emplace_back(10);
    // we need context from OV modelinstance.cpp
    std::unique_ptr<ov::intel_gpu::ocl::ClContext> ocl_context_cpp;
    cl_context ocl_context_c;
    {
        auto ocl_context = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
        ocl_context_cpp = std::make_unique<ov::intel_gpu::ocl::ClContext>(ocl_context);
        ocl_context_c = ocl_context_cpp->get();
    }
    SPDLOG_ERROR("{}", (void*)ocl_context_c);;
    // opencltensorfactory.hpp
    auto inputTensor = ocl_context_cpp->create_tensor(type, shape, openCLCppInputBuffer);
    auto outputTensor = ocl_context_cpp->create_tensor(type, shape, openCLCppOutputBuffer);
    request.set_tensor("b", inputTensor);
    request.set_tensor("a", outputTensor);
    request.start_async();
    request.wait();
    queue.enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, inputByteSize, outputBufferData);
    const float* outputData = reinterpret_cast<const float*>(outputBufferData);
    for (size_t i = 0; i < 10; ++i) {
        SPDLOG_ERROR("ER:{}", *(outputData + i));
    }
}

TEST_F(OpenVINO2, OutputTensorHasBiggerUnderlyingOCLBufferThanNeededPass) {
    bool retain = true;
    cl::Context openCLCppContext(ctxFromModel, retain);
    cl_int err;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize * 2, NULL, &err);
    auto inputOVOCLBufferTensor = gpu_context->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = gpu_context->create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    queue->enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    inferRequest->set_tensor(output, outputOVOCLBufferTensor);
    inferRequest->infer();
    std::vector<float> out(100, GARBAGE_VALUE);
    void* buffer_out = out.data();
    queue->enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, outputByteSize, buffer_out);
    const float* outputData = reinterpret_cast<const float*>(buffer_out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(in[i] + 1, outputData[i], FLOAT_TOLLERANCE) << "Different at:" << i << " place.";
    }
    // TODO separate test for below - extracting what kind of tensor in output it isa
    ov::Tensor outOvTensor = inferRequest->get_tensor(output);
    auto outAutoTensor = inferRequest->get_tensor(output);
    SPDLOG_ERROR("ov::Tensor type:{}", typeid(outOvTensor).name());
    SPDLOG_ERROR("auto type:{}", typeid(outAutoTensor).name());
    EXPECT_TRUE(outOvTensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
    EXPECT_TRUE(outOvTensor.is<ov::Tensor>());
    EXPECT_TRUE(outAutoTensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
    EXPECT_TRUE(outAutoTensor.is<ov::Tensor>());
}
TEST_F(OpenVINO2, OutputTensorHasBiggerShapeAndOCLBufferThanNeededThrowsOnSetTensor) {
    bool retain = true;
    cl::Context openCLCppContext(ctxFromModel, retain);
    cl_int err;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize * 2, NULL, &err);
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(inputSecondDim * 2);
    auto inputOVOCLBufferTensor = gpu_context->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = gpu_context->create_tensor(output->get_element_type(), ovShape, openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    queue->enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    EXPECT_THROW(inferRequest->set_tensor(output, outputOVOCLBufferTensor), ov::Exception);
}
TEST_F(OpenVINO2, OutputTensorHasSmallerUnderlyingOCLBufferThanNeededThrowsOnCreateRemoteTensor) {
    bool retain = true;
    cl::Context openCLCppContext(ctxFromModel, retain);
    cl_int err;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize / 2, NULL, &err);
    auto inputOVOCLBufferTensor = gpu_context->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    EXPECT_THROW(gpu_context->create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer), ov::Exception);
    // we will put data into input buffer
}
TEST_F(OpenVINO2, OutputTensorHasSmallerShapeAndUnderlyingOCLBufferThanNeededThrowsOnSetTensor) {
    bool retain = true;
    cl::Context openCLCppContext(ctxFromModel, retain);
    cl_int err;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    float divisionFactor = 2;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize / divisionFactor, NULL, &err);
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(inputSecondDim / divisionFactor);
    auto inputOVOCLBufferTensor = gpu_context->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = gpu_context->create_tensor(output->get_element_type(), ovShape, openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    queue->enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    EXPECT_THROW(inferRequest->set_tensor(output, outputOVOCLBufferTensor), ov::Exception);
}
TEST_F(OpenVINO2, OutputTensorHasSmallerShapeAndAppropriateOCLBufferThanNeededThrowsOnSetTensor) {
    bool retain = true;
    cl::Context openCLCppContext(ctxFromModel, retain);
    cl_int err;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    float divisionFactor = 2;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err);
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(inputSecondDim / divisionFactor);
    auto inputOVOCLBufferTensor = gpu_context->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = gpu_context->create_tensor(output->get_element_type(), ovShape, openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    queue->enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    EXPECT_THROW(inferRequest->set_tensor(output, outputOVOCLBufferTensor), ov::Exception);
}

static void callbackMarkingItWasUsedWith42(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    using ovms::StatusCode;
    SPDLOG_INFO("Using callback: callbackMarkingItWasUsedWith42!");
    uint32_t* usedFlag = reinterpret_cast<uint32_t*>(userStruct);
    *usedFlag = 42;
    OVMS_InferenceResponseDelete(response);
}
/*static void callbackMarkingItWasUsedWith42AndUnblocking(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    SPDLOG_INFO("Using callback: callbackMarkingItWasUsedWith42AndUnblocking!");
    std::cout << __LINE__ << "Calling set callback" << std::endl;
    CallbackUnblockingStruct* callbackUnblockingStruct = reinterpret_cast<CallbackUnblockingStruct*>(userStruct);
    callbackUnblockingStruct->signal.set_value(42);
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
}*/

static void callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPICorrectness(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    SPDLOG_INFO("Using callback: callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPICorrectness!");
    CallbackUnblockingStruct* callbackUnblockingStruct = reinterpret_cast<CallbackUnblockingStruct*>(userStruct);
    callbackUnblockingStruct->signal.set_value(42);
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);

    // verify GetOutput
    const void* voutputData{nullptr};
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_OPENCL);
    EXPECT_EQ(deviceId, 0);
    std::vector<int> expectedShape{1, 10};
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(expectedShape[i], shape[i]) << "Different at:" << i << " place.";
    }
    SPDLOG_INFO("Callback buffer addr:{}", (void*)voutputData);  // DEBUG does not work in callback
    EXPECT_EQ(callbackUnblockingStruct->bufferAddr, voutputData);
    const cl::Buffer* openCLCppOutputBuffer = reinterpret_cast<const cl::Buffer*>(voutputData);
    std::vector<float> out(10, GARBAGE_VALUE);
    void* buffer_out = out.data();
    SPDLOG_INFO("Queue address in callback:{}", (void*)callbackUnblockingStruct->queue);  // DEBUG does not work in callback
    callbackUnblockingStruct->queue->enqueueReadBuffer(*openCLCppOutputBuffer, /*blocking*/ true, 0, expectedShape[1] * sizeof(float), buffer_out);
    std::vector<float> expectedData(expectedShape[1], INITIAL_VALUE + 1);

    const float* outputData = reinterpret_cast<const float*>(buffer_out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expectedData[i], outputData[i], FLOAT_TOLLERANCE) << "Different at:" << i << " place.";
    }
    OVMS_InferenceResponseDelete(response);
}
static void callbackUnblockingAndFreeingRequest(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    SPDLOG_INFO("Using callback: callbackUnblockingAndFreeingRequest!");
    CallbackUnblockingStruct* callbackUnblockingStruct = reinterpret_cast<CallbackUnblockingStruct*>(userStruct);
    callbackUnblockingStruct->signal.set_value(42);
    OVMS_InferenceResponseDelete(response);
}

template <typename Key, typename Value>
class FilteredMap {
    using map_t = std::unordered_map<Key, Value>;
    using keySet_t = std::set<Key>;
    using value_t = typename map_t::value_type;
    const map_t& originalMap;
    const keySet_t& allowedKeys;

public:
    FilteredMap(const map_t& originalMap, const keySet_t& allowedKeys) :
        originalMap(originalMap),
        allowedKeys(allowedKeys) {}
    class Iterator {
    public:
        using reference_t = value_t&;
        using constReference_t = const value_t&;
        using pointer_t = value_t*;
        using constPointer_t = const value_t*;
        Iterator(typename map_t::const_iterator it, typename map_t::const_iterator end, const keySet_t& allowedKeys) :
            it(it),
            end(end),
            allowedKeys(allowedKeys) {
            ensureIsValidIterator();
        }
        constReference_t operator*() const {
            return *it;
        }
        constPointer_t operator->() {
            return &(*it);
        }
        constPointer_t operator->() const {
            return &(*it);
        }
        // preincrement
        Iterator& operator++() {
            if (it != end) {
                ++(this->it);
                ensureIsValidIterator();
            }
            return *this;
        }
        Iterator operator++(int) {
            Iterator tmp{*this};
            ++(*this);
            return tmp;
        }
        friend bool operator==(const Iterator& a, const Iterator& b) {
            return a.it == b.it;
        }
        friend bool operator!=(const Iterator& a, const Iterator& b) {
            return a.it != b.it;
        }

    private:
        typename map_t::const_iterator it;
        typename map_t::const_iterator end;
        const keySet_t allowedKeys;
        void ensureIsValidIterator() {
            while ((it != end) && (allowedKeys.find(it->first) == allowedKeys.end())) {
                ++it;
            }
        }
    };
    Iterator begin() const {
        return Iterator(originalMap.begin(), originalMap.end(), allowedKeys);
    }
    Iterator end() const {
        return Iterator(originalMap.end(), originalMap.end(), allowedKeys);
    }
    const Value& at(const Key& k) const {
        if (allowedKeys.find(k) == allowedKeys.end()) {
            throw std::out_of_range("Key not found in FilteredMap");
        }
        return originalMap.at(k);
    }
    Iterator find(const Key& k) const {
        if (allowedKeys.find(k) == allowedKeys.end()) {
            return end();
        }
        return Iterator(originalMap.find(k), originalMap.end(), allowedKeys);
    }
};

#define TEST_FILTER(ORIGINAL, FILTER)                                                            \
    {                                                                                            \
        FilteredMap FILTERED_MAP(ORIGINAL, FILTER);                                              \
        for (const auto& [k, v] : ORIGINAL) {                                                    \
            if (FILTER.find(k) != FILTER.end()) {                                                \
                EXPECT_EQ(FILTERED_MAP.at(k), ORIGINAL[k]) << "k:" << k << ", v:" << v;          \
            } else {                                                                             \
                EXPECT_EQ(FILTERED_MAP.find(k), FILTERED_MAP.end()) << "k:" << k << ", v:" << v; \
            }                                                                                    \
        }                                                                                        \
        for (auto [k, v] : FILTERED_MAP) {                                                       \
            EXPECT_NE(FILTER.find(k), FILTER.end()) << "k:" << k << ", v:" << v;                 \
            EXPECT_EQ(FILTERED_MAP.at(k), ORIGINAL.at(k)) << "k:" << k << ", v:" << v;           \
        }                                                                                        \
    }

TEST(FilteredMapTest, MapIntInt) {
    std::unordered_map<int, int> original{{1, 1}, {2, 2}, {3, 3}};
    std::set<int> filterEmpty{};
    std::set<int> filter1{1};
    std::set<int> filter2{2};
    std::set<int> filter3{3};
    std::set<int> filter12{1, 2};
    std::set<int> filter13{1, 3};
    std::set<int> filter23{1, 3};
    std::set<int> filter123{1, 2, 3};
    TEST_FILTER(original, filterEmpty);
    TEST_FILTER(original, filter1);
    TEST_FILTER(original, filter2);
    TEST_FILTER(original, filter3);
    TEST_FILTER(original, filter12);
    TEST_FILTER(original, filter13);
    TEST_FILTER(original, filter23);
    TEST_FILTER(original, filter123);
}
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
