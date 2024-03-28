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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>

#include "c_api_test_utils.hpp"
#include "test_utils.hpp"

using namespace ov;

using testing::HasSubstr;
using testing::Not;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
//#include "openvino/runtime/intel_gpu/properties.hpp"
//#include "openvino/runtime/remote_tensor.hpp"

cl_context get_cl_context(cl_platform_id& platformId, cl_device_id& deviceId) {
    cl_int err;

    // Step 1: Querying Platforms
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of platforms\n";
        throw 1;
    }
    clGetPlatformIDs(1, &platformId, nullptr);
    // Step 2: Querying Devices
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of devices\n";
        throw 1;
    }
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting GPU deviceId\n";
        throw 1;
    }
    // Step 3: Creating a Context
    cl_context openCLCContext = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
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
    ///
    cl_platform_id platformId;
    cl_device_id deviceId;
    auto nonused = get_cl_context(platformId, deviceId);
    auto compiledModel = core.compile_model(model, "GPU", config);
    auto gpu_context = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
    //    auto inferRequest = compiledModel.create_infer_request();
    //  auto some_context = compiledModel.get_context();
    // auto gpu_context = some_context.as<ov::intel_gpu::ocl::ClContext>;
    cl_context ctxFromModel = gpu_context.get();
    // TODO should all models share the same context?
    // gpu_context.create_tensor() -> when it is deleted?
    cl::Context openCLCppContext(ctxFromModel);  // Convert cl_context to cl::Context
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
    SPDLOG_ERROR("ER");
    auto queue = cl::CommandQueue(openCLCppContext, device);  //, oclQueueProperties);
    SPDLOG_ERROR("ER");
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    SPDLOG_ERROR("ER");
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
    cl::Context openCLCppContext(openCLCContext);  // Convert cl_context to cl::Context
    cl_int err;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err);
    // create tensors and perform inference
    // wrap in and out buffers into RemoteTensor and set them to infer request
    auto inputOVOCLBufferTensor = remote_context.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = remote_context.create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    SPDLOG_ERROR("ER");
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

TEST(OpenVINO, SetTensorTest) {
    size_t tSize = 10;
    int iterations = 10000;
    std::vector<size_t> sizeSet{10, 10 * 10, 10 * 100, 10 * 1000, 10 * 10000, 10 * 100000, 10 * 1000000};
    // load model
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto outputByteSize = ov::shape_size(output->get_shape());
    // we need byte size not no of elements
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);  // Convert cl_context to cl::Context
    cl::Device device(deviceId);
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
    // create
    // reset input/output tensors
    // copy output if needed
    enum {
        CPU_COPY,
        CPU_SET,
        GPU_COPY,
        GPU_OCL_COPY,
        GPU_SET,
        GPU_SET_OVTEN_OCL,
        GPU_SET_OCLTEN_OCL
    };
    std::unordered_map<int, std::unordered_map<int, double>> times;
    for (auto tSize : sizeSet) {
        SPDLOG_INFO("Performing tests for dummy shape (1,{}) ....", tSize);
        auto sizeStart = std::chrono::high_resolution_clock::now();
        ov::element::Type_t dtype = ov::element::Type_t::f32;
        ov::Shape ovShape;
        ovShape.emplace_back(1);
        ovShape.emplace_back(tSize);
        std::map<std::string, ov::PartialShape> inputShapes;
        inputShapes["b"] = ovShape;
        model->reshape(inputShapes);
        auto oclCompiledModel = core.compile_model(model, remote_context);
        auto oclInferRequest = oclCompiledModel.create_infer_request();
        auto gpuCompiledModel = core.compile_model(model, "GPU");
        auto gpuInferRequest = gpuCompiledModel.create_infer_request();
        auto cpuCompiledModel = core.compile_model(model, "CPU");
        auto cpuInferRequest = cpuCompiledModel.create_infer_request();
        // prepare data
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
        std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
        inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
        inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
        outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
        outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
        std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
        inputs.emplace_back(remote_context.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
        inputs.emplace_back(remote_context.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
        outputs.emplace_back(remote_context.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
        outputs.emplace_back(remote_context.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
        std::vector<float> outputData(tSize, 0);
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < iterations; ++i) {
            oclInferRequest.set_tensor(input, inputOvTensors[i % 2]);
            oclInferRequest.infer();
            const auto& outTensor = oclInferRequest.get_tensor(output);
            std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        times[GPU_OCL_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        start = std::chrono::high_resolution_clock::now();
        //GPU_SET_OCLTEN_OCL
        for (auto i = 0; i < iterations; ++i) {
            oclInferRequest.set_tensor(input, inputs[i % 2]);
            oclInferRequest.set_tensor(output, outputs[i % 2]);
            oclInferRequest.infer();
        }
        stop = std::chrono::high_resolution_clock::now();
        times[GPU_SET_OCLTEN_OCL][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        //GPU_SET_OVTEN_OCL
        start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < iterations; ++i) {
            oclInferRequest.set_tensor(input, inputOvTensors[i % 2]);
            oclInferRequest.set_tensor(output, outputOvTensors[i % 2]);
            oclInferRequest.infer();
        }
        stop = std::chrono::high_resolution_clock::now();
        times[GPU_SET_OVTEN_OCL][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        // GPU_COPY copy from output
        start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < iterations; ++i) {
            gpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
            gpuInferRequest.infer();
            const auto& outTensor = oclInferRequest.get_tensor(output);
            std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
        }
        stop = std::chrono::high_resolution_clock::now();
        times[GPU_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        // GPU set input & output
        start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < iterations; ++i) {
            gpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
            gpuInferRequest.set_tensor(output, outputOvTensors[i % 2]);
            gpuInferRequest.infer();
        }
        stop = std::chrono::high_resolution_clock::now();
        times[GPU_SET][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        // CPU_COPY COPY
        start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < iterations; ++i) {
            cpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
            cpuInferRequest.infer();
            const auto& outTensor = cpuInferRequest.get_tensor(output);
            std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
        }
        stop = std::chrono::high_resolution_clock::now();
        times[CPU_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < iterations; ++i) {
            cpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
            cpuInferRequest.set_tensor(output, outputOvTensors[i % 2]);
            cpuInferRequest.infer();
        }
        stop = std::chrono::high_resolution_clock::now();
        times[CPU_SET][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        auto sizeStop = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("Testing for size: {}, took {} seconds. Next step will take probably x10 longer ...", tSize, std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000000.0;
    }
    std::cout << std::right;
    for (auto s : {"CPU_COPY", "CPU_SET", "GPU_COPY", "GPU_OCL_COPY", "GPU_SET", "GPU_SET_OVTEN_OCL", "GPU_SET_OCL_OCL"}) {
        std::cout << s << "[FPS]"
                  << "\t\t" << s << "[MePS]"
                  << "\t\t";
    }
    std::cout << std::endl;
    //std::cout << "CPU_COPY\t\t\t\t\t CPU_SET\t\t\t\t\t GPU_COPY\t\t\t\t\tGPU_OCL_COPY\t\t\t\t\tGPU_SET\t\t\t\tGPU_SET_OVTEN_OCL\t\t\t\t\t GPU_SET_OCL_OCL" << std::endl;
    for (auto s : sizeSet) {
        for (auto t : {CPU_COPY, CPU_SET, GPU_COPY, GPU_OCL_COPY, GPU_SET, GPU_SET_OVTEN_OCL, GPU_SET_OCLTEN_OCL}) {
            double fps = iterations / (times[t][s] / 1000.);
            std::cout << "" << fps << " \t ";
            std::cout << "" << fps * s << " \t\t ";
        }
        std::cout << std::endl;
    }
}

TEST(CAPINonCopy, Flow) {
    // create openCL context
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);  // Convert cl_context to cl::Context
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
    // TODO make sure config points to GPU
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_standard_dummy.json"));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    auto bareOpenCLCMemory = openCLCppInputBuffer.get();
    //ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(bareOpenCLCMemory), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1)); // device id ?? TODO
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    // verify response
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
    // TODO
    // infer
    // TODO need to pass context into MP ??
}
#pragma GCC diagnostic pop
