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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>

#include "../ocl_utils.hpp"
#include "../ov_utils.hpp"
#include "../ovms.h"           // NOLINT
#include "../ovms_internal.h"  // NOLINT
#include "../status.hpp"
#include "c_api_test_utils.hpp"
#include "gpuenvironment.hpp"
#include "test_utils.hpp"

using namespace ov;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <va/va_drm.h>

typedef void* VADisplay;

cl_context get_cl_context(cl_platform_id& platformId, cl_device_id& deviceId) {
    cl_int clError;
    cl_uint numPlatforms = 0;
    clError = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (clError != CL_SUCCESS) {
        std::cerr << "Error getting number of platforms\n";
        throw 1;
    }
    // extract 1st platform from numPlatforms
    clGetPlatformIDs(1, &platformId, nullptr);
    cl_uint numDevices = 0;
    // query how many devices there are
    clError = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (clError != CL_SUCCESS) {
        std::cerr << "Error getting number of devices\n";
        throw 1;
    }
    if (0 == numDevices) {
        std::cerr << "There is no available devices\n";
        throw 1;
    }
    cl_uint numberOfDevicesInContext = 1;
    clError = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numberOfDevicesInContext, &deviceId, nullptr);
    if (clError != CL_SUCCESS) {
        std::cerr << "Error getting GPU deviceId\n";
        throw 1;
    }
    // since we only use 1 device we can use address of deviceId
    cl_context openCLCContext = clCreateContext(nullptr, numberOfDevicesInContext, &deviceId, nullptr, nullptr, &clError);
    if (clError != CL_SUCCESS) {
        std::cerr << "Error creating context\n";
        throw 1;
    }
    return openCLCContext;
}

const float INITIAL_VALUE{0.13666};
const float GARBAGE_VALUE = 42.66613;
const float FLOAT_TOLERANCE{0.001};

constexpr bool queueReadWriteBlockingTrue = true;
constexpr bool retainCLContextOwnership = true;

static void checkDummyOpenCLResponse(OVMS_InferenceResponse* response, cl::CommandQueue& queue, double expectedValue, double tolerance) {
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    const void* voutputData{nullptr};
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t ovmsDeviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &ovmsDeviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_OPENCL);
    EXPECT_EQ(ovmsDeviceId, 0);
    std::vector<int> expectedShape{1, 10};
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(expectedShape[i], shape[i]) << "Different at:" << i << " place.";
    }

    const cl::Buffer* openCLCppOutputBuffer = reinterpret_cast<const cl::Buffer*>(voutputData);
    std::vector<float> out(10, GARBAGE_VALUE);
    void* bufferOut = out.data();
    auto clError = queue.enqueueReadBuffer(*openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, expectedShape[1] * sizeof(float), bufferOut);
    EXPECT_EQ(0, clError);
    const float* outputData = reinterpret_cast<const float*>(bufferOut);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expectedValue, outputData[i], FLOAT_TOLERANCE) << "Different at:" << i << " place.";
    }
}

class OpenVINOGPU : public ::testing::Test {
public:
    void SetUp() override {
        GPUEnvironment::skipWithoutGPU();
    }
};

TEST_F(OpenVINOGPU, ExtractContextFromModel) {
    // TODO split
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
    auto ovGpuOclContext = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctxFromModel = ovGpuOclContext.get();
    cl::Context openCLCppContext(ctxFromModel, retainCLContextOwnership);
    cl_int clError;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(NULL, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError);
    EXPECT_EQ(NULL, clError);
    auto inputOVOCLBufferTensor = ovGpuOclContext.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = ovGpuOclContext.create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(10, 0.1);
    void* inputBufferData = in.data();
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    cl::Device device(deviceId);
    auto queue = cl::CommandQueue(openCLCppContext, device);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.set_tensor(input, inputOVOCLBufferTensor);
    inferRequest.set_tensor(output, outputOVOCLBufferTensor);
    inferRequest.infer();
    std::vector<float> out(10);
    void* bufferOut = out.data();
    EXPECT_EQ(0, queue.enqueueReadBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, outputByteSize, bufferOut));
    for (size_t i = 0; i < inputByteSize / sizeof(float); ++i) {
        // different precision on GPU vs CPU
        EXPECT_NEAR(in[i] + 1, out[i], 0.0004) << "i:" << i;
    }
}

class OpenVINOGPUContextFromModel : public OpenVINOGPU {
protected:
    Core core;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::CompiledModel> compiledModel;
    std::shared_ptr<ov::InferRequest> inferRequest;
    std::shared_ptr<ov::intel_gpu::ocl::ClContext> ovGpuOclContext;
    std::shared_ptr<cl::Context> oclCppContextFromModel;
    std::shared_ptr<cl::CommandQueue> queueFromModelContext;
    cl_context ctxFromModel;
    uint32_t inputSecondDim = 100;
    void SetUp() {
        OpenVINOGPU::SetUp();
        SKIP_AND_EXIT_IF_NO_GPU();
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
        this->ovGpuOclContext = std::make_shared<ov::intel_gpu::ocl::ClContext>(compiledModel->get_context().as<ov::intel_gpu::ocl::ClContext>());
        this->ctxFromModel = ovGpuOclContext->get();
        this->inferRequest = std::make_shared<ov::InferRequest>(compiledModel->create_infer_request());
        this->oclCppContextFromModel = std::make_shared<cl::Context>(this->ctxFromModel, retainCLContextOwnership);
        cl::Device device(deviceId);
        this->queueFromModelContext = std::make_shared<cl::CommandQueue>(*this->oclCppContextFromModel, device);
    }
    void TearDown() {}
};
TEST_F(OpenVINOGPU, LoadModelWithPrecreatedContext) {
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
    auto remoteContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
    auto compiledModel = core.compile_model(model, remoteContext);
    // now we create buffers
    cl::Context openCLCppContext(openCLCContext);
    cl_int clError;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    // create tensors and perform inference
    // wrap in and out buffers into RemoteTensor and set them to infer request
    auto inputOVOCLBufferTensor = remoteContext.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = remoteContext.create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(10, 0.1);
    void* inputBufferData = in.data();
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.set_tensor(input, inputOVOCLBufferTensor);
    inferRequest.set_tensor(output, outputOVOCLBufferTensor);
    inferRequest.infer();
    std::vector<float> out(10);
    void* bufferOut = out.data();
    EXPECT_EQ(0, queue.enqueueReadBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, outputByteSize, bufferOut));
    for (size_t i = 0; i < inputByteSize / sizeof(float); ++i) {
        // different precision on GPU vs CPU
        EXPECT_NEAR(in[i] + 1, out[i], 0.0004) << "i:" << i;
    }
}

struct CallbackUnblockingStructWithQueue {
    std::promise<uint32_t> signal;
    void* bufferAddr = nullptr;
    cl::CommandQueue* queue = nullptr;
};
struct CallbackUnblockingCPUStruct {
    std::promise<void> signal;
    OVMS_InferenceResponse* response{nullptr};
};
class CAPINonCopy : public ::testing::Test {
public:
    void SetUp() override {
        GPUEnvironment::skipWithoutGPU();
    }
};
static void callbackUnblocking(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct);

#ifdef BUILD_VAAPITESTS
// https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API.html#direct-nv12-video-surface-input
class VAHelper {
    int drmFiledescriptor = 0;
    VADisplay vaDisplay = nullptr;

public:
    VADisplay getVADisplay() { return vaDisplay; }
    VAHelper(const std::string deviceFilepath = "/dev/dri/renderD128") {
        drmFiledescriptor = open(deviceFilepath.c_str(), O_RDWR);
        if (drmFiledescriptor < 0) {
            EXPECT_TRUE(false) << "failed to open DRM device:" << deviceFilepath;
            throw std::runtime_error("failed to open DRM device");
        }
        vaDisplay = vaGetDisplayDRM(drmFiledescriptor);
        if (vaDisplay == nullptr) {
            close(drmFiledescriptor);
            EXPECT_TRUE(false) << "failed to get VA DRM display";
            throw std::runtime_error("failed to get VA DRM display");
            return;
        }
        int majorVersion, minorVersion;
        VAStatus status = vaInitialize(vaDisplay, &majorVersion, &minorVersion);
        if (status != VA_STATUS_SUCCESS) {
            vaTerminate(vaDisplay);
            close(drmFiledescriptor);
            EXPECT_TRUE(false) << "Failed to initialize VA API with error:" << status;
            throw std::runtime_error("Failed to initialize VA API");
        }
        this->vaDisplay = vaDisplay;
        SPDLOG_TRACE("Initialized VADisplay: {}, with DRM device: {}, version:  {}.{}", vaDisplay, drmFiledescriptor, majorVersion, minorVersion);
    }
    ~VAHelper() {
        if (vaDisplay) {
            SPDLOG_TRACE("Terminating vaDisplay:{}", vaDisplay);
            vaTerminate(vaDisplay);
        }
        if (drmFiledescriptor) {
            SPDLOG_TRACE("Closing  drmFiledescriptor:{}", drmFiledescriptor);
            close(drmFiledescriptor);
        }
    }
    VAHelper(const VAHelper&) = delete;
    VAHelper& operator=(const VAHelper&) = delete;
};
#endif

const std::shared_ptr<ov::Model> preprocessModel(const std::shared_ptr<ov::Model>& model) {
    ov::preprocess::PrePostProcessor preprocessor(model);
    preprocessor.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
        .set_memory_type(ov::intel_gpu::memory_type::surface);
    preprocessor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    preprocessor.input().model().set_layout("NCHW");
    return preprocessor.build();
}

const std::string FACE_DETECTION_ADAS_MODEL_CONFIG_JSON{"/ovms/src/test/configs/config_gpu_face_detection_adas.json"};
const std::string FACE_DETECTION_ADAS_MODEL_PATH{"/ovms/src/test/face_detection_adas/1/face-detection-adas-0001.xml"};
const std::string FACE_DETECTION_ADAS_MODEL_NAME{"face_detection_adas"};
const std::string FACE_DETECTION_ADAS_INPUT_NAME{"data"};
const std::string FACE_DETECTION_ADAS_OUTPUT_NAME{"detection_out"};
const std::vector<int64_t> FACE_DETECTION_ADAS_INPUT_SHAPE{1, 3, 384, 672};

TEST_F(OpenVINOGPU, LoadModelWithVAContextInferenceFaceDetectionAdasWithPreprocTest) {
#ifndef BUILD_VAAPITESTS
    GTEST_SKIP() << "Test not enabled on UBI images";
#else
    ov::element::Type_t dtype = ov::element::Type_t::f32;
    Core core;
    auto model = core.read_model(FACE_DETECTION_ADAS_MODEL_PATH);

    std::string outputName{"detection_out"};
    model = preprocessModel(model);
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        SPDLOG_INFO("input name: {}", input.get_any_name());
        SPDLOG_INFO("shape: {}", ovms::Shape(input.get_partial_shape()).toString());
    }
    for (const ov::Output<ov::Node>& output : model->outputs()) {
        SPDLOG_INFO("output name: {}", output.get_any_name());
        SPDLOG_INFO("shape: {}", ovms::Shape(output.get_partial_shape()).toString());
    }
    VAHelper vaHelper;
    ASSERT_NE(vaHelper.getVADisplay(), nullptr);
    ov::intel_gpu::ocl::VAContext vaGpuContext(core, vaHelper.getVADisplay());
    // long unsigned int width = FACE_DETECTION_ADAS_INPUT_SHAPE[2];
    uint32_t width = FACE_DETECTION_ADAS_INPUT_SHAPE[2];
    uint32_t height = FACE_DETECTION_ADAS_INPUT_SHAPE[3];
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeFloat;  // VAGenericValueTypeInteger; // TODO does it work with both
    surface_attrib.value.value.i = VA_FOURCC_NV12;        // Specify the desired pixel format TODO

    // Create the VA surface
    VASurfaceID vaSurface;
    auto status = vaCreateSurfaces(vaHelper.getVADisplay(), VA_RT_FORMAT_YUV420, width, height, &vaSurface, 1, &surface_attrib, 1);
    ASSERT_EQ(VA_STATUS_SUCCESS, status) << "vaCreateSurfaces failed: " << status;
    // this would not work since OV is not ale to create VADisplay
    // auto gpuCompiledModel = core.compile_model(model, "GPU");
    auto gpuCompiledModel = core.compile_model(model, vaGpuContext);
    auto ovWrappedVAContext = gpuCompiledModel.get_context().as<ov::intel_gpu::ocl::VAContext>();
    auto gpuInferRequest = gpuCompiledModel.create_infer_request();
    // alternatively we could use create_tensor_nv12 but that would require deserialization of two inputs at once
    // in OVMS which is not how it is implemented
    // auto remoteTensor = ovWrappedVAContext.create_tensor_nv12(width, height, vaSurface);
    AnyMap tensorParams = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
        {ov::intel_gpu::dev_object_handle.name(), vaSurface},
        {ov::intel_gpu::va_plane.name(), uint32_t(0)}};
    ov::Tensor firstTensor = ovWrappedVAContext.create_tensor(element::u8, {1, width, height, 1}, tensorParams);
    tensorParams[ov::intel_gpu::va_plane.name()] = uint32_t(1);
    ov::Tensor secondTensor = ovWrappedVAContext.create_tensor(element::u8, {1, width / 2, height / 2, 2}, tensorParams);
    gpuInferRequest.set_tensor(FACE_DETECTION_ADAS_INPUT_NAME + "/y", firstTensor);
    gpuInferRequest.set_tensor(FACE_DETECTION_ADAS_INPUT_NAME + "/uv", secondTensor);
    gpuInferRequest.infer();
    auto outputTensor = gpuInferRequest.get_tensor(FACE_DETECTION_ADAS_OUTPUT_NAME);
    auto data = outputTensor.data();
    auto shape = outputTensor.get_shape();
    for (auto d : shape) {
        SPDLOG_ERROR("Dim:{}", d);
    }
    float* val = (float*)data;
    SPDLOG_ERROR("Dumping output data");
    for (int i = 0; i < 10; i++) {
        std::string row;
        row += std::to_string(i);
        row += " [";
        for (int j = 0; j < 7; j++) {
            row += std::to_string(*(val + i * 7 + j));
            row += ",";
        }
        row += "]";
        SPDLOG_ERROR(row);
    }
#endif
}
TEST_F(OpenVINOGPU, LoadModelWithVAContextInferenceFaceDetectionAdasNoPreprocTest) {
    GTEST_SKIP() << "It seems there is no way to use VAAPI without preprocessing";
}

TEST_F(CAPINonCopy, VAContextGlobalPreprocHardcodedInput) {  // TODO rename
#ifndef BUILD_VAAPITESTS
    GTEST_SKIP() << "Test not enabled on UBI images";
#else
    std::string port = "9000";
    randomizeAndEnsureFree(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, FACE_DETECTION_ADAS_MODEL_CONFIG_JSON.c_str()));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    // TODO requires model mgmt otherwise
    // we need to set up global VA Context before we start the server
    VAHelper vaHelper;
    ASSERT_NE(vaHelper.getVADisplay(), nullptr);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSetGlobalVADisplay(cserver, vaHelper.getVADisplay()));  // TODO reset always on exit
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, FACE_DETECTION_ADAS_MODEL_NAME.c_str(), 1));
    const std::string inputName_y = FACE_DETECTION_ADAS_INPUT_NAME + "/y";
    const std::string inputName_uv = FACE_DETECTION_ADAS_INPUT_NAME + "/uv";
    // prepare input
    int width = 384;  // FP32
    int height = 672;
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeFloat;  // VAGenericValueTypeInteger;
    surface_attrib.value.value.i = VA_FOURCC_NV12;        // Specify the desired pixel format

    // Create the VA surface
    VASurfaceID vaSurface;
    SPDLOG_ERROR("ZZZ vaSurface: {}", cl_uint(vaSurface));
    auto status = vaCreateSurfaces(vaHelper.getVADisplay(), VA_RT_FORMAT_YUV420, width, height, &vaSurface, 1, &surface_attrib, 1);
    SPDLOG_ERROR("ZZZ vaSurface: {}", cl_uint(vaSurface));
    ASSERT_EQ(VA_STATUS_SUCCESS, status) << "vaCreateSurfaces failed: " << status;
    const std::vector<int64_t> inputShape{1, 3, 384, 672};
    const std::vector<int64_t> inputShape_y{1, 384, 672, 1};
    const std::vector<int64_t> inputShape_uv{1, 384 / 2, 672 / 2, 2};
    constexpr size_t inputBytesize = 1 * 3 * 384 * 672;
    constexpr size_t inputBytesize_y = 1 * 1 * 384 * 672;
    constexpr size_t inputBytesize_uv = 1 * 2 * 384 / 2 * 672 / 2;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, inputName_y.c_str(), OVMS_DATATYPE_U8, inputShape_y.data(), inputShape_y.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, inputName_y.c_str(), reinterpret_cast<void*>(vaSurface), inputBytesize_y * sizeof(uint8_t), OVMS_BUFFERTYPE_VASURFACE_Y, 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, inputName_uv.c_str(), OVMS_DATATYPE_U8, inputShape_uv.data(), inputShape_uv.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, inputName_uv.c_str(), reinterpret_cast<void*>(vaSurface), inputBytesize_uv * sizeof(uint8_t), OVMS_BUFFERTYPE_VASURFACE_UV, 1));
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
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
    SPDLOG_ERROR("Name: {}, bytesize:{}", outputName, bytesize);
    float* val = (float*)voutputData;
    SPDLOG_ERROR("Dumping output data");
    for (int i = 0; i < 10; i++) {
        std::string row;
        row += std::to_string(i);
        row += " [";
        for (int j = 0; j < 7; j++) {
            row += std::to_string(*(val + i * 7 + j));
            row += ",";
        }
        row += "]";
        SPDLOG_ERROR(row);
    }
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSetGlobalVADisplay(cserver, 0));  // TODO reset always on exit
    OVMS_ServerDelete(cserver);
#endif
}

TEST_F(OpenVINOGPU, SetTensorTest) {
    size_t tSize = 10;
    int iterations = 10;
    iterations = 1'000;
    // std::vector<size_t> sizeSet{10, 10 * 10, 10 * 100, 10 * 1'000, 10 * 10'000, 10 * 100'000, 10 * 1'000'000};
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

        cl_int clError;
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextDifferent, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextDifferent, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
        {  // GPU_OCL_SET_OV model loaded on gpu with both output & input being ov::Tensor
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
            cl::Context openCLCppContextFromModel(openCLCContextFromModel, retainCLContextOwnership);
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            CallbackUnblockingStruct callbackStruct;
            auto unblockSignal = callbackStruct.signal.get_future();
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                // SPDLOG_INFO("iter start");
                // std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
                // inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[i % 2]));
                // outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[i % 2]));
                ov::Tensor inputOVTensor = inputs[i % 2];
                ov::Tensor outputOVTensor = outputs[i % 2];
                // gpuInferRequest.set_tensor(input,inputOVTensor);
                // gpuInferRequest.set_tensor(output, outputOVTensor);
                gpuInferRequest.set_tensor(input, inputs[i % 2]);
                gpuInferRequest.set_tensor(output, outputs[i % 2]);
                gpuInferRequest.set_callback([&gpuInferRequest, &callbackStruct](std::exception_ptr exception) {
                    // SPDLOG_INFO("entered callback");
                    gpuInferRequest.set_callback([](std::exception_ptr exception) {});
                    // SPDLOG_INFO("callback start");
                    callbackStruct.signal.set_value(42);
                    // SPDLOG_INFO("callback end");
                });
                // SPDLOG_INFO("callback end");
                gpuInferRequest.start_async();
                // SPDLOG_INFO("waiting to unblock");
                unblockSignal.get();
                // SPDLOG_INFO("Unblocked thread");
                callbackStruct.signal = std::promise<uint32_t>();
                // SPDLOG_INFO("reset promise");
                unblockSignal = callbackStruct.signal.get_future();
                // SPDLOG_INFO("reset future");
                // gpuInferRequest.wait(); // TODO probably not required
                // SPDLOG_INFO("iter end");
            }
            auto stop = std::chrono::high_resolution_clock::now();
            SPDLOG_ERROR("Log plugin");
            ovms::logOVPluginConfig([&gpuCompiledModel](const std::string& key) { return gpuCompiledModel.get_property(key); }, " {someAuthor} ", " {some details} ");
            SPDLOG_ERROR("Log plugin end");
            times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;  // ms
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS:{}", times[GPU_OV_SET_OCL_BUFF_DIFF_TENS_SAME_FULL_OVMS][tSize]);
        }
#ifdef TEST_VAAPI
        // TODO
        // * no get_va_display function
        // * no allocate_image
        {  // GPU_OV_SET_VAA_BUF model loaded with ov context and vaapi tensors used
            VADisplay vaDisplay = createVADisplay();
            VAHelper vaHelper;
            ASSERT_NE(vaHelper.getVADisplay(), nullptr);
            ov::intel_gpu::ocl::VAContext vaGpuContext(core, vaHelper.getVADisplay());
            int width = 200;
            int height = 200;
            VASurfaceAttrib surface_attrib;
            surface_attrib.type = VASurfaceAttribPixelFormat;
            surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
            surface_attrib.value.type = VAGenericValueTypeFloat;  // VAGenericValueTypeInteger;
            surface_attrib.value.value.i = VA_FOURCC_NV12;        // Specify the desired pixel format

            // Create the VA surface
            VASurfaceID vaSurface;
            status = vaCreateSurfaces(vaHelper.getVADisplay(), VA_RT_FORMAT_YUV420, width, height, &vaSurface, 1, &surface_attrib, 1);
            ASSERT_EQ(VA_STATUS_SUCCESS, status) << "vaCreateSurfaces failed: " << status;
            auto remoteTensor = vaGpuContext.create_tensor_nv12(width, height, vaSurface);
            SPDLOG_ERROR("finished GPU_OV_SET_VAA_BUF:{}", times[GPU_OV_SET_VAA_BUF][tSize]);
            gpuInferRequest.set_tensor(input, remoteTensor.second);
            gpuInferRequest.infer();
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
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError));
            EXPECT_EQ(0, clError);
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
            ov::Tensor inputOVTensor = inputs[j];
            ov::Tensor outputOVTensor = outputs[j];
            gpuInferRequest.set_tensor(input, inputs[j]);
            gpuInferRequest.set_tensor(output, outputs[j]);
            SPDLOG_INFO("set_callback");
            gpuInferRequest.set_callback([&gpuInferRequest, &callbackStruct, j](std::exception_ptr exception) {
                gpuInferRequest.set_callback([](std::exception_ptr exception) {});
                callbackStruct[j].signal.set_value(42);
            });
            SPDLOG_INFO("start async");
            gpuInferRequest.start_async();
            for (auto i = 0; i < iterations; ++i) {
                auto j = (i + 1) % 2;
                auto& gpuInferRequest = gpuInferRequests[j];

                ov::Tensor inputOVTensor = inputs[j];
                ov::Tensor outputOVTensor = outputs[j];
                gpuInferRequest.set_tensor(input, inputs[j]);
                gpuInferRequest.set_tensor(output, outputs[j]);
                gpuInferRequest.set_callback([&gpuInferRequest, &callbackStruct, j](std::exception_ptr exception) {
                    gpuInferRequest.set_callback([](std::exception_ptr exception) {});
                    callbackStruct[j].signal.set_value(42);
                });
                gpuInferRequest.start_async();
                // as we scheduled next infer we receive results from previous
                j = i % 2;
                auto callbackReturnValue = unblockSignal[j].get();
                callbackStruct[j].signal = std::promise<uint32_t>();
                unblockSignal[j] = callbackStruct[j].signal.get_future();
            }
            auto callbackReturnValue = unblockSignal[iterations % 2].get();
            auto stop = std::chrono::high_resolution_clock::now();
            SPDLOG_ERROR("Log plugin");
            ovms::logOVPluginConfig([&gpuCompiledModel](const std::string& key) { return gpuCompiledModel.get_property(key); }, " {someAuthor} ", " {some details} ");
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
            double fps = iterations / (times[t][s] / 1000.);  // FPS[Frame/second]
            std::cout << "" << fps * s << " \t\t ";
        }
        std::cout << std::endl;
    }
}

const std::string DUMMY_MODEL_GPU_CONFIG_PATH{"/ovms/src/test/configs/config_gpu_dummy.json"};
const std::string DUMMY_MODEL_CPU_CONFIG_PATH{"/ovms/src/test/configs/config_cpu_dummy.json"};

TEST_F(CAPINonCopy, SetOpenCLBufferAsInputTensor) {
    // start CAPI server
    // TODO load model with passed in context
    ServerGuard serverGuard(DUMMY_MODEL_GPU_CONFIG_PATH);
    OVMS_Server* cserver = serverGuard.server;

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
    cl_int clError;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));

    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)&openCLCppInputBuffer);
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
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);  // TODO
    EXPECT_EQ(capiDeviceId, 0);                  // TODO
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(in[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
}

class OpenCL : public ::testing::Test {
public:
    void SetUp() override {
        GPUEnvironment::skipWithoutGPU();
    }
};

TEST_F(OpenCL, UseDifferentContextWhenReadingAndWritingToBuffer) {
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
    cl_int clError;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(0, queue2.enqueueReadBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, outputBufferData));
    const float* outputData = reinterpret_cast<const float*>(outputBufferData);
    for (size_t i = 0; i < 10; ++i) {
        SPDLOG_INFO("OutputData[{}]:{}", i, *(outputData + i));
    }
}

TEST_F(CAPINonCopy, SetOpenCLBufferAsInputAndOutputTensor) {
    // start CAPI server
    ServerGuard serverGuard(DUMMY_MODEL_GPU_CONFIG_PATH);
    OVMS_Server* cserver = serverGuard.server;
    cl_context* contextFromModel;
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableContext(cserver, "dummy", 1, reinterpret_cast<void**>(&contextFromModel)));

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    // cl::Context openCLCppContext(openCLCContext);
    cl::Context openCLCppContext(*contextFromModel, retainCLContextOwnership);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, INITIAL_VALUE);
    void* inputBufferData = in.data();
    std::vector<float> out(10, 13.1);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int clError;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)&openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));     // device id ?? TODO
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    cl::vector<cl::Event> readEvents;
    EXPECT_EQ(0, queue.enqueueReadBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, outputBufferData, &readEvents));
    readEvents[0].wait();
    checkDummyOpenCLResponse(response, queue, INITIAL_VALUE + 1, FLOAT_TOLERANCE);
}
static void callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPIOpenCLCorrectness(OVMS_InferenceResponse*, uint32_t flag, void* userstruct);

TEST_F(CAPINonCopy, OpenCL_SyncWithCallbackDummy) {
    ServerGuard serverGuard(DUMMY_MODEL_GPU_CONFIG_PATH);
    OVMS_Server* cserver = serverGuard.server;
    cl_context* contextFromModel;
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableContext(cserver, "dummy", 1, reinterpret_cast<void**>(&contextFromModel)));

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);  // THIS is required to get correct device Id needed for queue
    cl::Context openCLCppContext(*contextFromModel, retainCLContextOwnership);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    cl_int clError;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties, &clError);
    EXPECT_EQ(0, clError);
    // create OpenCL buffers
    std::vector<float> in(10, INITIAL_VALUE);
    void* inputBufferData = in.data();
    std::vector<float> out(10, GARBAGE_VALUE);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, outputBufferData));
    // start CAPI server
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)&openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));
    OVMS_InferenceResponse* response = nullptr;
    // set callback
    uint32_t callbackUsed = 31;

    CallbackUnblockingStructWithQueue callbackStruct;
    SPDLOG_ERROR("ER:{}", (void*)&callbackStruct.signal);
    auto unblockSignal = callbackStruct.signal.get_future();
    callbackStruct.bufferAddr = &openCLCppOutputBuffer;
    callbackStruct.queue = &queue;
    SPDLOG_ERROR("ER:{}", (void*)&callbackStruct);

    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPIOpenCLCorrectness, reinterpret_cast<void*>(&callbackStruct)));
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // check is done in callback
    auto callbackReturnValue = unblockSignal.get();
    SPDLOG_INFO("Using callbacks!");
}
static OVMS_Server* startCAPIServerFromConfig(const std::string configPath) {
    std::string port = "9000";
    randomizeAndEnsureFree(port);
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

TEST_F(CAPINonCopy, OpenCL_SyncWithCallbackDummyCheckResetOutputGPU) {
    ServerGuard serverGuard(DUMMY_MODEL_GPU_CONFIG_PATH);
    OVMS_Server* cserver = serverGuard.server;

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
    std::vector<float> out(10, GARBAGE_VALUE + 13);
    void* outputBufferData = out.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int clError;
    auto openCLCppInputBufferPtr = std::make_unique<cl::Buffer>(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, nullptr, &clError);
    cl::Buffer& openCLCppInputBuffer = *openCLCppInputBufferPtr;
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    EXPECT_EQ(0, clError);
    auto openCLCppOutputBufferPtr = std::make_unique<cl::Buffer>(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, nullptr, &clError);
    cl::Buffer& openCLCppOutputBuffer = *openCLCppOutputBufferPtr;
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, outputBufferData));
    EXPECT_EQ(0, clError);

    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    uint32_t notUsedNum = 0;
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)&openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));
    OVMS_InferenceResponse* response = nullptr;
    // set callback
    uint32_t callbackUsed = 31;
    CallbackUnblockingStructWithQueue callbackStruct;
    auto unblockSignal = callbackStruct.signal.get_future();
    callbackStruct.bufferAddr = &openCLCppOutputBuffer;
    callbackStruct.queue = &queue;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPIOpenCLCorrectness, reinterpret_cast<void*>(&callbackStruct)));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request));
    auto callbackReturnValue = unblockSignal.get();
    openCLCppInputBufferPtr.reset();
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, nullptr, nullptr));
    std::vector<float> in2(10, INITIAL_VALUE * 2);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputRemoveData(request, DUMMY_MODEL_INPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(in2.data()), inputByteSize, OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputRemoveData(request, DUMMY_MODEL_OUTPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveOutput(request, DUMMY_MODEL_OUTPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    checkDummyResponse(response, INITIAL_VALUE * 2 + 1, FLOAT_TOLERANCE);
    OVMS_InferenceResponseDelete(response);
    std::vector<float> dataFromPreviousOutputBuffer(10, 1231521);
    // now we need to check if previous output wasn't changed
    EXPECT_EQ(0, queue.enqueueReadBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, dataFromPreviousOutputBuffer.data()));
    for (int i = 0; i < DUMMY_MODEL_INPUT_SIZE; ++i) {
        EXPECT_NEAR(dataFromPreviousOutputBuffer[i], INITIAL_VALUE + 1, FLOAT_TOLERANCE) << " at place i:" << i;
    }
}
TEST_F(CAPINonCopy, SyncWithoutCallbackDummyCheckResetOutputCPU) {
    ServerGuard serverGuard(DUMMY_MODEL_CPU_CONFIG_PATH);
    OVMS_Server* cserver = serverGuard.server;
    std::vector<float> in(10, INITIAL_VALUE);
    std::vector<float> out1(10, GARBAGE_VALUE);
    size_t inputByteSize = sizeof(float) * in.size();
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(in.data()), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(out1.data()), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // check
    checkDummyResponse(response, INITIAL_VALUE + 1, FLOAT_TOLERANCE);
    OVMS_InferenceResponseDelete(response);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, nullptr, nullptr));
    // now check with default output buffer
    std::vector<float> in2(10, INITIAL_VALUE + 42);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputRemoveData(request, DUMMY_MODEL_INPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(in2.data()), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputRemoveData(request, DUMMY_MODEL_OUTPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveOutput(request, DUMMY_MODEL_OUTPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    checkDummyResponse(response, INITIAL_VALUE + 42 + 1, FLOAT_TOLERANCE);
    // intentional check for original output buffer if they were not overridden
    for (size_t i = 0; i < out1.size(); ++i) {
        EXPECT_NEAR(INITIAL_VALUE + 1, out1[i], FLOAT_TOLERANCE) << "Different at:" << i << " place.";
    }
    OVMS_InferenceResponseDelete(response);
}
TEST_F(CAPINonCopy, AsyncDummyCheckResetOutputCPU) {
    ServerGuard serverGuard(DUMMY_MODEL_CPU_CONFIG_PATH);
    OVMS_Server* cserver = serverGuard.server;
    std::vector<float> in(10, INITIAL_VALUE);
    std::vector<float> out1(10, GARBAGE_VALUE);
    size_t inputByteSize = sizeof(float) * in.size();
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(in.data()), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(out1.data()), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    // perform 1st inference
    CallbackUnblockingCPUStruct callbackStruct;
    auto unblockSignal = callbackStruct.signal.get_future();
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, callbackUnblocking, reinterpret_cast<void*>(&callbackStruct)));
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request));
    unblockSignal.get();
    // check
    checkDummyResponse(callbackStruct.response, INITIAL_VALUE + 1, FLOAT_TOLERANCE);
    OVMS_InferenceResponseDelete(callbackStruct.response);
    callbackStruct.response = nullptr;
    // perform 2nd inference
    callbackStruct.signal = std::promise<void>();
    unblockSignal = callbackStruct.signal.get_future();
    // now check with default output buffer
    std::vector<float> in2(10, INITIAL_VALUE + 42);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputRemoveData(request, DUMMY_MODEL_INPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(in2.data()), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputRemoveData(request, DUMMY_MODEL_OUTPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveOutput(request, DUMMY_MODEL_OUTPUT_NAME));

    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request));
    unblockSignal.get();
    checkDummyResponse(callbackStruct.response, INITIAL_VALUE + 42 + 1, FLOAT_TOLERANCE);
    OVMS_InferenceResponseDelete(callbackStruct.response);
    // intentional check for original output buffer if they were not overridden
    for (size_t i = 0; i < out1.size(); ++i) {
        EXPECT_NEAR(INITIAL_VALUE + 1, out1[i], FLOAT_TOLERANCE) << "Different at:" << i << " place.";
    }
}

static void callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPIOpenCLCorrectness(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct);
static void callbackUnblockingAndFreeingRequest(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct);

cl::CommandQueue* globalQueue = nullptr;

TEST_F(CAPINonCopy, AsyncWithCallbackDummy) {
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
    cl_int clError;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    // start CAPI server
    std::string port = "9000";
    randomizeAndEnsureFree(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, DUMMY_MODEL_GPU_CONFIG_PATH.c_str()));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::vector<float> data(DUMMY_MODEL_INPUT_SIZE, INITIAL_VALUE);
    SPDLOG_DEBUG("openCLCppInputBuffer:{}", (void*)&openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    SPDLOG_DEBUG("openCLCppOutputBuffer:{}", (void*)&openCLCppOutputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(&openCLCppOutputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
    // set callback
    CallbackUnblockingStructWithQueue callbackStruct;
    auto unblockSignal = callbackStruct.signal.get_future();
    callbackStruct.bufferAddr = &openCLCppOutputBuffer;
    callbackStruct.queue = &queue;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPIOpenCLCorrectness, reinterpret_cast<void*>(&callbackStruct)));
    // infer
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request));
    // check
    auto callbackReturnValue = unblockSignal.get();

    EXPECT_EQ(0, queue.enqueueReadBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, outputBufferData));
    const float* outputData = reinterpret_cast<const float*>(outputBufferData);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(in[i] + 1, outputData[i], FLOAT_TOLERANCE) << "Different at:" << i << " place.";
    }
    ASSERT_EQ(42, callbackReturnValue);
    SPDLOG_INFO("Using callbacks!");
    // TODO cleanup settings
    OVMS_ServerDelete(cserver);
}

class CAPIGPUPerfComparison : public TestWithTempDir {
protected:
    const uint32_t afterConfigChangeLoadTimeMs = 50;
    const int stressIterationsLimit = 5000;

    std::string configFilePath;
    std::string ovmsConfig;
    std::string modelPath;
    OVMS_Server* cserver;

public:
    void SetUp() override {
        GPUEnvironment::skipWithoutGPU();
        TestWithTempDir::SetUp();
    }
    void SetUpConfig(const std::string& configContent, size_t elementsCount) {
        ovmsConfig = configContent;
        const std::string STRING_TO_REPLACE{"SECOND_DIM_TO_REPLACE"};
        auto it = ovmsConfig.find("SECOND_DIM_TO_REPLACE");
        if (it != std::string::npos) {
            ovmsConfig.replace(it, STRING_TO_REPLACE.size(), std::to_string(elementsCount));
        }
        configFilePath = directoryPath + "/ovms_config.json";
        SPDLOG_INFO("ConfigContent:{}", ovmsConfig);
        SPDLOG_INFO("config path:{}", configFilePath);
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
    cl_int clError;
    std::vector<cl::Buffer> openCLCppInputBuffer;
    openCLCppInputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
    EXPECT_EQ(0, clError);
    openCLCppInputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
    EXPECT_EQ(0, clError);
    std::vector<cl::Buffer> openCLCppOutputBuffer;
    openCLCppOutputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
    EXPECT_EQ(0, clError);
    openCLCppOutputBuffer.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError));
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer[0], queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer[1], queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
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
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request[0], callbackUnblockingAndFreeingRequest, reinterpret_cast<void*>(&callbackStruct[0])));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request[1], callbackUnblockingAndFreeingRequest, reinterpret_cast<void*>(&callbackStruct[1])));
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
    auto callbackReturnValue = unblockSignal[iterations % 2].get();
    auto stop = std::chrono::high_resolution_clock::now();
    times[1] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    // TODO check non-remote tensors async
    // TODO check sync non-remote tensors
    // TODO check sync remote tensors

    OVMS_ServerDelete(cserver);
    double fps = iterations / (times[1] / 1'000.);  // FPS[Frame/second]
    std::cout << "" << fps * elementsCount << " \t\t ";
}

TEST_F(OpenVINOGPUContextFromModel, UseCLContextForBuffersOVContextForInference) {
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
    cl_int clError;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    EXPECT_EQ(0, queue.enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
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
    std::unique_ptr<ov::intel_gpu::ocl::ClContext> oclContextCpp;
    cl_context oclContextC;
    {
        auto oclContext = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
        oclContextCpp = std::make_unique<ov::intel_gpu::ocl::ClContext>(oclContext);
        oclContextC = oclContextCpp->get();
    }
    SPDLOG_ERROR("{}", (void*)oclContextC);
    // opencltensorfactory.hpp
    auto inputTensor = oclContextCpp->create_tensor(type, shape, openCLCppInputBuffer);
    auto outputTensor = oclContextCpp->create_tensor(type, shape, openCLCppOutputBuffer);
    request.set_tensor("b", inputTensor);
    request.set_tensor("a", outputTensor);
    request.start_async();
    request.wait();
    EXPECT_EQ(0, queue.enqueueReadBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, outputBufferData));
    const float* outputData = reinterpret_cast<const float*>(outputBufferData);
    for (size_t i = 0; i < 10; ++i) {
        SPDLOG_ERROR("ER:{}", *(outputData + i));
    }
}

TEST_F(OpenVINOGPUContextFromModel, OutputTensorHasBiggerUnderlyingOCLBufferThanNeededPass) {
    cl::Context openCLCppContext(ctxFromModel, retainCLContextOwnership);
    cl_int clError;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize * 2, NULL, &clError);
    EXPECT_EQ(0, clError);
    auto inputOVOCLBufferTensor = ovGpuOclContext->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = ovGpuOclContext->create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    EXPECT_EQ(0, queueFromModelContext->enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    inferRequest->set_tensor(output, outputOVOCLBufferTensor);
    inferRequest->infer();
    std::vector<float> out(100, GARBAGE_VALUE);
    void* bufferOut = out.data();
    EXPECT_EQ(0, queueFromModelContext->enqueueReadBuffer(openCLCppOutputBuffer, queueReadWriteBlockingTrue, 0, outputByteSize, bufferOut));
    const float* outputData = reinterpret_cast<const float*>(bufferOut);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(in[i] + 1, outputData[i], FLOAT_TOLERANCE) << "Different at:" << i << " place.";
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
TEST_F(OpenVINOGPUContextFromModel, OutputTensorHasBiggerShapeAndOCLBufferThanNeededThrowsOnSetTensor) {
    cl::Context openCLCppContext(ctxFromModel, retainCLContextOwnership);
    cl_int clError;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize * 2, NULL, &clError);
    EXPECT_EQ(0, clError);
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(inputSecondDim * 2);
    auto inputOVOCLBufferTensor = ovGpuOclContext->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = ovGpuOclContext->create_tensor(output->get_element_type(), ovShape, openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    EXPECT_EQ(0, queueFromModelContext->enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    EXPECT_THROW(inferRequest->set_tensor(output, outputOVOCLBufferTensor), ov::Exception);
}
TEST_F(OpenVINOGPUContextFromModel, OutputTensorHasSmallerUnderlyingOCLBufferThanNeededThrowsOnCreateRemoteTensor) {
    cl::Context openCLCppContext(ctxFromModel, retainCLContextOwnership);
    cl_int clError;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize / 2, NULL, &clError);
    EXPECT_EQ(0, clError);
    auto inputOVOCLBufferTensor = ovGpuOclContext->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    EXPECT_THROW(ovGpuOclContext->create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer), ov::Exception);
    // we will put data into input buffer
}
TEST_F(OpenVINOGPUContextFromModel, OutputTensorHasSmallerShapeAndUnderlyingOCLBufferThanNeededThrowsOnSetTensor) {
    cl::Context openCLCppContext(ctxFromModel, retainCLContextOwnership);
    cl_int clError;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    float divisionFactor = 2;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize / divisionFactor, NULL, &clError);
    EXPECT_EQ(0, clError);
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(inputSecondDim / divisionFactor);
    auto inputOVOCLBufferTensor = ovGpuOclContext->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = ovGpuOclContext->create_tensor(output->get_element_type(), ovShape, openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    EXPECT_EQ(0, queueFromModelContext->enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    EXPECT_THROW(inferRequest->set_tensor(output, outputOVOCLBufferTensor), ov::Exception);
}
TEST_F(OpenVINOGPUContextFromModel, OutputTensorHasSmallerShapeAndAppropriateOCLBufferThanNeededThrowsOnSetTensor) {
    cl::Context openCLCppContext(ctxFromModel, retainCLContextOwnership);
    cl_int clError;
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    float divisionFactor = 2;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &clError);
    EXPECT_EQ(0, clError);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &clError);
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(inputSecondDim / divisionFactor);
    auto inputOVOCLBufferTensor = ovGpuOclContext->create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = ovGpuOclContext->create_tensor(output->get_element_type(), ovShape, openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(100, 0.1);
    void* inputBufferData = in.data();
    EXPECT_EQ(0, queueFromModelContext->enqueueWriteBuffer(openCLCppInputBuffer, queueReadWriteBlockingTrue, 0, inputByteSize, inputBufferData));
    inferRequest->set_tensor(input, inputOVOCLBufferTensor);
    EXPECT_THROW(inferRequest->set_tensor(output, outputOVOCLBufferTensor), ov::Exception);
}

static void callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPIOpenCLCorrectness(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    SPDLOG_INFO("Using callback: callbackMarkingItWasUsedWith42AndUnblockingAndCheckingCAPIOpenCLCorrectness!");
    CallbackUnblockingStructWithQueue* callbackUnblockingStruct = reinterpret_cast<CallbackUnblockingStructWithQueue*>(userStruct);
    SPDLOG_ERROR("ER:{}", userStruct);
    SPDLOG_ERROR("ER:{}", (void*)&callbackUnblockingStruct->signal);
    callbackUnblockingStruct->signal.set_value(42);
    checkDummyOpenCLResponse(response, *callbackUnblockingStruct->queue, INITIAL_VALUE + 1, FLOAT_TOLERANCE);
    OVMS_InferenceResponseDelete(response);
}
static void callbackUnblockingAndFreeingRequest(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    SPDLOG_DEBUG("Using callback: callbackUnblockingAndFreeingRequest!");
    CallbackUnblockingStruct* callbackUnblockingStruct = reinterpret_cast<CallbackUnblockingStruct*>(userStruct);
    callbackUnblockingStruct->signal.set_value(42);
    OVMS_InferenceResponseDelete(response);
}
static void callbackUnblocking(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    SPDLOG_ERROR("Using callback: callbackUnblocking!");
    CallbackUnblockingCPUStruct* callbackStruct = reinterpret_cast<CallbackUnblockingCPUStruct*>(userStruct);
    callbackStruct->signal.set_value();
    callbackStruct->response = response;
    SPDLOG_ERROR("Using callback: callbackUnblocking!:{}", (void*)callbackStruct->response);
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

// TODO
// test inference with CPU with callback
// test inferene with GPU with different context than from model
// sync/async, with without callback
// split serialization/deserialization
// TBD if we should expose extraction of context from model
// remove logs
// verify TODOS
// replan
// test negative paths with set callback
// add negative result signaling with callback
// split tests between files
// refactor tests
// test one input/output on device, one on cpu
// ensure callback & output tensor is reset after inference
// add tests after capi with output tensors set on the same ov::InferReq
//
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
