// Copyright (c) 2026 Intel Corporation
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

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <chrono>
#include <gtest/gtest.h>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <openvino/runtime/properties.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif
#include "src/ovms.h"
#include "mpi_inference_benchmark_utils.hpp"

extern "C" void* OVMS_MPGetExternalServerHandle() {
    ADD_FAILURE() << "Benchmark must supply the SERVER side packet";
    return nullptr;
}

namespace {
namespace bench = inference_benchmark;

bool check(OVMS_Status* status) {
    if (!status)
        return true;
    const char* details = nullptr;
    OVMS_Status* detailsStatus = OVMS_StatusDetails(status, &details);
    ADD_FAILURE() << (details ? details : "OVMS C-API operation failed");
    if (detailsStatus)
        OVMS_StatusDelete(detailsStatus);
    OVMS_StatusDelete(status);
    return false;
}

class MpiCapiBenchmark : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(std::filesystem::exists(bench::modelPath()));
#ifdef _WIN32
        D3D_FEATURE_LEVEL level;
        ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, nullptr, 0, D3D11_SDK_VERSION, &device_, &level, nullptr)));
        imp_video_set_d3d11_device(device_.Get());
#else
        descriptor_ = open(bench::environment("OVMS_PERF_RENDER_NODE", "/dev/dri/renderD128").c_str(), O_RDWR);
        ASSERT_GE(descriptor_, 0);
        display_ = vaGetDisplayDRM(descriptor_);
        ASSERT_NE(display_, nullptr);
        int major = 0;
        int minor = 0;
        const VAStatus initialize = vaInitialize(display_, &major, &minor);
        initialized_ = initialize == VA_STATUS_SUCCESS;
        ASSERT_TRUE(initialized_) << vaErrorStr(initialize);
        imp_video_set_va_display(display_);
#endif
        const std::string testName = testing::UnitTest::GetInstance()->current_test_info()->name();
        directory_ = std::filesystem::path(bench::environment("TEST_TMPDIR", (bench::root() / "logs").string())) /
                     ("mpi_capi_" + testName);
        std::filesystem::create_directories(directory_ / "model/1");
        ov::Core core;
        ov::serialize(core.read_model(bench::modelPath()), (directory_ / "model/1/model.xml").string());
        rapidjson::Document document;
        document.SetObject();
        auto& allocator = document.GetAllocator();
        rapidjson::Value model(rapidjson::kObjectType);
        model.AddMember("name", rapidjson::Value(benchmark_config::kModelName, allocator), allocator);
        const std::string basePath = (directory_ / "model").generic_string();
        model.AddMember("base_path", rapidjson::Value(basePath.c_str(), allocator), allocator);
        model.AddMember("target_device", "GPU", allocator);
        model.AddMember("nireq", benchmark_config::kRequestCount, allocator);
        rapidjson::Value properties(rapidjson::kObjectType);
        properties.AddMember("PERFORMANCE_HINT", rapidjson::Value(benchmark_config::kPerformanceHint, allocator), allocator);
        properties.AddMember("NUM_STREAMS", rapidjson::Value(benchmark_config::kNumStreams, allocator), allocator);
        model.AddMember("plugin_config", properties, allocator);
        rapidjson::Value entry(rapidjson::kObjectType);
        entry.AddMember("config", model, allocator);
        rapidjson::Value models(rapidjson::kArrayType);
        models.PushBack(entry, allocator);
        document.AddMember("model_config_list", models, allocator);
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        ASSERT_TRUE(document.Accept(writer));
        const std::string configPath = (directory_ / "config.json").string();
        {
            std::ofstream config(configPath);
            config << buffer.GetString();
            ASSERT_TRUE(config.good());
        }
        ASSERT_TRUE(check(OVMS_ServerNew(&server_)));
#ifdef _WIN32
        ASSERT_TRUE(check(OVMS_ServerSetGlobalD3D11Device(server_, device_.Get())));
#else
        ASSERT_TRUE(check(OVMS_ServerSetGlobalVADisplay(server_, display_)));
#endif
        OVMS_ServerSettings* rawSettings = nullptr;
        OVMS_ModelsSettings* rawModels = nullptr;
        ASSERT_TRUE(check(OVMS_ServerSettingsNew(&rawSettings)));
        std::unique_ptr<OVMS_ServerSettings, decltype(&OVMS_ServerSettingsDelete)> settings(rawSettings, OVMS_ServerSettingsDelete);
        ASSERT_TRUE(check(OVMS_ModelsSettingsNew(&rawModels)));
        std::unique_ptr<OVMS_ModelsSettings, decltype(&OVMS_ModelsSettingsDelete)> modelsSettings(rawModels, OVMS_ModelsSettingsDelete);
        ASSERT_TRUE(check(OVMS_ServerSettingsSetGrpcPort(settings.get(), 0)));
        const std::string logLevel = bench::environment("OVMS_CAPI_LOG_LEVEL", "INFO");
        if (logLevel == "DEBUG")
            ASSERT_TRUE(check(OVMS_ServerSettingsSetLogLevel(settings.get(), OVMS_LOG_DEBUG)));
        else if (logLevel == "TRACE")
            ASSERT_TRUE(check(OVMS_ServerSettingsSetLogLevel(settings.get(), OVMS_LOG_TRACE)));
        else
            ASSERT_EQ(logLevel, "INFO") << "OVMS_CAPI_LOG_LEVEL must be INFO, DEBUG, or TRACE";
        ASSERT_TRUE(check(OVMS_ModelsSettingsSetConfigPath(modelsSettings.get(), configPath.c_str())));
        ASSERT_TRUE(check(OVMS_ServerStartFromConfigurationFile(server_, settings.get(), modelsSettings.get())));
    }

    void TearDown() override {
        if (server_)
            OVMS_ServerDelete(server_);
#ifdef _WIN32
        imp_video_set_d3d11_device(nullptr);
#else
        imp_video_set_va_display(nullptr);
        if (initialized_)
            vaTerminate(display_);
        if (descriptor_ >= 0)
            close(descriptor_);
#endif
        if (!directory_.empty())
            std::filesystem::remove_all(directory_);
    }

    void run(bool split) {
        bench::measureGraph(split ? "mpi_capi_two_calculators" : "mpi_capi_one_calculator", true, split,
            {{"server", mediapipe::MakePacket<OVMS_Server*>(server_)},
                {"servable_name", mediapipe::MakePacket<std::string>(benchmark_config::kModelName)},
                {"servable_version", mediapipe::MakePacket<int>(1)},
                {"input_name", mediapipe::MakePacket<std::string>(benchmark_config::kInputName)}});
    }

#ifndef _WIN32
    void runNativeSurfaceReplay(bool paired) {
        std::vector<double> directSamples;
        std::vector<double> capiSamples;
        const std::string inputName = benchmark_config::kInputName;
        const std::string yName = inputName + "/y";
        const std::string uvName = inputName + "/uv";
        const int64_t shapeY[] = {1, benchmark_config::kModelHeight, benchmark_config::kModelWidth, 1};
        const int64_t shapeUv[] = {1, benchmark_config::kModelHeight / 2, benchmark_config::kModelWidth / 2, 2};
        const size_t bytesY = static_cast<size_t>(benchmark_config::kModelWidth) * benchmark_config::kModelHeight;
        const size_t bytesUv = bytesY / 2;

        for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
            bench::DecodeStream decode;
            decode.open();
            ov::Core core;
            std::unique_ptr<ov::intel_gpu::ocl::VAContext> context;
            ov::CompiledModel compiled;
            ov::InferRequest directRequest;
            OVMS_InferenceRequest* rawRequest = nullptr;
            ASSERT_TRUE(check(OVMS_InferenceRequestNew(&rawRequest, server_, benchmark_config::kModelName, 1)));
            std::unique_ptr<OVMS_InferenceRequest, decltype(&OVMS_InferenceRequestDelete)> request(
                rawRequest, OVMS_InferenceRequestDelete);
            ASSERT_TRUE(check(OVMS_InferenceRequestAddInput(request.get(), yName.c_str(),
                OVMS_DATATYPE_U8, shapeY, 4)));
            ASSERT_TRUE(check(OVMS_InferenceRequestAddInput(request.get(), uvName.c_str(),
                OVMS_DATATYPE_U8, shapeUv, 4)));

            int frames = 0;
            bool requestHasBoundData = false;
            std::chrono::nanoseconds directImport{};
            std::chrono::nanoseconds directBinding{};
            std::chrono::nanoseconds directInfer{};
            std::chrono::nanoseconds directOutput{};
            std::chrono::nanoseconds directWall{};
            std::chrono::nanoseconds capiRemove{};
            std::chrono::nanoseconds capiBinding{};
            std::chrono::nanoseconds capiInference{};
            std::chrono::nanoseconds capiOutput{};
            std::chrono::nanoseconds capiResponseDelete{};
            std::chrono::nanoseconds capiWall{};
            std::chrono::nanoseconds capiInferenceDirectFirst{};
            std::chrono::nanoseconds capiInferenceCapiFirst{};
            std::chrono::nanoseconds capiWallDirectFirst{};
            std::chrono::nanoseconds capiWallCapiFirst{};
            size_t directFirstFrames = 0;
            size_t capiFirstFrames = 0;

            for (;;) {
                imp_tensor_t* rawTensor = nullptr;
                const imp_status_t readStatus = imp_video_read_frame(&rawTensor, decode.stream, 0);
                std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(rawTensor, imp_tensor_release);
                if (readStatus == IMP_ERROR_STREAM_END)
                    break;
                ASSERT_EQ(readStatus, IMP_OK);
                ASSERT_NE(tensor, nullptr);
                uint32_t surface = 0;
                void* display = nullptr;
                int width = 0;
                int height = 0;
                ASSERT_EQ(imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height), IMP_OK);
                ASSERT_EQ(display, display_) << "Decoded surface must use the server-registered VADisplay";
                ASSERT_EQ(width, benchmark_config::kModelWidth);
                ASSERT_EQ(height, benchmark_config::kModelHeight);
                const void* surfaceHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(surface));
                const bool measuredFrame = frames >= benchmark_config::kWarmupFrames;
                int directDetections = -1;
                int capiDetections = -1;

                auto runDirect = [&]() {
                    const auto wallStart = bench::Clock::now();
                    if (!context) {
                        context = std::make_unique<ov::intel_gpu::ocl::VAContext>(core, display);
                        compiled = core.compile_model(bench::model(core, true), *context, bench::properties());
                        directRequest = compiled.create_infer_request();
                    }
                    const auto importStart = bench::Clock::now();
                    auto planes = context->create_tensor_nv12(height, width, surface);
                    const auto importStop = bench::Clock::now();
                    const auto bindingStart = bench::Clock::now();
                    directRequest.set_tensor(bench::kYInputName, planes.first);
                    directRequest.set_tensor(bench::kUvInputName, planes.second);
                    const auto bindingStop = bench::Clock::now();
                    const auto inferStart = bench::Clock::now();
                    directRequest.infer();
                    const auto inferStop = bench::Clock::now();
                    const auto outputStart = bench::Clock::now();
                    const ov::Tensor output = directRequest.get_output_tensor();
                    const float* values = output.data<const float>();
                    directDetections = 0;
                    for (size_t result = 0; result + 7 <= output.get_size(); result += 7) {
                        if (values[result + 2] >= 0.5f)
                            ++directDetections;
                    }
                    const auto outputStop = bench::Clock::now();
                    if (measuredFrame) {
                        directImport += importStop - importStart;
                        directBinding += bindingStop - bindingStart;
                        directInfer += inferStop - inferStart;
                        directOutput += outputStop - outputStart;
                        directWall += outputStop - wallStart;
                    }
                };

                auto runCapi = [&]() -> bool {
                    const auto wallStart = bench::Clock::now();
                    if (requestHasBoundData) {
                        const auto removeStart = bench::Clock::now();
                        if (!check(OVMS_InferenceRequestInputRemoveData(request.get(), yName.c_str())) ||
                            !check(OVMS_InferenceRequestInputRemoveData(request.get(), uvName.c_str())))
                            return false;
                        const auto removeStop = bench::Clock::now();
                        if (measuredFrame)
                            capiRemove += removeStop - removeStart;
                    }
                    const auto bindingStart = bench::Clock::now();
                    if (!check(OVMS_InferenceRequestInputSetData(request.get(), yName.c_str(), surfaceHandle,
                            bytesY, OVMS_BUFFERTYPE_VASURFACE_Y, 1)) ||
                        !check(OVMS_InferenceRequestInputSetData(request.get(), uvName.c_str(), surfaceHandle,
                            bytesUv, OVMS_BUFFERTYPE_VASURFACE_UV, 1)))
                        return false;
                    const auto bindingStop = bench::Clock::now();
                    if (measuredFrame)
                        capiBinding += bindingStop - bindingStart;
                    requestHasBoundData = true;

                    OVMS_InferenceResponse* rawResponse = nullptr;
                    const auto inferenceStart = bench::Clock::now();
                    if (!check(OVMS_Inference(server_, request.get(), &rawResponse)))
                        return false;
                    const auto inferenceStop = bench::Clock::now();
                    if (measuredFrame) {
                        capiInference += inferenceStop - inferenceStart;
                        if ((frames & 1) == 0)
                            capiInferenceDirectFirst += inferenceStop - inferenceStart;
                        else
                            capiInferenceCapiFirst += inferenceStop - inferenceStart;
                    }
                    std::unique_ptr<OVMS_InferenceResponse, decltype(&OVMS_InferenceResponseDelete)> response(
                        rawResponse, OVMS_InferenceResponseDelete);

                    const auto outputStart = bench::Clock::now();
                    const void* outputData = nullptr;
                    size_t outputBytes = 0;
                    uint32_t outputId = 0;
                    OVMS_DataType outputType = static_cast<OVMS_DataType>(199);
                    const int64_t* outputShape = nullptr;
                    size_t outputDimensions = 0;
                    OVMS_BufferType outputBufferType = OVMS_BUFFERTYPE_CPU;
                    uint32_t outputDeviceId = 0;
                    const char* outputName = nullptr;
                    if (!check(OVMS_InferenceResponseOutput(response.get(), outputId, &outputName, &outputType,
                            &outputShape, &outputDimensions, &outputData, &outputBytes,
                            &outputBufferType, &outputDeviceId)))
                        return false;
                    if (outputType != OVMS_DATATYPE_FP32 || outputData == nullptr) {
                        ADD_FAILURE() << "Expected an FP32 C-API detection output";
                        return false;
                    }
                    const float* values = static_cast<const float*>(outputData);
                    capiDetections = 0;
                    for (size_t result = 0; result + 7 <= outputBytes / sizeof(float); result += 7) {
                        if (values[result + 2] >= 0.5f)
                            ++capiDetections;
                    }
                    const auto outputStop = bench::Clock::now();
                    if (measuredFrame)
                        capiOutput += outputStop - outputStart;
                    const auto deleteStart = bench::Clock::now();
                    response.reset();
                    const auto deleteStop = bench::Clock::now();
                    if (measuredFrame) {
                        capiResponseDelete += deleteStop - deleteStart;
                        capiWall += deleteStop - wallStart;
                        if ((frames & 1) == 0) {
                            capiWallDirectFirst += deleteStop - wallStart;
                            ++directFirstFrames;
                        } else {
                            capiWallCapiFirst += deleteStop - wallStart;
                            ++capiFirstFrames;
                        }
                    }
                    return true;
                };

                if (paired) {
                    if ((frames & 1) == 0) {
                        runDirect();
                        ASSERT_TRUE(runCapi());
                    } else {
                        ASSERT_TRUE(runCapi());
                        runDirect();
                    }
                    ASSERT_EQ(directDetections, capiDetections)
                        << "Direct OV and C-API outputs differ at frame " << frames;
                } else {
                    ASSERT_TRUE(runCapi());
                }
                ++frames;
            }
            ASSERT_EQ(frames, bench::kTotalFrames) << "Source must reach EOS, not a MAX_FRAMES cutoff";

            const double scale = 1.0 / benchmark_config::kMeasuredFrames;
            const double directMs = paired
                ? std::chrono::duration<double, std::milli>(directWall).count() * scale
                : 0.0;
            const double capiMs = std::chrono::duration<double, std::milli>(capiWall).count() * scale;
            if (paired)
                directSamples.push_back(directMs);
            capiSamples.push_back(capiMs);
            std::string executionDevice = "unavailable";
            std::string inferencePrecision = "unavailable";
            if (paired) {
                try {
                    const auto devices = compiled.get_property(ov::execution_devices);
                    if (!devices.empty())
                        executionDevice = devices.front();
                } catch (const std::exception&) {
                }
                try {
                    inferencePrecision = compiled.get_property(ov::hint::inference_precision).to_string();
                } catch (const std::exception&) {
                }
            }
            std::cout << "E3_CONFIG sample=" << sample + 1
                      << " paired=" << (paired ? "1" : "0")
                      << " direct_device=" << executionDevice
                      << " direct_precision=" << inferencePrecision
                      << " capi_requested_device=GPU capi_requested_precision=unavailable"
                      << " performance_hint=" << benchmark_config::kPerformanceHint
                      << " num_streams=" << benchmark_config::kNumStreams
                      << " capi_pool_size=" << benchmark_config::kRequestCount
                      << " outstanding_requests=1 same_va_surface=1 alternating_order=" << (paired ? "1" : "0")
                      << " consumed=" << frames << " eos=1" << std::endl;
            if (paired) {
                std::cout << std::fixed << std::setprecision(6)
                          << "E3_DIRECT sample=" << sample + 1
                          << " import_ms_per_frame=" << std::chrono::duration<double, std::milli>(directImport).count() * scale
                          << " binding_ms_per_frame=" << std::chrono::duration<double, std::milli>(directBinding).count() * scale
                          << " infer_ms_per_frame=" << std::chrono::duration<double, std::milli>(directInfer).count() * scale
                          << " output_ms_per_frame=" << std::chrono::duration<double, std::milli>(directOutput).count() * scale
                          << " paired_path_ms_per_frame=" << directMs
                          << " paired_path_fps=" << 1000.0 / directMs << std::endl;
            }
            std::cout << "E3_CAPI sample=" << sample + 1
                      << " remove_ms_per_frame=" << std::chrono::duration<double, std::milli>(capiRemove).count() * scale
                      << " binding_ms_per_frame=" << std::chrono::duration<double, std::milli>(capiBinding).count() * scale
                      << " ovms_inference_ms_per_frame=" << std::chrono::duration<double, std::milli>(capiInference).count() * scale
                      << " output_ms_per_frame=" << std::chrono::duration<double, std::milli>(capiOutput).count() * scale
                      << " response_delete_ms_per_frame=" << std::chrono::duration<double, std::milli>(capiResponseDelete).count() * scale
                      << " paired_path_ms_per_frame=" << capiMs
                      << " paired_path_fps=" << 1000.0 / capiMs << std::endl;
                if (paired)
                    std::cout << "E3_ORDER sample=" << sample + 1
                     << " measured_direct_first_frames=" << directFirstFrames
                     << " capi_first_frames=" << capiFirstFrames
                     << " capi_infer_after_direct_ms_per_frame="
                     << std::chrono::duration<double, std::milli>(capiInferenceDirectFirst).count() /
                         std::max<size_t>(directFirstFrames, 1)
                     << " capi_infer_before_direct_ms_per_frame="
                     << std::chrono::duration<double, std::milli>(capiInferenceCapiFirst).count() /
                         std::max<size_t>(capiFirstFrames, 1)
                     << " capi_wall_after_direct_ms_per_frame="
                     << std::chrono::duration<double, std::milli>(capiWallDirectFirst).count() /
                         std::max<size_t>(directFirstFrames, 1)
                     << " capi_wall_before_direct_ms_per_frame="
                     << std::chrono::duration<double, std::milli>(capiWallCapiFirst).count() /
                         std::max<size_t>(capiFirstFrames, 1)
                          << std::endl;
        }
        if (paired)
            bench::report("e3_paired_direct_ov", directSamples);
        bench::report(paired ? "e3_paired_capi" : "e3_capi_native_surface_only", capiSamples);
    }

    struct FairPathResult {
        std::vector<int> detections;
        double millisecondsPerFrame = 0;
    };

    void runFairDirectPath(FairPathResult& result) {
        ov::Core core;
        std::unique_ptr<ov::intel_gpu::ocl::VAContext> context;
        ov::CompiledModel compiled;
        ov::InferRequest request;
        try {
            context = std::make_unique<ov::intel_gpu::ocl::VAContext>(core, display_);
            compiled = core.compile_model(bench::model(core, true), *context, bench::properties());
            request = compiled.create_infer_request();
        } catch (const std::exception& error) {
            FAIL() << "Direct OV initialization failed before decode: " << error.what();
            return;
        }

        bench::DecodeStream decode;
        decode.open();
        int frames = 0;
        bench::Clock::time_point start;
        bench::Clock::time_point stop;
        for (;;) {
            imp_tensor_t* rawTensor = nullptr;
            const imp_status_t readStatus = imp_video_read_frame(&rawTensor, decode.stream, 0);
            std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(rawTensor, imp_tensor_release);
            if (readStatus == IMP_ERROR_STREAM_END)
                break;
            ASSERT_EQ(readStatus, IMP_OK);
            ASSERT_NE(tensor, nullptr);

            uint32_t surface = 0;
            void* display = nullptr;
            int width = 0;
            int height = 0;
            ASSERT_EQ(imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height), IMP_OK);
            ASSERT_EQ(display, display_);
            ASSERT_EQ(width, benchmark_config::kModelWidth);
            ASSERT_EQ(height, benchmark_config::kModelHeight);
            auto planes = context->create_tensor_nv12(height, width, surface);
            request.set_tensor(bench::kYInputName, planes.first);
            request.set_tensor(bench::kUvInputName, planes.second);
            request.infer();
            const ov::Tensor output = request.get_output_tensor();
            const float* values = output.data<const float>();
            int detections = 0;
            for (size_t resultIndex = 0; resultIndex + 7 <= output.get_size(); resultIndex += 7) {
                if (values[resultIndex + 2] >= 0.5f)
                    ++detections;
            }
            result.detections.push_back(detections);
            tensor.reset();
            ++frames;
            if (frames == benchmark_config::kWarmupFrames)
                start = bench::Clock::now();
            if (frames == bench::kTotalFrames)
                stop = bench::Clock::now();
        }
        ASSERT_EQ(frames, bench::kTotalFrames) << "Direct path must consume all 250 frames and reach EOS";
        result.millisecondsPerFrame = bench::milliseconds(start, stop) / benchmark_config::kMeasuredFrames;
    }

    void runFairCapiPath(FairPathResult& result) {
        const std::string inputName = benchmark_config::kInputName;
        const std::string yName = inputName + "/y";
        const std::string uvName = inputName + "/uv";
        const int64_t shapeY[] = {1, benchmark_config::kModelHeight, benchmark_config::kModelWidth, 1};
        const int64_t shapeUv[] = {1, benchmark_config::kModelHeight / 2, benchmark_config::kModelWidth / 2, 2};
        const size_t bytesY = static_cast<size_t>(benchmark_config::kModelWidth) * benchmark_config::kModelHeight;
        const size_t bytesUv = bytesY / 2;

        OVMS_InferenceRequest* rawRequest = nullptr;
        ASSERT_TRUE(check(OVMS_InferenceRequestNew(&rawRequest, server_, benchmark_config::kModelName, 1)));
        std::unique_ptr<OVMS_InferenceRequest, decltype(&OVMS_InferenceRequestDelete)> request(
            rawRequest, OVMS_InferenceRequestDelete);
        ASSERT_TRUE(check(OVMS_InferenceRequestAddInput(request.get(), yName.c_str(),
            OVMS_DATATYPE_U8, shapeY, 4)));
        ASSERT_TRUE(check(OVMS_InferenceRequestAddInput(request.get(), uvName.c_str(),
            OVMS_DATATYPE_U8, shapeUv, 4)));

        bench::DecodeStream decode;
        decode.open();
        int frames = 0;
        bool requestHasBoundData = false;
        bench::Clock::time_point start;
        bench::Clock::time_point stop;
        for (;;) {
            imp_tensor_t* rawTensor = nullptr;
            const imp_status_t readStatus = imp_video_read_frame(&rawTensor, decode.stream, 0);
            std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(rawTensor, imp_tensor_release);
            if (readStatus == IMP_ERROR_STREAM_END)
                break;
            ASSERT_EQ(readStatus, IMP_OK);
            ASSERT_NE(tensor, nullptr);

            uint32_t surface = 0;
            void* display = nullptr;
            int width = 0;
            int height = 0;
            ASSERT_EQ(imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height), IMP_OK);
            ASSERT_EQ(display, display_);
            ASSERT_EQ(width, benchmark_config::kModelWidth);
            ASSERT_EQ(height, benchmark_config::kModelHeight);
            const void* surfaceHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(surface));

            if (requestHasBoundData) {
                ASSERT_TRUE(check(OVMS_InferenceRequestInputRemoveData(request.get(), yName.c_str())));
                ASSERT_TRUE(check(OVMS_InferenceRequestInputRemoveData(request.get(), uvName.c_str())));
            }
            ASSERT_TRUE(check(OVMS_InferenceRequestInputSetData(request.get(), yName.c_str(), surfaceHandle,
                bytesY, OVMS_BUFFERTYPE_VASURFACE_Y, 1)));
            ASSERT_TRUE(check(OVMS_InferenceRequestInputSetData(request.get(), uvName.c_str(), surfaceHandle,
                bytesUv, OVMS_BUFFERTYPE_VASURFACE_UV, 1)));
            requestHasBoundData = true;

            OVMS_InferenceResponse* rawResponse = nullptr;
            ASSERT_TRUE(check(OVMS_Inference(server_, request.get(), &rawResponse)));
            std::unique_ptr<OVMS_InferenceResponse, decltype(&OVMS_InferenceResponseDelete)> response(
                rawResponse, OVMS_InferenceResponseDelete);
            const void* outputData = nullptr;
            size_t outputBytes = 0;
            uint32_t outputId = 0;
            OVMS_DataType outputType = static_cast<OVMS_DataType>(199);
            const int64_t* outputShape = nullptr;
            size_t outputDimensions = 0;
            OVMS_BufferType outputBufferType = OVMS_BUFFERTYPE_CPU;
            uint32_t outputDeviceId = 0;
            const char* outputName = nullptr;
            ASSERT_TRUE(check(OVMS_InferenceResponseOutput(response.get(), outputId, &outputName, &outputType,
                &outputShape, &outputDimensions, &outputData, &outputBytes, &outputBufferType, &outputDeviceId)));
            ASSERT_EQ(outputType, OVMS_DATATYPE_FP32);
            ASSERT_NE(outputData, nullptr);
            const float* values = static_cast<const float*>(outputData);
            int detections = 0;
            for (size_t resultIndex = 0; resultIndex + 7 <= outputBytes / sizeof(float); resultIndex += 7) {
                if (values[resultIndex + 2] >= 0.5f)
                    ++detections;
            }
            result.detections.push_back(detections);
            response.reset();
            tensor.reset();
            ++frames;
            if (frames == benchmark_config::kWarmupFrames)
                start = bench::Clock::now();
            if (frames == bench::kTotalFrames)
                stop = bench::Clock::now();
        }
        ASSERT_EQ(frames, bench::kTotalFrames) << "C-API path must consume all 250 frames and reach EOS";
        result.millisecondsPerFrame = bench::milliseconds(start, stop) / benchmark_config::kMeasuredFrames;
    }

    void runFairFullPathComparison() {
        std::vector<double> directSamples;
        std::vector<double> capiSamples;
        for (int sample = 0; sample < benchmark_config::kSampleCount; ++sample) {
            FairPathResult direct;
            FairPathResult capi;
            const bool directFirst = sample % 2 == 0;
            if (directFirst) {
                runFairDirectPath(direct);
                if (testing::Test::HasFatalFailure())
                    return;
                runFairCapiPath(capi);
            } else {
                runFairCapiPath(capi);
                if (testing::Test::HasFatalFailure())
                    return;
                runFairDirectPath(direct);
            }
            if (testing::Test::HasFatalFailure())
                return;
            ASSERT_EQ(direct.detections.size(), bench::kTotalFrames);
            ASSERT_EQ(capi.detections.size(), bench::kTotalFrames);
            for (size_t frame = 0; frame < bench::kTotalFrames; ++frame) {
                SCOPED_TRACE(frame);
                ASSERT_EQ(direct.detections[frame], capi.detections[frame]);
            }
            directSamples.push_back(direct.millisecondsPerFrame);
            capiSamples.push_back(capi.millisecondsPerFrame);
            std::cout << std::fixed << std::setprecision(6)
                      << "FAIR_CONFIG sample=" << sample + 1
                      << " direct_compile_before_decode=1 capi_model_before_decode=1"
                      << " request_count=1 performance_hint=" << benchmark_config::kPerformanceHint
                      << " num_streams=" << benchmark_config::kNumStreams
                      << " frame_window=completion_20_to_250 alternating_pass_order=" << (sample % 2 == 1)
                      << " detection_counts_equal=1 eos=1" << std::endl
                      << "FAIR_PATH case=direct_ov sample=" << sample + 1
                      << " ms_per_frame=" << direct.millisecondsPerFrame
                      << " fps=" << 1000.0 / direct.millisecondsPerFrame << std::endl
                      << "FAIR_PATH case=capi sample=" << sample + 1
                      << " ms_per_frame=" << capi.millisecondsPerFrame
                      << " fps=" << 1000.0 / capi.millisecondsPerFrame << std::endl;
        }
        bench::report("fair_full_path_direct_ov", directSamples);
        bench::report("fair_full_path_capi", capiSamples);
    }
#endif

    OVMS_Server* server_ = nullptr;
    std::filesystem::path directory_;
#ifdef _WIN32
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
#else
    int descriptor_ = -1;
    VADisplay display_ = nullptr;
    bool initialized_ = false;
#endif
};

TEST_F(MpiCapiBenchmark, OneCalculator) { run(false); }
TEST_F(MpiCapiBenchmark, TwoCalculators) { run(true); }
#ifndef _WIN32
TEST_F(MpiCapiBenchmark, PairedNativeSurface) { runNativeSurfaceReplay(true); }
TEST_F(MpiCapiBenchmark, CapiNativeSurfaceOnly) { runNativeSurfaceReplay(false); }
TEST_F(MpiCapiBenchmark, FairFullPathComparison) { runFairFullPathComparison(); }
#endif
}  // namespace