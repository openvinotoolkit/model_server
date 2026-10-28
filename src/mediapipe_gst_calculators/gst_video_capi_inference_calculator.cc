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

#include <cstdint>
#include <memory>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/ovms.h"
#ifndef _WIN32
#include <va/va.h>
#include "src/mediapipe_gst_calculators/gst_capi_stage_timing.hpp"
#endif

namespace mediapipe {

namespace {
constexpr char kVideoPathTag[] = "VIDEO_PATH";
constexpr char kWidthTag[] = "WIDTH";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kMaxFramesTag[] = "MAX_FRAMES";
constexpr char kServableNameTag[] = "SERVABLE_NAME";
constexpr char kServableVersionTag[] = "SERVABLE_VERSION";
constexpr char kInputNameTag[] = "INPUT_NAME";
constexpr char kServerTag[] = "SERVER";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr float kFaceConfidenceThreshold = 0.5f;

absl::Status checkCapiStatus(OVMS_Status* status, const char* operation) {
    if (status == nullptr)
        return absl::OkStatus();
    uint32_t code = 0;
    OVMS_Status* code_status = OVMS_StatusCode(status, &code);
    if (code_status != nullptr)
        OVMS_StatusDelete(code_status);
    OVMS_StatusDelete(status);
    return absl::InternalError(std::string(operation) + " failed with OVMS status " +
                               std::to_string(code));
}
}  // namespace

class GstVideoCapiInferenceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->InputSidePackets().Tag(kVideoPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kWidthTag).Set<int>();
        cc->InputSidePackets().Tag(kHeightTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kMaxFramesTag))
            cc->InputSidePackets().Tag(kMaxFramesTag).Set<int>();
        cc->InputSidePackets().Tag(kServableNameTag).Set<std::string>();
        cc->InputSidePackets().Tag(kServableVersionTag).Set<int>();
        cc->InputSidePackets().Tag(kInputNameTag).Set<std::string>();
        cc->InputSidePackets().Tag(kServerTag).Set<OVMS_Server*>();
        cc->Outputs().Tag(kDetectionsTag).Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        server_ = cc->InputSidePackets().Tag(kServerTag).Get<OVMS_Server*>();
        RET_CHECK(server_ != nullptr) << "GstVideoCapiInferenceCalculator requires a started OVMS server";
        servable_name_ = cc->InputSidePackets().Tag(kServableNameTag).Get<std::string>();
        servable_version_ = cc->InputSidePackets().Tag(kServableVersionTag).Get<int>();
        input_name_ = cc->InputSidePackets().Tag(kInputNameTag).Get<std::string>();
        if (imp_context_create(&context_, nullptr) != IMP_OK)
            return absl::InternalError("GstVideoCapiInferenceCalculator: imp_context_create failed");
        imp_video_source_t* source = nullptr;
        if (imp_video_source_create(&source, IMP_SOURCE_FILE) != IMP_OK)
            return absl::InternalError("GstVideoCapiInferenceCalculator: imp_video_source_create failed");
        const std::string video_path = cc->InputSidePackets().Tag(kVideoPathTag).Get<std::string>();
        const int width = cc->InputSidePackets().Tag(kWidthTag).Get<int>();
        const int height = cc->InputSidePackets().Tag(kHeightTag).Get<int>();
        if (cc->InputSidePackets().HasTag(kMaxFramesTag))
            max_frames_ = cc->InputSidePackets().Tag(kMaxFramesTag).Get<int>();
        imp_video_source_set(source, "path", video_path.c_str());
        imp_video_source_set(source, "width", std::to_string(width).c_str());
        imp_video_source_set(source, "height", std::to_string(height).c_str());
        imp_video_decode_opts_t decode_options{};
    #ifdef _WIN32
        decode_options.use_d3d11_surface_memory = true;
    #else
        decode_options.use_va_surface_memory = true;
    #endif
        const imp_status_t status = imp_video_open(&stream_, source, context_, &decode_options);
        imp_video_source_destroy(source);
        if (status != IMP_OK)
            return absl::InternalError(std::string("GstVideoCapiInferenceCalculator: ") +
                                       imp_context_get_error(context_));
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        if (max_frames_ > 0 && frame_index_ >= max_frames_)
            return tool::StatusStop();
#ifndef _WIN32
        const bool e4Measured = e4Timing_.beginFrame();
        const auto e4ProcessStart = e4Timing_.mark();
        const auto readStart = e4Timing_.mark();
#endif
        imp_tensor_t* raw_tensor = nullptr;
        const imp_status_t read_status = imp_video_read_frame(&raw_tensor, stream_, 0);
#ifndef _WIN32
        const auto readStop = e4Timing_.mark();
        e4Timing_.add(GstCapiStageTiming::Stage::ReadPull, readStart, readStop, e4Measured);
#endif
        if (read_status == IMP_ERROR_STREAM_END)
            return tool::StatusStop();
        if (read_status != IMP_OK || raw_tensor == nullptr)
            return absl::InternalError("GstVideoCapiInferenceCalculator: frame read failed");
        std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw_tensor, imp_tensor_release);
        void* surface_handle = nullptr;
        OVMS_BufferType buffer_type_y;
        OVMS_BufferType buffer_type_uv;
        int width = 0;
        int height = 0;
    #ifdef _WIN32
        void* decode_device = nullptr;
        uint32_t subresource = 0;
        if (imp_tensor_get_d3d11_texture(tensor.get(), &surface_handle, &decode_device, &subresource,
                                         &width, &height) != IMP_OK)
            return absl::InternalError("GstVideoCapiInferenceCalculator: expected D3D11 texture");
        RET_CHECK_EQ(subresource, 0u) << "D3D11 texture array slice is not importable by OpenVINO";
        buffer_type_y = OVMS_BUFFERTYPE_D3D11_TEXTURE_Y;
        buffer_type_uv = OVMS_BUFFERTYPE_D3D11_TEXTURE_UV;
    #else
        uint32_t surface = 0;
        void* display = nullptr;
    #ifndef _WIN32
        const auto surfaceStart = e4Timing_.mark();
    #endif
        if (imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height) != IMP_OK)
            return absl::InternalError("GstVideoCapiInferenceCalculator: expected VA-API surface");
    #ifndef _WIN32
        e4Timing_.add(GstCapiStageTiming::Stage::SurfaceExtract, surfaceStart, e4Timing_.mark(), e4Measured);
        if (e4Timing_.surfaceSyncEnabled()) {
            const auto syncStart = e4Timing_.mark();
            const VAStatus syncStatus = vaSyncSurface(display, surface);
            const auto syncStop = e4Timing_.mark();
            e4Timing_.add(GstCapiStageTiming::Stage::SurfaceSync, syncStart, syncStop, e4Measured);
            if (syncStatus != VA_STATUS_SUCCESS)
                return absl::InternalError("GstVideoCapiInferenceCalculator: vaSyncSurface failed");
        }
    #endif
        surface_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(surface));
        buffer_type_y = OVMS_BUFFERTYPE_VASURFACE_Y;
        buffer_type_uv = OVMS_BUFFERTYPE_VASURFACE_UV;
    #endif

        const std::string y_name = input_name_ + "/y";
        const std::string uv_name = input_name_ + "/uv";
        const int64_t shape_y[] = {1, height, width, 1};
        const int64_t shape_uv[] = {1, height / 2, width / 2, 2};
        const size_t bytes_y = static_cast<size_t>(width) * height;
        const size_t bytes_uv = bytes_y / 2;
        absl::Status status;
        if (request_ == nullptr) {
            status = checkCapiStatus(OVMS_InferenceRequestNew(
                &request_, server_, servable_name_.c_str(), servable_version_), "OVMS_InferenceRequestNew");
            if (!status.ok())
                return status;
            status = checkCapiStatus(OVMS_InferenceRequestAddInput(request_, y_name.c_str(),
                OVMS_DATATYPE_U8, shape_y, 4), "OVMS_InferenceRequestAddInput(Y)");
            if (!status.ok())
                return status;
            status = checkCapiStatus(OVMS_InferenceRequestAddInput(request_, uv_name.c_str(),
                OVMS_DATATYPE_U8, shape_uv, 4), "OVMS_InferenceRequestAddInput(UV)");
            if (!status.ok())
                return status;
            request_width_ = width;
            request_height_ = height;
        }
        RET_CHECK_EQ(width, request_width_);
        RET_CHECK_EQ(height, request_height_);
    #ifndef _WIN32
        const auto removeStart = e4Timing_.mark();
    #endif
        if (request_has_bound_data_) {
            status = checkCapiStatus(OVMS_InferenceRequestInputRemoveData(request_, y_name.c_str()),
                "OVMS_InferenceRequestInputRemoveData(Y)");
            if (!status.ok())
                return status;
            status = checkCapiStatus(OVMS_InferenceRequestInputRemoveData(request_, uv_name.c_str()),
                "OVMS_InferenceRequestInputRemoveData(UV)");
            if (!status.ok())
                return status;
            request_has_bound_data_ = false;
        }
        #ifndef _WIN32
            e4Timing_.add(GstCapiStageTiming::Stage::RemoveData, removeStart, e4Timing_.mark(), e4Measured);
            const auto setDataStart = e4Timing_.mark();
        #endif
        status = checkCapiStatus(OVMS_InferenceRequestInputSetData(request_, y_name.c_str(),
            surface_handle, bytes_y, buffer_type_y, 1), "OVMS_InferenceRequestInputSetData(Y)");
        if (!status.ok())
            return status;
        status = checkCapiStatus(OVMS_InferenceRequestInputSetData(request_, uv_name.c_str(),
            surface_handle, bytes_uv, buffer_type_uv, 1), "OVMS_InferenceRequestInputSetData(UV)");
        if (!status.ok())
            return status;
        request_has_bound_data_ = true;
        #ifndef _WIN32
            const auto inferenceStart = e4Timing_.mark();
        #endif
        OVMS_InferenceResponse* raw_response = nullptr;
        status = checkCapiStatus(OVMS_Inference(server_, request_, &raw_response), "OVMS_Inference");
        if (!status.ok())
            return status;
        #ifndef _WIN32
            const auto inferenceStop = e4Timing_.mark();
            e4Timing_.add(GstCapiStageTiming::Stage::SetData, setDataStart, inferenceStart, e4Measured);
            e4Timing_.add(GstCapiStageTiming::Stage::Inference, inferenceStart, inferenceStop, e4Measured);
        #endif
        std::unique_ptr<OVMS_InferenceResponse, decltype(&OVMS_InferenceResponseDelete)> response(
            raw_response, OVMS_InferenceResponseDelete);
        #ifndef _WIN32
            const auto responseAccessStart = e4Timing_.mark();
        #endif
        const void* output = nullptr;
        size_t output_bytes = 0;
        uint32_t output_id = 0;
        OVMS_DataType output_type = static_cast<OVMS_DataType>(199);
        const int64_t* output_shape = nullptr;
        size_t output_dimensions = 0;
        OVMS_BufferType output_buffer_type = OVMS_BUFFERTYPE_CPU;
        uint32_t output_device_id = 0;
        const char* output_name = nullptr;
        status = checkCapiStatus(OVMS_InferenceResponseOutput(response.get(), output_id, &output_name,
            &output_type, &output_shape, &output_dimensions, &output, &output_bytes,
            &output_buffer_type, &output_device_id), "OVMS_InferenceResponseOutput");
        if (!status.ok())
            return status;
        RET_CHECK_EQ(output_type, OVMS_DATATYPE_FP32);
        RET_CHECK(output != nullptr);
        const float* values = static_cast<const float*>(output);
        int detections = 0;
        for (size_t result = 0; result + 7 <= output_bytes / sizeof(float); result += 7) {
            if (values[result + 2] >= kFaceConfidenceThreshold)
                ++detections;
        }
    #ifndef _WIN32
        e4Timing_.add(GstCapiStageTiming::Stage::ResponseAccess, responseAccessStart, e4Timing_.mark(), e4Measured);
        if (e4Measured) {
            const auto responseDeleteStart = e4Timing_.mark();
            response.reset();
            e4Timing_.add(GstCapiStageTiming::Stage::ResponseDelete, responseDeleteStart, e4Timing_.mark(), true);
        }
    #endif
        cc->Outputs().Tag(kDetectionsTag).AddPacket(
            MakePacket<int>(detections).At(Timestamp(frame_index_++)));
    #ifndef _WIN32
        e4Timing_.finishFrame(e4ProcessStart, e4Measured);
    #endif
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext*) override {
    #ifndef _WIN32
        e4Timing_.report("GstVideoCapiInferenceCalculator");
    #endif
        if (request_ != nullptr) {
            OVMS_InferenceRequestDelete(request_);
            request_ = nullptr;
        }
        if (stream_ != nullptr)
            imp_video_close(stream_);
        if (context_ != nullptr)
            imp_context_destroy(context_);
        return absl::OkStatus();
    }

 private:
    imp_context_t* context_ = nullptr;
    imp_video_stream_t* stream_ = nullptr;
    OVMS_Server* server_ = nullptr;
    OVMS_InferenceRequest* request_ = nullptr;
    std::string servable_name_;
    std::string input_name_;
    int servable_version_ = 1;
    int max_frames_ = 0;
    int64_t frame_index_ = 0;
    int request_width_ = 0;
    int request_height_ = 0;
    bool request_has_bound_data_ = false;
#ifndef _WIN32
    GstCapiStageTiming e4Timing_;
#endif
};

REGISTER_CALCULATOR(GstVideoCapiInferenceCalculator);

}  // namespace mediapipe