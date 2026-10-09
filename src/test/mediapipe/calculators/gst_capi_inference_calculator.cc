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
#pragma GCC diagnostic pop

#include "src/ovms.h"
#include "src/test/mediapipe/calculators/gst_video_frame_packet.hpp"

namespace mediapipe {

namespace {
constexpr char kFrameTag[] = "FRAME";
constexpr char kServableNameTag[] = "SERVABLE_NAME";
constexpr char kServableVersionTag[] = "SERVABLE_VERSION";
constexpr char kInputNameTag[] = "INPUT_NAME";
constexpr char kServerTag[] = "SERVER";
constexpr char kDetectionsTag[] = "DETECTIONS";

absl::Status checkCapiStatus(OVMS_Status* status, const char* operation) {
    if (!status)
        return absl::OkStatus();

    uint32_t code = 0;
    OVMS_Status* code_status = OVMS_StatusCode(status, &code);
    if (code_status)
        OVMS_StatusDelete(code_status);
    OVMS_StatusDelete(status);
    return absl::InternalError(std::string(operation) + " failed with OVMS status " +
                               std::to_string(code));
}
}  // namespace

class GstCapiInferenceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
        cc->InputSidePackets().Tag(kServableNameTag).Set<std::string>();
        cc->InputSidePackets().Tag(kServableVersionTag).Set<int>();
        cc->InputSidePackets().Tag(kInputNameTag).Set<std::string>();
        cc->InputSidePackets().Tag(kServerTag).Set<OVMS_Server*>();
        cc->Outputs().Tag(kDetectionsTag).Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        servable_name_ = cc->InputSidePackets().Tag(kServableNameTag).Get<std::string>();
        servable_version_ = cc->InputSidePackets().Tag(kServableVersionTag).Get<int>();
        input_name_ = cc->InputSidePackets().Tag(kInputNameTag).Get<std::string>();
        server_ = cc->InputSidePackets().Tag(kServerTag).Get<OVMS_Server*>();
        RET_CHECK(server_ != nullptr) << "missing started OVMS server";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        const auto& frame = cc->Inputs().Tag(kFrameTag).Get<GstVideoFramePacket>();
        RET_CHECK_GT(frame.width, 0);
        RET_CHECK_GT(frame.height, 0);
        RET_CHECK_EQ(frame.width % 2, 0) << "NV12 width must be even";
        RET_CHECK_EQ(frame.height % 2, 0) << "NV12 height must be even";

        // Pick the C-API buffer type and native handle from the tagged resource.
        void* surface_handle = nullptr;
        OVMS_BufferType buffer_type_y = OVMS_BUFFERTYPE_VASURFACE_Y;
        OVMS_BufferType buffer_type_uv = OVMS_BUFFERTYPE_VASURFACE_UV;
        if (const auto* va_frame = std::get_if<VaSurfaceFrame>(&frame.resource)) {
            RET_CHECK(va_frame->surface_id != 0) << "invalid VA surface id";
            surface_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(va_frame->surface_id));
            buffer_type_y = OVMS_BUFFERTYPE_VASURFACE_Y;
            buffer_type_uv = OVMS_BUFFERTYPE_VASURFACE_UV;
        } else if (const auto* d3d11_frame = std::get_if<D3D11Frame>(&frame.resource)) {
            RET_CHECK(d3d11_frame->texture_handle != 0) << "invalid D3D11 texture handle";
            // OpenVINO's D3D11 import addresses a whole ID3D11Texture2D (plane only);
            // the source must emit a non-array texture (subresource 0).
            RET_CHECK_EQ(d3d11_frame->subresource, 0u)
                << "D3D11 texture array slice not importable by OpenVINO; expected subresource 0";
            surface_handle = reinterpret_cast<void*>(d3d11_frame->texture_handle);
            buffer_type_y = OVMS_BUFFERTYPE_D3D11_TEXTURE_Y;
            buffer_type_uv = OVMS_BUFFERTYPE_D3D11_TEXTURE_UV;
        } else {
            RET_CHECK(false) << "unsupported frame resource for C-API inference";
        }

        OVMS_InferenceRequest* raw_request = nullptr;
        absl::Status status = checkCapiStatus(
            OVMS_InferenceRequestNew(&raw_request, server_, servable_name_.c_str(), servable_version_),
            "OVMS_InferenceRequestNew");
        if (!status.ok())
            return status;
        std::unique_ptr<OVMS_InferenceRequest, decltype(&OVMS_InferenceRequestDelete)> request(
            raw_request, OVMS_InferenceRequestDelete);

        const std::string input_name_y = input_name_ + "/y";
        const std::string input_name_uv = input_name_ + "/uv";
        const int64_t shape_y[] = {1, frame.height, frame.width, 1};
        const int64_t shape_uv[] = {1, frame.height / 2, frame.width / 2, 2};
        const size_t bytes_y = static_cast<size_t>(frame.width) * frame.height;
        const size_t bytes_uv = bytes_y / 2;

        status = checkCapiStatus(
            OVMS_InferenceRequestAddInput(request.get(), input_name_y.c_str(), OVMS_DATATYPE_U8,
                                          shape_y, 4),
            "OVMS_InferenceRequestAddInput(Y)");
        if (!status.ok())
            return status;
        status = checkCapiStatus(
            OVMS_InferenceRequestInputSetData(request.get(), input_name_y.c_str(), surface_handle,
                                              bytes_y, buffer_type_y, 1),
            "OVMS_InferenceRequestInputSetData(Y)");
        if (!status.ok())
            return status;
        status = checkCapiStatus(
            OVMS_InferenceRequestAddInput(request.get(), input_name_uv.c_str(), OVMS_DATATYPE_U8,
                                          shape_uv, 4),
            "OVMS_InferenceRequestAddInput(UV)");
        if (!status.ok())
            return status;
        status = checkCapiStatus(
            OVMS_InferenceRequestInputSetData(request.get(), input_name_uv.c_str(), surface_handle,
                                              bytes_uv, buffer_type_uv, 1),
            "OVMS_InferenceRequestInputSetData(UV)");
        if (!status.ok())
            return status;

        OVMS_InferenceResponse* raw_response = nullptr;
        status = checkCapiStatus(OVMS_Inference(server_, request.get(), &raw_response), "OVMS_Inference");
        if (!status.ok())
            return status;
        std::unique_ptr<OVMS_InferenceResponse, decltype(&OVMS_InferenceResponseDelete)> response(
            raw_response, OVMS_InferenceResponseDelete);

        const void* output_data = nullptr;
        size_t output_bytes = 0;
        uint32_t output_id = 0;
        OVMS_DataType output_type = static_cast<OVMS_DataType>(199);
        const int64_t* output_shape = nullptr;
        size_t output_dimensions = 0;
        OVMS_BufferType output_buffer_type = OVMS_BUFFERTYPE_CPU;
        uint32_t output_device_id = 0;
        const char* output_name = nullptr;
        status = checkCapiStatus(
            OVMS_InferenceResponseOutput(response.get(), output_id, &output_name, &output_type,
                                         &output_shape, &output_dimensions, &output_data,
                                         &output_bytes, &output_buffer_type, &output_device_id),
            "OVMS_InferenceResponseOutput");
        if (!status.ok())
            return status;
        RET_CHECK_EQ(output_type, OVMS_DATATYPE_FP32) << "expected FP32 detection output";
        RET_CHECK(output_data != nullptr);

        const auto* values = static_cast<const float*>(output_data);
        const size_t float_count = output_bytes / sizeof(float);
        int detections = 0;
        for (size_t i = 0; i + 7 <= float_count; i += 7) {
            if (values[i + 2] >= 0.5f)
                ++detections;
        }
        cc->Outputs()
            .Tag(kDetectionsTag)
            .AddPacket(MakePacket<int>(detections).At(cc->InputTimestamp()));
        return absl::OkStatus();
    }

 private:
    OVMS_Server* server_ = nullptr;
    std::string servable_name_;
    std::string input_name_;
    int servable_version_ = 1;
};

REGISTER_CALCULATOR(GstCapiInferenceCalculator);

}  // namespace mediapipe