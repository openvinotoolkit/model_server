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

/**
 * GstNv12InferenceCalculator — P3 in-tree inference calculator (GPU VA path).
 *
 * Consumes GstVideoFramePacket frames from GstVideoSourceCalculator and runs
 * inference, binding the NV12 two planes (Y, UV) to one model input. The model
 * is compiled on the SHARED VA context published by the source (input side
 * packet VA_CONTEXT), so the imported surfaces match the compiled model's
 * context — zero-copy, no host round trip. Mismatched contexts are rejected
 * (error) rather than silently copied to CPU.
 *
 * This is a temporary in-tree stand-in for the external OpenVINOInferenceCalculator
 * (which today has no remote-tensor path). To be reconciled later.
 *
 * Input stream:
 *   FRAME       : GstVideoFramePacket — decoded frame with [Y, UV] RemoteTensors.
 * Input side packets:
 *   MODEL_PATH  : std::string        — model xml (NV12-preprocessable input).
 *   VA_CONTEXT  : ov::RemoteContext  — shared context from the source.
 * Outputs (each optional):
 *   DETECTIONS   : int        — count above threshold for this frame.
 *   INFER_OUTPUT : ov::Tensor — host copy of the raw model output.
 */

#include <cstring>
#include <string>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/test/mediapipe/calculators/gst_video_frame_packet.hpp"

namespace mediapipe {

namespace {
constexpr char kFrameTag[] = "FRAME";
constexpr char kModelPathTag[] = "MODEL_PATH";
constexpr char kVaContextTag[] = "VA_CONTEXT";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kInferOutputTag[] = "INFER_OUTPUT";
constexpr float kDetectionThreshold = 0.5f;
}  // namespace

class GstNv12InferenceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
        cc->InputSidePackets().Tag(kModelPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kVaContextTag).Set<ov::RemoteContext>();
        if (cc->Outputs().HasTag(kDetectionsTag))
            cc->Outputs().Tag(kDetectionsTag).Set<int>();
        if (cc->Outputs().HasTag(kInferOutputTag))
            cc->Outputs().Tag(kInferOutputTag).Set<ov::Tensor>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        const std::string model_path =
            cc->InputSidePackets().Tag(kModelPathTag).Get<std::string>();
        ov::RemoteContext context =
            cc->InputSidePackets().Tag(kVaContextTag).Get<ov::RemoteContext>();

        try {
            auto model = core_.read_model(model_path);
            ov::preprocess::PrePostProcessor ppp(model);
            ppp.input()
                .tensor()
                .set_element_type(ov::element::u8)
                .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                .set_memory_type(ov::intel_gpu::memory_type::surface);
            ppp.input()
                .preprocess()
                .convert_color(ov::preprocess::ColorFormat::BGR)
                .convert_element_type(ov::element::f32);
            ppp.input().model().set_layout("NCHW");
            compiled_ = core_.compile_model(ppp.build(), context);
        } catch (const std::exception& e) {
            return absl::InternalError(std::string("model compile on shared context failed: ") +
                                       e.what());
        }
        req_ = compiled_.create_infer_request();
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        const auto& frame = cc->Inputs().Tag(kFrameTag).Get<GstVideoFramePacket>();
        RET_CHECK_EQ(frame.planes.size(), 2u) << "NV12 requires [Y, UV] planes";

        try {
            req_.set_input_tensor(0, frame.planes[0]);
            req_.set_input_tensor(1, frame.planes[1]);
            req_.infer();
        } catch (const std::exception& e) {
            // Mismatched context / unsupported surface — reject, do not copy to CPU.
            return absl::InternalError(std::string("remote inference failed (context mismatch?): ") +
                                       e.what());
        }

        auto out = req_.get_output_tensor(0);  // [1,1,N,7]
        if (cc->Outputs().HasTag(kDetectionsTag)) {
            const float* det = out.data<const float>();
            size_t num_dets = out.get_shape()[2];
            int frame_dets = 0;
            for (size_t i = 0; i < num_dets; i++)
                if (det[i * 7 + 2] > kDetectionThreshold)
                    frame_dets++;
            cc->Outputs()
                .Tag(kDetectionsTag)
                .AddPacket(MakePacket<int>(frame_dets).At(cc->InputTimestamp()));
        }
        if (cc->Outputs().HasTag(kInferOutputTag)) {
            ov::Tensor host(out.get_element_type(), out.get_shape());
            std::memcpy(host.data(), out.data(), out.get_byte_size());
            cc->Outputs()
                .Tag(kInferOutputTag)
                .AddPacket(MakePacket<ov::Tensor>(std::move(host)).At(cc->InputTimestamp()));
        }
        return absl::OkStatus();
    }

 private:
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest req_;
};

REGISTER_CALCULATOR(GstNv12InferenceCalculator);

}  // namespace mediapipe
