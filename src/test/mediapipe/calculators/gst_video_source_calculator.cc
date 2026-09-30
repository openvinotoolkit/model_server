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
 * GstVideoSourceCalculator — P2 split-out video source (GPU VA path).
 *
 * A MediaPipe source calculator: decodes a video with GStreamer/VA and emits one
 * GstVideoFramePacket per frame, each owning its own VA surface (per-packet
 * ownership from P2). No inference here — that is the downstream calculator's job
 * (P3). The VA context is created internally in Open() from the first frame's
 * shared display and used to import surfaces as ov::RemoteTensor pairs.
 *
 * Side packets:
 *   VIDEO_PATH : std::string — path to input video file.
 *   WIDTH      : int         — decode output width  (size to model input).
 *   HEIGHT     : int         — decode output height (size to model input).
 *
 * Output stream:
 *   FRAME      : GstVideoFramePacket — one per decoded frame, timestamped by index.
 */

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/test/mediapipe/calculators/gst_video_frame_packet.hpp"

namespace mediapipe {

namespace {
constexpr char kVideoPathTag[] = "VIDEO_PATH";
constexpr char kWidthTag[] = "WIDTH";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kFrameTag[] = "FRAME";
constexpr char kVaContextTag[] = "VA_CONTEXT";
}  // namespace

class GstVideoSourceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->InputSidePackets().Tag(kVideoPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kWidthTag).Set<int>();
        cc->InputSidePackets().Tag(kHeightTag).Set<int>();
        cc->Outputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
        // Shared VA context so a downstream inference calculator can compile its
        // model on the exact same context the surfaces were imported on.
        if (cc->OutputSidePackets().HasTag(kVaContextTag)) {
            cc->OutputSidePackets().Tag(kVaContextTag).Set<ov::RemoteContext>();
        }
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        video_path_ = cc->InputSidePackets().Tag(kVideoPathTag).Get<std::string>();
        width_ = cc->InputSidePackets().Tag(kWidthTag).Get<int>();
        height_ = cc->InputSidePackets().Tag(kHeightTag).Get<int>();

        if (!imp_video_va_available())
            return absl::UnavailableError("GstVideoSourceCalculator: VA/GPU not available");

        if (imp_context_create(&ctx_, nullptr) != IMP_OK)
            return absl::InternalError("imp_context_create failed");

        imp_video_source_t* src = nullptr;
        imp_video_source_create(&src, IMP_SOURCE_FILE);
        imp_video_source_set(src, "path", video_path_.c_str());
        imp_video_source_set(src, "width", std::to_string(width_).c_str());
        imp_video_source_set(src, "height", std::to_string(height_).c_str());

        imp_video_decode_opts_t vopts{};
        vopts.use_va_surface_memory = true;
        imp_status_t st = imp_video_open(&stream_, src, ctx_, &vopts);
        imp_video_source_destroy(src);
        if (st != IMP_OK)
            return absl::InternalError(std::string("imp_video_open failed: ") +
                                       imp_context_get_error(ctx_));

        // Build the VA context from the first frame's shared display (P2/D2:
        // context created internally by the source). Keep the first frame to
        // emit it on the first Process() call.
        st = imp_video_read_frame(&pending_, stream_, 0);
        if (st != IMP_OK || !pending_)
            return absl::InternalError("first frame read failed");

        uint32_t surf = 0;
        void* disp = nullptr;
        int fw = 0, fh = 0;
        imp_tensor_get_va_surface(pending_, &surf, &disp, &fw, &fh);
        if (!disp)
            return absl::InternalError("no VADisplay from first frame");
        try {
            va_ctx_ = ov::intel_gpu::ocl::VAContext(core_, disp);
        } catch (const std::exception& e) {
            return absl::InternalError(std::string("VAContext creation failed: ") + e.what());
        }

        // Publish the context so the inference calculator compiles on the same one.
        if (cc->OutputSidePackets().HasTag(kVaContextTag)) {
            cc->OutputSidePackets()
                .Tag(kVaContextTag)
                .Set(MakePacket<ov::RemoteContext>(*va_ctx_));
        }
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        imp_tensor_t* tensor = pending_;
        pending_ = nullptr;
        if (!tensor) {
            imp_status_t st = imp_video_read_frame(&tensor, stream_, 0);
            if (st == IMP_ERROR_STREAM_END)
                return tool::StatusStop();
            if (st != IMP_OK || !tensor)
                return tool::StatusStop();
        }

        uint32_t surface_id = 0;
        void* disp = nullptr;
        int fw = 0, fh = 0;
        if (imp_tensor_get_va_surface(tensor, &surface_id, &disp, &fw, &fh) != IMP_OK) {
            imp_tensor_release(tensor);
            return absl::InternalError("tensor is not a VA surface");
        }

        auto pkt = std::make_unique<GstVideoFramePacket>();
        try {
            auto nv12 = va_ctx_->create_tensor_nv12(static_cast<size_t>(fh),
                                                    static_cast<size_t>(fw),
                                                    surface_id);
            pkt->planes = {nv12.first, nv12.second};
        } catch (const std::exception& e) {
            imp_tensor_release(tensor);
            return absl::InternalError(std::string("create_tensor_nv12 failed: ") + e.what());
        }
        pkt->va_surface_id = surface_id;
        // Transfer surface ownership into the packet; released when the packet dies.
        pkt->owner = std::shared_ptr<imp_tensor_t>(tensor, imp_tensor_release);

        cc->Outputs()
            .Tag(kFrameTag)
            .Add(pkt.release(), Timestamp(frame_index_++));
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) override {
        if (pending_) {
            imp_tensor_release(pending_);
            pending_ = nullptr;
        }
        if (stream_) {
            imp_video_close(stream_);
            stream_ = nullptr;
        }
        if (ctx_) {
            imp_context_destroy(ctx_);
            ctx_ = nullptr;
        }
        return absl::OkStatus();
    }

 private:
    std::string video_path_;
    int width_ = 0;
    int height_ = 0;
    int64_t frame_index_ = 0;

    imp_context_t* ctx_ = nullptr;
    imp_video_stream_t* stream_ = nullptr;
    imp_tensor_t* pending_ = nullptr;  // first frame, emitted on first Process()

    ov::Core core_;
    std::optional<ov::intel_gpu::ocl::VAContext> va_ctx_;
};

REGISTER_CALCULATOR(GstVideoSourceCalculator);

}  // namespace mediapipe
