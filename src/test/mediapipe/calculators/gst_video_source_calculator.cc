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
 * GstVideoSourceCalculator — split-out video source (GPU VA path).
 *
 * A MediaPipe source calculator: decodes a video with GStreamer/VA and emits one
 * GstVideoFramePacket per frame, each owning its native frame resource. No
 * inference here — that is the downstream calculator's job. The packet carries
 * a backend-specific resource descriptor, not an OpenVINO tensor or context.
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
#include <string>

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
constexpr int64_t kMaxFrames = 4;
}  // namespace

class GstVideoSourceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->InputSidePackets().Tag(kVideoPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kWidthTag).Set<int>();
        cc->InputSidePackets().Tag(kHeightTag).Set<int>();
        cc->Outputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        video_path_ = cc->InputSidePackets().Tag(kVideoPathTag).Get<std::string>();
        width_ = cc->InputSidePackets().Tag(kWidthTag).Get<int>();
        height_ = cc->InputSidePackets().Tag(kHeightTag).Get<int>();

        // Pick the native GPU surface path available on this platform: D3D11 on
        // Windows, VA on Linux. Both emit a GstVideoFramePacket; only the tagged
        // resource variant differs.
        use_d3d11_ = imp_video_d3d11_available();
        if (!use_d3d11_ && !imp_video_va_available())
            return absl::UnavailableError("GstVideoSourceCalculator: no GPU surface path (VA/D3D11) available");

        if (imp_context_create(&ctx_, nullptr) != IMP_OK)
            return absl::InternalError("imp_context_create failed");

        imp_video_source_t* src = nullptr;
        imp_video_source_create(&src, IMP_SOURCE_FILE);
        imp_video_source_set(src, "path", video_path_.c_str());
        imp_video_source_set(src, "width", std::to_string(width_).c_str());
        imp_video_source_set(src, "height", std::to_string(height_).c_str());

        imp_video_decode_opts_t vopts{};
        vopts.use_va_surface_memory = !use_d3d11_;
        vopts.use_d3d11_surface_memory = use_d3d11_;
        imp_status_t st = imp_video_open(&stream_, src, ctx_, &vopts);
        imp_video_source_destroy(src);
        if (st != IMP_OK)
            return absl::InternalError(std::string("imp_video_open failed: ") +
                                       imp_context_get_error(ctx_));

        // Keep the first frame to emit it on the first Process() call.
        st = imp_video_read_frame(&pending_, stream_, 0);
        if (st != IMP_OK || !pending_)
            return absl::InternalError("first frame read failed");

        const imp_tensor_memory_type_t expected =
            use_d3d11_ ? IMP_MEM_D3D11_SURFACE : IMP_MEM_VA_SURFACE;
        if (imp_tensor_get_memory_type(pending_) != expected)
            return absl::InternalError("expected a GPU surface from GStreamer");
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        if (frame_index_ >= kMaxFrames)
            return tool::StatusStop();
        imp_tensor_t* tensor = pending_;
        pending_ = nullptr;
        if (!tensor) {
            imp_status_t st = imp_video_read_frame(&tensor, stream_, 0);
            if (st == IMP_ERROR_STREAM_END)
                return tool::StatusStop();
            if (st != IMP_OK || !tensor)
                return tool::StatusStop();
        }

        auto pkt = std::make_unique<GstVideoFramePacket>();
        if (use_d3d11_) {
            void* texture = nullptr;
            void* device = nullptr;
            uint32_t subresource = 0;
            int fw = 0, fh = 0;
            if (imp_tensor_get_d3d11_texture(tensor, &texture, &device, &subresource, &fw, &fh) != IMP_OK) {
                imp_tensor_release(tensor);
                return absl::InternalError("tensor is not a D3D11 surface");
            }
            pkt->resource = D3D11Frame{reinterpret_cast<uintptr_t>(texture), subresource};
            pkt->width = fw;
            pkt->height = fh;
        } else {
            uint32_t surface_id = 0;
            void* disp = nullptr;
            int fw = 0, fh = 0;
            if (imp_tensor_get_va_surface(tensor, &surface_id, &disp, &fw, &fh) != IMP_OK) {
                imp_tensor_release(tensor);
                return absl::InternalError("tensor is not a VA surface");
            }
            pkt->resource = VaSurfaceFrame{surface_id};
            pkt->width = fw;
            pkt->height = fh;
        }
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
    bool use_d3d11_ = false;

    imp_context_t* ctx_ = nullptr;
    imp_video_stream_t* stream_ = nullptr;
    imp_tensor_t* pending_ = nullptr;  // first frame, emitted on first Process()

};

REGISTER_CALCULATOR(GstVideoSourceCalculator);

}  // namespace mediapipe
