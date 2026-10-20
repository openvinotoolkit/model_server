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
 * Two output modes, selected by which output tags the graph wires:
 *   FRAME       — single model-res GPU surface stream (legacy P6a inference graph).
 *   FRAME_INFER — same model-res GPU surface, for the overlay pipeline's
 *                 zero-copy inference branch.
 *   FRAME_FULL  — full-res frame for the overlay/encode branch. By default a
 *                 host NV12 frame (IMP_MEM_SYSTEM) for a CPU plane overlay. If
 *                 the FULL_VA_SURFACE side packet is true, it is instead a GPU
 *                 VA surface so the overlay/encode branch stays zero-copy
 *                 (P6a.2): the overlay draws by mapping the surface in place.
 *
 * Because an imp_video_stream carries a single surface-memory mode for all its
 * branches, the dual-resolution / dual-memory contract (VA surface + host NV12)
 * is served by opening two streams on the same file: one VA-surface stream at
 * model res, one host-NV12 stream at full res. Both are read in lockstep and
 * emitted with identical Timestamp(frame_index) so the overlay can join them.
 *
 * Side packets:
 *   VIDEO_PATH      : std::string — legacy path to input video file.
 *   SOURCE_MODE     : std::string (opt) — "file" (default), "camera", or "rtsp".
 *   SOURCE_URI      : std::string (opt) — file path or RTSP URI; overrides VIDEO_PATH.
 *   CAMERA_DEVICE   : std::string (opt) — camera identifier for SOURCE_MODE=camera
 *                     (device index, device path, or human-readable device name).
 *   WIDTH           : int         — model-res decode width  (inference surface).
 *   HEIGHT          : int         — model-res decode height (inference surface).
 *   FULL_WIDTH      : int (opt)   — full-res decode width  (overlay frame; 0 = source res).
 *   FULL_HEIGHT     : int (opt)   — full-res decode height (overlay frame; 0 = source res).
 *   FULL_VA_SURFACE : bool (opt)  — emit FRAME_FULL as a GPU VA surface (zero-copy overlay).
 *   MAX_FRAMES      : int (opt)   — stop after N frames (0/absent = whole stream).
 */

#include <cstdint>
#include <memory>
#include <string>

#include <openvino/openvino.hpp>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/mediapipe_gst_calculators/gst_video_frame_packet.hpp"

namespace mediapipe {

namespace {
constexpr char kVideoPathTag[] = "VIDEO_PATH";
constexpr char kSourceModeTag[] = "SOURCE_MODE";
constexpr char kSourceUriTag[] = "SOURCE_URI";
constexpr char kCameraDeviceTag[] = "CAMERA_DEVICE";
constexpr char kWidthTag[] = "WIDTH";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kFullWidthTag[] = "FULL_WIDTH";
constexpr char kFullHeightTag[] = "FULL_HEIGHT";
constexpr char kFullVaSurfaceTag[] = "FULL_VA_SURFACE";
constexpr char kMaxFramesTag[] = "MAX_FRAMES";
constexpr char kFrameTag[] = "FRAME";
constexpr char kFrameInferTag[] = "FRAME_INFER";
constexpr char kFrameFullTag[] = "FRAME_FULL";
constexpr char kCompletionTag[] = "COMPLETION";

void* getServerOwnedVaDisplay() {
#ifndef _WIN32
    void* mainProcess = dlopen(nullptr, RTLD_NOW);
    if (mainProcess == nullptr)
        return nullptr;
    using GetVaDisplay = void* (*)();
    const auto getVaDisplay = reinterpret_cast<GetVaDisplay>(
        dlsym(mainProcess, "ovms_get_global_va_display"));
    void* vaDisplay = getVaDisplay ? getVaDisplay() : nullptr;
    dlclose(mainProcess);
    return vaDisplay;
#else
    return nullptr;
#endif
}
}  // namespace

class GstVideoSourceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        if (cc->InputSidePackets().HasTag(kVideoPathTag))
            cc->InputSidePackets().Tag(kVideoPathTag).Set<std::string>();
        if (cc->InputSidePackets().HasTag(kSourceModeTag))
            cc->InputSidePackets().Tag(kSourceModeTag).Set<std::string>();
        if (cc->InputSidePackets().HasTag(kSourceUriTag))
            cc->InputSidePackets().Tag(kSourceUriTag).Set<std::string>();
        if (cc->InputSidePackets().HasTag(kCameraDeviceTag))
            cc->InputSidePackets().Tag(kCameraDeviceTag).Set<std::string>();
        RET_CHECK(cc->InputSidePackets().HasTag(kVideoPathTag) ||
                  cc->InputSidePackets().HasTag(kSourceUriTag) ||
                  cc->InputSidePackets().HasTag(kSourceModeTag))
            << "GstVideoSourceCalculator needs VIDEO_PATH, SOURCE_URI, or SOURCE_MODE";
        cc->InputSidePackets().Tag(kWidthTag).Set<int>();
        cc->InputSidePackets().Tag(kHeightTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kFullWidthTag))
            cc->InputSidePackets().Tag(kFullWidthTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kFullHeightTag))
            cc->InputSidePackets().Tag(kFullHeightTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kFullVaSurfaceTag))
            cc->InputSidePackets().Tag(kFullVaSurfaceTag).Set<bool>();
        if (cc->InputSidePackets().HasTag(kMaxFramesTag))
            cc->InputSidePackets().Tag(kMaxFramesTag).SetOneOf<int, int64_t>();
        if (cc->Outputs().HasTag(kFrameTag))
            cc->Outputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
        if (cc->Outputs().HasTag(kFrameInferTag))
            cc->Outputs().Tag(kFrameInferTag).Set<GstVideoFramePacket>();
        if (cc->Outputs().HasTag(kFrameFullTag))
            cc->Outputs().Tag(kFrameFullTag).Set<GstVideoFramePacket>();
        if (cc->Outputs().HasTag(kCompletionTag))
            cc->Outputs().Tag(kCompletionTag).Set<ov::Tensor>();
        RET_CHECK(cc->Outputs().HasTag(kFrameTag) || cc->Outputs().HasTag(kFrameInferTag))
            << "GstVideoSourceCalculator needs a FRAME or FRAME_INFER output";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        if (cc->InputSidePackets().HasTag(kVideoPathTag))
            source_uri_ = cc->InputSidePackets().Tag(kVideoPathTag).Get<std::string>();
        if (cc->InputSidePackets().HasTag(kSourceUriTag))
            source_uri_ = cc->InputSidePackets().Tag(kSourceUriTag).Get<std::string>();
        if (cc->InputSidePackets().HasTag(kCameraDeviceTag))
            camera_device_ = cc->InputSidePackets().Tag(kCameraDeviceTag).Get<std::string>();
        if (cc->InputSidePackets().HasTag(kSourceModeTag)) {
            std::string mode = cc->InputSidePackets().Tag(kSourceModeTag).Get<std::string>();
            std::transform(mode.begin(), mode.end(), mode.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (mode == "file") {
                source_type_ = IMP_SOURCE_FILE;
            } else if (mode == "camera") {
                source_type_ = IMP_SOURCE_CAMERA;
            } else if (mode == "rtsp") {
                source_type_ = IMP_SOURCE_URL;
            } else {
                return absl::InvalidArgumentError("GstVideoSourceCalculator: SOURCE_MODE must be 'file', 'camera', or 'rtsp'");
            }
        }
        if (source_type_ != IMP_SOURCE_CAMERA && source_uri_.empty())
            return absl::InvalidArgumentError("GstVideoSourceCalculator: empty source URI");
        width_ = cc->InputSidePackets().Tag(kWidthTag).Get<int>();
        height_ = cc->InputSidePackets().Tag(kHeightTag).Get<int>();
        if (cc->InputSidePackets().HasTag(kFullWidthTag))
            full_width_ = cc->InputSidePackets().Tag(kFullWidthTag).Get<int>();
        if (cc->InputSidePackets().HasTag(kFullHeightTag))
            full_height_ = cc->InputSidePackets().Tag(kFullHeightTag).Get<int>();
        if (cc->InputSidePackets().HasTag(kFullVaSurfaceTag))
            full_va_surface_ = cc->InputSidePackets().Tag(kFullVaSurfaceTag).Get<bool>();
        if (cc->InputSidePackets().HasTag(kMaxFramesTag)) {
            const auto& limit = cc->InputSidePackets().Tag(kMaxFramesTag);
            max_frames_ = limit.ValidateAsType<int64_t>().ok() ? limit.Get<int64_t>() : limit.Get<int>();
        }

        emit_frame_ = cc->Outputs().HasTag(kFrameTag);
        emit_infer_ = cc->Outputs().HasTag(kFrameInferTag);
        emit_full_ = cc->Outputs().HasTag(kFrameFullTag);

        // Pick the native GPU surface path available on this platform: D3D11 on
        // Windows, VA on Linux. Both emit a GstVideoFramePacket; only the tagged
        // resource variant differs.
        use_d3d11_ = imp_video_d3d11_available();
        if (!use_d3d11_ && !imp_video_va_available())
            return absl::UnavailableError("GstVideoSourceCalculator: no GPU surface path (VA/D3D11) available");
        if (!use_d3d11_) {
            void* vaDisplay = getServerOwnedVaDisplay();
            if (vaDisplay != nullptr)
                imp_video_set_va_display(vaDisplay);
        }

#ifdef _WIN32
        if (use_d3d11_ && source_type_ == IMP_SOURCE_CAMERA && emit_full_) {
            imp_video_branch_t branches[2]{};
            branches[0].width = width_;
            branches[0].height = height_;
            branches[0].name = "inference";
            branches[1].width = full_width_;
            branches[1].height = full_height_;
            branches[1].name = "overlay";
            branches[1].use_system_memory = !full_va_surface_;
            absl::Status status = openStream(&infer_ctx_, &infer_stream_, full_width_, full_height_,
                true, &pending_infer_, IMP_MEM_D3D11_SURFACE, branches, 2);
            if (!status.ok())
                return status;
            shared_camera_stream_ = true;
            if (imp_video_read_frame(&pending_full_, infer_stream_, 1) != IMP_OK || !pending_full_)
                return absl::InternalError("first camera overlay frame read failed");
            return absl::OkStatus();
        }
#endif
        if (emit_frame_ || emit_infer_) {
            const imp_tensor_memory_type_t expected =
                use_d3d11_ ? IMP_MEM_D3D11_SURFACE : IMP_MEM_VA_SURFACE;
            absl::Status st = openStream(&infer_ctx_, &infer_stream_, width_, height_,
                                         /*va_surface=*/true, &pending_infer_, expected);
            if (!st.ok())
                return st;
        }
        if (emit_full_) {
            // Overlay/encode branch. VA surface (zero-copy, P6a.2) if requested,
            // otherwise an explicit host NV12 (GPU->CPU) frame.
            const imp_tensor_memory_type_t expected_full =
                full_va_surface_ ? (use_d3d11_ ? IMP_MEM_D3D11_SURFACE : IMP_MEM_VA_SURFACE)
                                 : IMP_MEM_SYSTEM;
            absl::Status st = openStream(&full_ctx_, &full_stream_, full_width_, full_height_,
                                         /*va_surface=*/full_va_surface_, &pending_full_, expected_full);
            if (!st.ok())
                return st;
        }
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        if (max_frames_ > 0 && frame_index_ >= max_frames_) {
            if (cc->Outputs().HasTag(kCompletionTag)) {
                ov::Tensor completion(ov::element::f32, {1});
                completion.data<float>()[0] = 1.0f;
                cc->Outputs().Tag(kCompletionTag).AddPacket(
                    MakePacket<ov::Tensor>(std::move(completion)).At(Timestamp(frame_index_)));
            }
            return tool::StatusStop();
        }

        imp_tensor_t* infer_tensor = nullptr;
        imp_tensor_t* full_tensor = nullptr;
        if (emit_frame_ || emit_infer_) {
            infer_tensor = nextFrame(infer_stream_, &pending_infer_);
            if (!infer_tensor)
                return tool::StatusStop();
        }
        if (emit_full_) {
            full_tensor = nextFrame(shared_camera_stream_ ? infer_stream_ : full_stream_,
                                   &pending_full_, shared_camera_stream_ ? 1 : 0);
            if (!full_tensor) {
                if (infer_tensor)
                    imp_tensor_release(infer_tensor);
                return tool::StatusStop();
            }
        }

        const Timestamp ts(frame_index_);
        if (infer_tensor) {
            auto pkt = makeSurfacePacket(infer_tensor);
            if (!pkt)
                return absl::InternalError("GstVideoSourceCalculator: surface frame build failed");
            if (emit_frame_)
                cc->Outputs().Tag(kFrameTag).Add(pkt.release(), ts);
            else
                cc->Outputs().Tag(kFrameInferTag).Add(pkt.release(), ts);
        }
        if (full_tensor) {
            auto pkt = full_va_surface_ ? makeSurfacePacket(full_tensor)
                                        : makeHostNv12Packet(full_tensor);
            if (!pkt)
                return absl::InternalError("GstVideoSourceCalculator: full frame build failed");
            cc->Outputs().Tag(kFrameFullTag).Add(pkt.release(), ts);
        }
        ++frame_index_;
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) override {
        releaseStream(&infer_ctx_, &infer_stream_, &pending_infer_);
        releaseStream(&full_ctx_, &full_stream_, &pending_full_);
        return absl::OkStatus();
    }

 private:
    absl::Status openStream(imp_context_t** ctx, imp_video_stream_t** stream,
                            int w, int h, bool va_surface, imp_tensor_t** pending,
                            imp_tensor_memory_type_t expected_first,
                            const imp_video_branch_t* branches = nullptr, uint32_t branch_count = 0) {
        if (imp_context_create(ctx, nullptr) != IMP_OK)
            return absl::InternalError("imp_context_create failed");

        imp_video_source_t* src = nullptr;
        imp_video_source_create(&src, source_type_);
        if (source_type_ == IMP_SOURCE_CAMERA) {
            if (!camera_device_.empty())
                imp_video_source_set(src, "device", camera_device_.c_str());
        } else {
            imp_video_source_set(src, source_type_ == IMP_SOURCE_FILE ? "path" : "url",
                         source_uri_.c_str());
        }
        if (w > 0)
            imp_video_source_set(src, "width", std::to_string(w).c_str());
        if (h > 0)
            imp_video_source_set(src, "height", std::to_string(h).c_str());

        imp_video_decode_opts_t vopts{};
        vopts.branches = branches;
        vopts.branch_count = branch_count;
        vopts.use_va_surface_memory = va_surface && !use_d3d11_;
        vopts.use_d3d11_surface_memory = va_surface && use_d3d11_;
        imp_status_t st = imp_video_open(stream, src, *ctx, &vopts);
        imp_video_source_destroy(src);
        if (st != IMP_OK)
            return absl::InternalError(std::string("imp_video_open failed: ") +
                                       imp_context_get_error(*ctx));

        st = imp_video_read_frame(pending, *stream, 0);
        if (st != IMP_OK || !*pending)
            return absl::InternalError("first frame read failed");
        if (imp_tensor_get_memory_type(*pending) != expected_first)
            return absl::InternalError("unexpected frame memory type from GStreamer");
        return absl::OkStatus();
    }

    imp_tensor_t* nextFrame(imp_video_stream_t* stream, imp_tensor_t** pending, uint32_t branch = 0) {
        if (*pending) {
            imp_tensor_t* t = *pending;
            *pending = nullptr;
            return t;
        }
        imp_tensor_t* tensor = nullptr;
        imp_status_t st = imp_video_read_frame(&tensor, stream, branch);
        if (st != IMP_OK || !tensor)
            return nullptr;
        return tensor;
    }

    std::unique_ptr<GstVideoFramePacket> makeSurfacePacket(imp_tensor_t* tensor) {
        auto pkt = std::make_unique<GstVideoFramePacket>();
        if (use_d3d11_) {
            void* texture = nullptr;
            void* device = nullptr;
            uint32_t subresource = 0;
            int fw = 0, fh = 0;
            if (imp_tensor_get_d3d11_texture(tensor, &texture, &device, &subresource, &fw, &fh) != IMP_OK) {
                imp_tensor_release(tensor);
                return nullptr;
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
                return nullptr;
            }
            pkt->resource = VaSurfaceFrame{surface_id};
            pkt->width = fw;
            pkt->height = fh;
        }
        pkt->owner = std::shared_ptr<imp_tensor_t>(tensor, imp_tensor_release);
        return pkt;
    }

    std::unique_ptr<GstVideoFramePacket> makeHostNv12Packet(imp_tensor_t* tensor) {
        const uint8_t* y = nullptr;
        const uint8_t* uv = nullptr;
        int fw = 0, fh = 0;
        if (imp_tensor_get_nv12_planes(tensor, &y, &uv, &fw, &fh) != IMP_OK) {
            imp_tensor_release(tensor);
            return nullptr;
        }
        auto pkt = std::make_unique<GstVideoFramePacket>();
        pkt->resource = CpuNv12Frame{y, uv, fw, fw};  // packed NV12: stride == width
        pkt->width = fw;
        pkt->height = fh;
        pkt->owner = std::shared_ptr<imp_tensor_t>(tensor, imp_tensor_release);
        return pkt;
    }

    void releaseStream(imp_context_t** ctx, imp_video_stream_t** stream, imp_tensor_t** pending) {
        if (*pending) {
            imp_tensor_release(*pending);
            *pending = nullptr;
        }
        if (*stream) {
            imp_video_close(*stream);
            *stream = nullptr;
        }
        if (*ctx) {
            imp_context_destroy(*ctx);
            *ctx = nullptr;
        }
    }

    std::string source_uri_;
    std::string camera_device_;
    imp_source_type_t source_type_ = IMP_SOURCE_FILE;
    int width_ = 0;
    int height_ = 0;
    int full_width_ = 0;
    int full_height_ = 0;
    bool full_va_surface_ = false;
    int64_t max_frames_ = 0;
    int64_t frame_index_ = 0;
    bool use_d3d11_ = false;
    bool emit_frame_ = false;
    bool emit_infer_ = false;
    bool emit_full_ = false;
    bool shared_camera_stream_ = false;

    imp_context_t* infer_ctx_ = nullptr;
    imp_video_stream_t* infer_stream_ = nullptr;
    imp_tensor_t* pending_infer_ = nullptr;

    imp_context_t* full_ctx_ = nullptr;
    imp_video_stream_t* full_stream_ = nullptr;
    imp_tensor_t* pending_full_ = nullptr;
};

REGISTER_CALCULATOR(GstVideoSourceCalculator);

}  // namespace mediapipe
