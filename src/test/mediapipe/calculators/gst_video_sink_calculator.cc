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
 * GstVideoSinkCalculator — platform-neutral file-output sink.
 *
 * Consumes the native decoded frame stream (GstVideoFramePacket) and writes it
 * to a readable H.264/MP4 file through the existing imp_video_encoder_* API.
 * The packet's owner (a shared_ptr<imp_tensor_t>) keeps the native source frame
 * alive through encoder submission, so a GPU surface is encoded in place with no
 * host copy on the fast path. One shared calculator serves both the Linux VA and
 * Windows D3D11 graphs; the encoder's device path is selected from the frame's
 * tagged resource.
 *
 * Side packets:
 *   OUTPUT_PATH : std::string — destination video file path.
 *   FRAMERATE   : int         — encode framerate (fps).
 *
 * Input stream:
 *   FRAME       : GstVideoFramePacket — one per decoded frame.
 */

#include <fstream>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "src/mpi/intel_mpi.h"
#include "src/test/mediapipe/calculators/gst_video_frame_packet.hpp"

namespace mediapipe {

namespace {
constexpr char kFrameTag[] = "FRAME";
constexpr char kOutputPathTag[] = "OUTPUT_PATH";
constexpr char kFramerateTag[] = "FRAMERATE";
}  // namespace

class GstVideoSinkCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
        cc->InputSidePackets().Tag(kOutputPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kFramerateTag).Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        output_path_ = cc->InputSidePackets().Tag(kOutputPathTag).Get<std::string>();
        framerate_ = cc->InputSidePackets().Tag(kFramerateTag).Get<int>();
        if (output_path_.empty())
            return absl::InvalidArgumentError("GstVideoSinkCalculator: empty OUTPUT_PATH");
        if (framerate_ <= 0)
            return absl::InvalidArgumentError("GstVideoSinkCalculator: FRAMERATE must be positive");
        if (imp_context_create(&ctx_, nullptr) != IMP_OK)
            return absl::InternalError("imp_context_create failed");
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        const auto& frame = cc->Inputs().Tag(kFrameTag).Get<GstVideoFramePacket>();
        imp_tensor_t* tensor = frame.owner.get();
        RET_CHECK(tensor != nullptr) << "frame packet has no native tensor owner";

        // The encoder needs frame dimensions, known only once the first frame
        // arrives. Create it lazily; pick GPU (zero-copy) encode for native
        // surfaces, host encode for CPU frames.
        if (!encoder_) {
            RET_CHECK_GT(frame.width, 0);
            RET_CHECK_GT(frame.height, 0);
            const bool native_surface =
                std::holds_alternative<VaSurfaceFrame>(frame.resource) ||
                std::holds_alternative<D3D11Frame>(frame.resource);
            imp_video_encode_opts_t eopts{};
            eopts.codec = "h264";
            eopts.framerate = static_cast<uint32_t>(framerate_);
            eopts.encode_device = native_surface ? "GPU" : nullptr;  // VA/D3D11 zero-copy vs host
            eopts.output_path = output_path_.c_str();
            if (imp_video_encoder_create(&encoder_, static_cast<uint32_t>(frame.width),
                                         static_cast<uint32_t>(frame.height), ctx_, &eopts) != IMP_OK)
                return absl::InternalError(std::string("imp_video_encoder_create failed: ") +
                                           imp_context_get_error(ctx_));
        }

        // owner keeps the native frame alive across this synchronous submission.
        if (imp_video_encoder_write(encoder_, tensor) != IMP_OK)
            return absl::InternalError(std::string("imp_video_encoder_write failed: ") +
                                       imp_context_get_error(ctx_));
        ++frames_written_;
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) override {
        if (encoder_) {
            imp_video_encoder_close(encoder_);  // EOS + finalize the file
            encoder_ = nullptr;
        }
        if (ctx_) {
            imp_context_destroy(ctx_);
            ctx_ = nullptr;
        }
        if (frames_written_ == 0)
            return absl::InternalError("GstVideoSinkCalculator: no frames were encoded");
        std::ifstream out(output_path_, std::ios::binary | std::ios::ate);
        if (!out.good() || out.tellg() <= 0)
            return absl::InternalError("GstVideoSinkCalculator: encoder did not finalize " +
                                       output_path_);
        return absl::OkStatus();
    }

 private:
    std::string output_path_;
    int framerate_ = 0;
    int64_t frames_written_ = 0;

    imp_context_t* ctx_ = nullptr;
    imp_video_encoder_t* encoder_ = nullptr;
};

REGISTER_CALCULATOR(GstVideoSinkCalculator);

}  // namespace mediapipe
