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

#include <memory>
#include <string>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/tool/status_util.h"
#include "src/mediapipe_gst_calculators/gst_openvino_inference_calculator.pb.h"
#include "src/mediapipe_gst_calculators/gst_vaapi_inference.hpp"
#include "src/mpi/intel_mpi.h"

namespace mediapipe {
class GstVideoOpenVinoInferenceCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->InputSidePackets().Tag("VIDEO_PATH").Set<std::string>();
        cc->InputSidePackets().Tag("WIDTH").Set<int>();
        cc->InputSidePackets().Tag("HEIGHT").Set<int>();
        if (cc->InputSidePackets().HasTag("MAX_FRAMES"))
            cc->InputSidePackets().Tag("MAX_FRAMES").Set<int>();
        if (cc->InputSidePackets().HasTag("COMPILED_MODEL"))
            cc->InputSidePackets().Tag("COMPILED_MODEL").Set<std::shared_ptr<ov::CompiledModel>>();
        cc->Outputs().Tag("DETECTIONS").Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        options_ = cc->Options<GstOpenVinoInferenceCalculatorOptions>();
        if (cc->InputSidePackets().HasTag("COMPILED_MODEL")) {
            compiled_model_ = cc->InputSidePackets().Tag("COMPILED_MODEL").Get<std::shared_ptr<ov::CompiledModel>>();
            if (!compiled_model_)
                return absl::InvalidArgumentError("COMPILED_MODEL must not be null");
        }
        if ((options_.model_path().empty() && !compiled_model_) || options_.num_streams() == 0)
            return absl::InvalidArgumentError("Either model path or COMPILED_MODEL and positive num_streams are required");
        if (cc->InputSidePackets().HasTag("MAX_FRAMES"))
            max_frames_ = cc->InputSidePackets().Tag("MAX_FRAMES").Get<int>();
        if (imp_context_create(&context_, nullptr) != IMP_OK)
            return absl::InternalError("MPI context creation failed");
        imp_video_source_t* raw_source = nullptr;
        if (imp_video_source_create(&raw_source, IMP_SOURCE_FILE) != IMP_OK)
            return absl::InternalError("MPI source creation failed");
        std::unique_ptr<imp_video_source_t, decltype(&imp_video_source_destroy)> source(raw_source, imp_video_source_destroy);
        const std::string path = cc->InputSidePackets().Tag("VIDEO_PATH").Get<std::string>();
        const std::string width = std::to_string(cc->InputSidePackets().Tag("WIDTH").Get<int>());
        const std::string height = std::to_string(cc->InputSidePackets().Tag("HEIGHT").Get<int>());
        if (imp_video_source_set(source.get(), "path", path.c_str()) != IMP_OK ||
            imp_video_source_set(source.get(), "width", width.c_str()) != IMP_OK ||
            imp_video_source_set(source.get(), "height", height.c_str()) != IMP_OK)
            return absl::InvalidArgumentError("MPI source configuration failed");
        imp_video_decode_opts_t decode{};
        decode.use_va_surface_memory = true;
        if (imp_video_open(&stream_, source.get(), context_, &decode) != IMP_OK)
            return absl::InternalError(imp_context_get_error(context_));
        inference_ = std::make_unique<GstVaapiInference>();
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        if (max_frames_ > 0 && frame_index_ >= max_frames_)
            return tool::StatusStop();
        imp_tensor_t* raw_tensor = nullptr;
        const imp_status_t status = imp_video_read_frame(&raw_tensor, stream_, 0);
        if (status == IMP_ERROR_STREAM_END)
            return tool::StatusStop();
        std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw_tensor, imp_tensor_release);
        if (status != IMP_OK || !tensor)
            return absl::InternalError("MPI frame read failed");
        uint32_t surface = 0;
        void* display = nullptr;
        int width = 0;
        int height = 0;
        if (imp_tensor_get_va_surface(tensor.get(), &surface, &display, &width, &height) != IMP_OK)
            return absl::InternalError("Expected a VA-API surface, no host-memory fallback");
        try {
            if (compiled_model_ && !inference_->initialized())
                inference_->adoptCompiledModel(*compiled_model_, display, width, height);
            const int detections = inference_->infer(display, surface, width, height,
                options_.model_path(), options_.input_name(), options_.performance_hint(), options_.num_streams());
            cc->Outputs().Tag("DETECTIONS").AddPacket(MakePacket<int>(detections).At(Timestamp(frame_index_++)));
            return absl::OkStatus();
        } catch (const std::exception& error) {
            return absl::InternalError(error.what());
        }
    }

    absl::Status Close(CalculatorContext*) override {
        cleanup();
        return absl::OkStatus();
    }

    ~GstVideoOpenVinoInferenceCalculator() override { cleanup(); }

private:
    void cleanup() {
        inference_.reset();
        compiled_model_.reset();
        if (stream_)
            imp_video_close(stream_);
        stream_ = nullptr;
        if (context_)
            imp_context_destroy(context_);
        context_ = nullptr;
    }
    GstOpenVinoInferenceCalculatorOptions options_;
    std::unique_ptr<GstVaapiInference> inference_;
    std::shared_ptr<ov::CompiledModel> compiled_model_;
    imp_context_t* context_ = nullptr;
    imp_video_stream_t* stream_ = nullptr;
    int max_frames_ = 0;
    int64_t frame_index_ = 0;
};
REGISTER_CALCULATOR(GstVideoOpenVinoInferenceCalculator);
}  // namespace mediapipe