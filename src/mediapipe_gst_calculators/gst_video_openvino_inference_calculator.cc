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

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#pragma GCC diagnostic pop

#include "src/mediapipe_gst_calculators/gst_openvino_inference_calculator.pb.h"
#include "src/mpi/intel_mpi.h"

#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#endif

namespace mediapipe {

namespace {
constexpr char kVideoPathTag[] = "VIDEO_PATH";
constexpr char kWidthTag[] = "WIDTH";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kMaxFramesTag[] = "MAX_FRAMES";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr float kFaceConfidenceThreshold = 0.5f;
}  // namespace

class GstVideoOpenVinoInferenceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->InputSidePackets().Tag(kVideoPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kWidthTag).Set<int>();
        cc->InputSidePackets().Tag(kHeightTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kMaxFramesTag))
            cc->InputSidePackets().Tag(kMaxFramesTag).Set<int>();
        cc->Outputs().Tag(kDetectionsTag).Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
#ifndef _WIN32
        return absl::UnimplementedError("GstVideoOpenVinoInferenceCalculator supports Windows D3D11 only");
#else
        const GstOpenVinoInferenceCalculatorOptions& options =
            cc->Options<GstOpenVinoInferenceCalculatorOptions>();
        if (options.model_path().empty())
            return absl::InvalidArgumentError("GstVideoOpenVinoInferenceCalculator: empty model_path");
        model_path_ = options.model_path();
        input_name_ = options.input_name();
        performance_hint_ = options.performance_hint();
        num_streams_ = options.num_streams();
        if (num_streams_ == 0)
            return absl::InvalidArgumentError("GstVideoOpenVinoInferenceCalculator: num_streams must be positive");
        if (imp_context_create(&context_, nullptr) != IMP_OK)
            return absl::InternalError("GstVideoOpenVinoInferenceCalculator: imp_context_create failed");
        imp_video_source_t* source = nullptr;
        if (imp_video_source_create(&source, IMP_SOURCE_FILE) != IMP_OK)
            return absl::InternalError("GstVideoOpenVinoInferenceCalculator: imp_video_source_create failed");
        const std::string video_path = cc->InputSidePackets().Tag(kVideoPathTag).Get<std::string>();
        const int width = cc->InputSidePackets().Tag(kWidthTag).Get<int>();
        const int height = cc->InputSidePackets().Tag(kHeightTag).Get<int>();
        if (cc->InputSidePackets().HasTag(kMaxFramesTag))
            max_frames_ = cc->InputSidePackets().Tag(kMaxFramesTag).Get<int>();
        imp_video_source_set(source, "path", video_path.c_str());
        imp_video_source_set(source, "width", std::to_string(width).c_str());
        imp_video_source_set(source, "height", std::to_string(height).c_str());
        imp_video_decode_opts_t decode_options{};
        decode_options.use_d3d11_surface_memory = true;
        const imp_status_t status = imp_video_open(&stream_, source, context_, &decode_options);
        imp_video_source_destroy(source);
        if (status != IMP_OK)
            return absl::InternalError(std::string("GstVideoOpenVinoInferenceCalculator: ") +
                                       imp_context_get_error(context_));
        return absl::OkStatus();
#endif
    }

    absl::Status Process(CalculatorContext* cc) override {
#ifndef _WIN32
        return absl::UnimplementedError("GstVideoOpenVinoInferenceCalculator supports Windows D3D11 only");
#else
        if (max_frames_ > 0 && frame_index_ >= max_frames_)
            return tool::StatusStop();
        imp_tensor_t* raw_tensor = nullptr;
        const imp_status_t read_status = imp_video_read_frame(&raw_tensor, stream_, 0);
        if (read_status == IMP_ERROR_STREAM_END)
            return tool::StatusStop();
        if (read_status != IMP_OK || raw_tensor == nullptr)
            return absl::InternalError("GstVideoOpenVinoInferenceCalculator: frame read failed");
        std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> tensor(raw_tensor, imp_tensor_release);
        void* texture_handle = nullptr;
        void* decode_device = nullptr;
        uint32_t subresource = 0;
        int width = 0;
        int height = 0;
        if (imp_tensor_get_d3d11_texture(tensor.get(), &texture_handle, &decode_device, &subresource,
                                         &width, &height) != IMP_OK)
            return absl::InternalError("GstVideoOpenVinoInferenceCalculator: expected D3D11 texture");
        RET_CHECK_EQ(subresource, 0u) << "D3D11 texture array slice is not importable by OpenVINO";
        ID3D11Texture2D* texture = static_cast<ID3D11Texture2D*>(texture_handle);
        if (!compiled_model_) {
            ID3D11Device* raw_device = nullptr;
            texture->GetDevice(&raw_device);
            RET_CHECK(raw_device != nullptr);
            device_.Attach(raw_device);
            absl::Status initialize_status = initializeModel();
            if (!initialize_status.ok())
                return initialize_status;
        }
        const ov::Shape shape_y{1, static_cast<size_t>(height), static_cast<size_t>(width), 1};
        const ov::Shape shape_uv{1, static_cast<size_t>(height / 2), static_cast<size_t>(width / 2), 2};
        infer_request_.set_tensor(input_name_ + "/y", d3d_context_->create_tensor(
            ov::element::u8, shape_y, texture, 0));
        infer_request_.set_tensor(input_name_ + "/uv", d3d_context_->create_tensor(
            ov::element::u8, shape_uv, texture, 1));
        try {
            infer_request_.infer();
        } catch (const std::exception& exception) {
            return absl::InternalError(std::string("GstVideoOpenVinoInferenceCalculator inference failed: ") + exception.what());
        }
        const ov::Tensor output = infer_request_.get_output_tensor(0);
        const float* values = output.data<const float>();
        int detections = 0;
        for (size_t result = 0; result + 7 <= output.get_size(); result += 7) {
            if (values[result + 2] >= kFaceConfidenceThreshold)
                ++detections;
        }
        cc->Outputs().Tag(kDetectionsTag).AddPacket(
            MakePacket<int>(detections).At(Timestamp(frame_index_++)));
        return absl::OkStatus();
#endif
    }

    absl::Status Close(CalculatorContext*) override {
        if (stream_ != nullptr)
            imp_video_close(stream_);
        if (context_ != nullptr)
            imp_context_destroy(context_);
        return absl::OkStatus();
    }

 private:
#ifdef _WIN32
    absl::Status initializeModel() {
        try {
            std::shared_ptr<ov::Model> raw_model = core_.read_model(
                model_path_, {}, ov::AnyMap{{"ENABLE_MMAP", "NO"}});
            ov::preprocess::PrePostProcessor preprocessor(raw_model);
            preprocessor.input().tensor().set_element_type(ov::element::u8)
                .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                .set_memory_type(ov::intel_gpu::memory_type::surface);
            preprocessor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
            preprocessor.input().model().set_layout("NCHW");
            d3d_context_ = std::make_unique<ov::intel_gpu::ocl::D3DContext>(core_, device_.Get());
            const ov::AnyMap plugin_config{{"PERFORMANCE_HINT", performance_hint_},
                                            {"NUM_STREAMS", std::to_string(num_streams_)}};
            compiled_model_ = core_.compile_model(preprocessor.build(), *d3d_context_, plugin_config);
            infer_request_ = compiled_model_.create_infer_request();
            return absl::OkStatus();
        } catch (const std::exception& exception) {
            return absl::InternalError(std::string("GstVideoOpenVinoInferenceCalculator initialization failed: ") + exception.what());
        }
    }

    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    std::unique_ptr<ov::intel_gpu::ocl::D3DContext> d3d_context_;
#endif
    imp_context_t* context_ = nullptr;
    imp_video_stream_t* stream_ = nullptr;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    std::string model_path_;
    std::string input_name_;
    std::string performance_hint_;
    uint32_t num_streams_ = 1;
    int max_frames_ = 0;
    int64_t frame_index_ = 0;
};

REGISTER_CALCULATOR(GstVideoOpenVinoInferenceCalculator);

}  // namespace mediapipe