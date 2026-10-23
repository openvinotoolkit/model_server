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
#pragma GCC diagnostic pop

#include "src/mediapipe_gst_calculators/gst_openvino_inference_calculator.pb.h"
#include "src/mediapipe_gst_calculators/gst_video_frame_packet.hpp"

#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#endif

namespace mediapipe {

namespace {
constexpr char kFrameTag[] = "FRAME";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr float kFaceConfidenceThreshold = 0.5f;
}  // namespace

class GstOpenVinoInferenceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
        cc->Outputs().Tag(kDetectionsTag).Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        const GstOpenVinoInferenceCalculatorOptions& options =
            cc->Options<GstOpenVinoInferenceCalculatorOptions>();
        if (options.model_path().empty())
            return absl::InvalidArgumentError("GstOpenVinoInferenceCalculator: empty model_path");
        model_path_ = options.model_path();
        input_name_ = options.input_name();
        performance_hint_ = options.performance_hint();
        num_streams_ = options.num_streams();
        if (num_streams_ == 0)
            return absl::InvalidArgumentError("GstOpenVinoInferenceCalculator: num_streams must be positive");
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
#ifndef _WIN32
        return absl::UnimplementedError("GstOpenVinoInferenceCalculator supports D3D11 frames on Windows only");
#else
        const GstVideoFramePacket& frame = cc->Inputs().Tag(kFrameTag).Get<GstVideoFramePacket>();
        RET_CHECK_GT(frame.width, 0);
        RET_CHECK_GT(frame.height, 0);
        RET_CHECK_EQ(frame.width % 2, 0) << "NV12 width must be even";
        RET_CHECK_EQ(frame.height % 2, 0) << "NV12 height must be even";
        const D3D11Frame* d3d11_frame = std::get_if<D3D11Frame>(&frame.resource);
        RET_CHECK(d3d11_frame != nullptr) << "direct OpenVINO requires a D3D11 NV12 frame";
        RET_CHECK(d3d11_frame->texture_handle != 0) << "invalid D3D11 texture handle";
        RET_CHECK_EQ(d3d11_frame->subresource, 0u)
            << "D3D11 texture array slice is not importable by OpenVINO";

        ID3D11Texture2D* texture = reinterpret_cast<ID3D11Texture2D*>(d3d11_frame->texture_handle);
        if (!compiled_model_) {
            ID3D11Device* raw_device = nullptr;
            texture->GetDevice(&raw_device);
            RET_CHECK(raw_device != nullptr) << "D3D11 texture has no device";
            device_.Attach(raw_device);
            absl::Status initialize_status = initializeModel();
            if (!initialize_status.ok())
                return initialize_status;
        }

        const ov::Shape shape_y{1, static_cast<size_t>(frame.height), static_cast<size_t>(frame.width), 1};
        const ov::Shape shape_uv{1, static_cast<size_t>(frame.height / 2), static_cast<size_t>(frame.width / 2), 2};
        const ov::Tensor y_tensor = d3d_context_->create_tensor(
            ov::element::u8, shape_y, texture, 0);
        const ov::Tensor uv_tensor = d3d_context_->create_tensor(
            ov::element::u8, shape_uv, texture, 1);
        infer_request_.set_tensor(input_name_ + "/y", y_tensor);
        infer_request_.set_tensor(input_name_ + "/uv", uv_tensor);
        try {
            infer_request_.infer();
        } catch (const std::exception& exception) {
            return absl::InternalError(
                std::string("GstOpenVinoInferenceCalculator inference failed: ") + exception.what());
        }

        const ov::Tensor output = infer_request_.get_output_tensor(0);
        const float* values = output.data<const float>();
        int detections = 0;
        for (size_t result = 0; result + 7 <= output.get_size(); result += 7) {
            if (values[result + 2] >= kFaceConfidenceThreshold)
                ++detections;
        }
        cc->Outputs().Tag(kDetectionsTag).AddPacket(
            MakePacket<int>(detections).At(cc->InputTimestamp()));
        return absl::OkStatus();
#endif
    }

 private:
#ifdef _WIN32
    absl::Status initializeModel() {
        try {
            std::shared_ptr<ov::Model> raw_model = core_.read_model(
                model_path_, {}, ov::AnyMap{{"ENABLE_MMAP", "NO"}});
            ov::preprocess::PrePostProcessor preprocessor(raw_model);
            preprocessor.input()
                .tensor()
                .set_element_type(ov::element::u8)
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
            return absl::InternalError(
                std::string("GstOpenVinoInferenceCalculator initialization failed: ") + exception.what());
        }
    }

    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    std::unique_ptr<ov::intel_gpu::ocl::D3DContext> d3d_context_;
#endif
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    std::string model_path_;
    std::string input_name_;
    std::string performance_hint_;
    uint32_t num_streams_ = 1;
};

REGISTER_CALCULATOR(GstOpenVinoInferenceCalculator);

}  // namespace mediapipe