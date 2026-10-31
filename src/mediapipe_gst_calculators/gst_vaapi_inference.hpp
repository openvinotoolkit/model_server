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

#ifndef SRC_MEDIAPIPE_GST_CALCULATORS_GST_VAAPI_INFERENCE_HPP_
#define SRC_MEDIAPIPE_GST_CALCULATORS_GST_VAAPI_INFERENCE_HPP_

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

namespace mediapipe {
struct GstVaapiInferenceTimings {
    std::chrono::nanoseconds tensorImport{};
    std::chrono::nanoseconds tensorBinding{};
    std::chrono::nanoseconds inferCall{};
    std::chrono::nanoseconds outputHandling{};
};

class GstVaapiInference {
public:
    // Compiles modelPath with NV12 surface input and BGR conversion for the given VA context and frame size.
    static ov::CompiledModel compileForVaContext(ov::Core& core, ov::intel_gpu::ocl::VAContext& context,
        int width, int height, const std::string& modelPath,
        const std::string& performanceHint, uint32_t streams) {
        ov::preprocess::PrePostProcessor processor(core.read_model(modelPath));
        processor.input().tensor().set_element_type(ov::element::u8).set_spatial_static_shape(height, width).set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"}).set_memory_type(ov::intel_gpu::memory_type::surface);
        processor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
        processor.input().model().set_layout("NCHW");
        return core.compile_model(processor.build(), context,
            {{"PERFORMANCE_HINT", performanceHint}, {"NUM_STREAMS", std::to_string(streams)}});
    }

    bool initialized() const {
        return context_ != nullptr;
    }

    // compiled must come from compileForVaContext() on a context created from the same display and frame size.
    void adoptCompiledModel(const ov::CompiledModel& compiled, void* display, int width, int height) {
        if (display == nullptr || width <= 0 || height <= 0 || width % 2 || height % 2)
            throw std::invalid_argument("Invalid VA-API frame");
        if (context_)
            throw std::logic_error("GstVaapiInference is already initialized");
        context_ = std::make_unique<ov::intel_gpu::ocl::VAContext>(compiled.get_context().as<ov::intel_gpu::ocl::VAContext>());
        compiled_ = compiled;
        request_ = compiled_.create_infer_request();
        display_ = display;
        width_ = width;
        height_ = height;
    }

    void initialize(void* display, int width, int height,
        const std::string& modelPath, const std::string& inputName,
        const std::string& performanceHint, uint32_t streams) {
        if (display == nullptr || width <= 0 || height <= 0 || width % 2 || height % 2)
            throw std::invalid_argument("Invalid VA-API frame");
        if (!context_) {
            display_ = display;
            width_ = width;
            height_ = height;
            context_ = std::make_unique<ov::intel_gpu::ocl::VAContext>(core_, display);
            compiled_ = compileForVaContext(core_, *context_, width, height, modelPath, performanceHint, streams);
            request_ = compiled_.create_infer_request();
        }
        if (display != display_ || width != width_ || height != height_)
            throw std::invalid_argument("VA-API display or frame dimensions changed during inference");
    }

    int infer(void* display, uint32_t surface, int width, int height,
        const std::string& modelPath, const std::string& inputName,
        const std::string& performanceHint, uint32_t streams,
        GstVaapiInferenceTimings* timings = nullptr) {
        initialize(display, width, height, modelPath, inputName, performanceHint, streams);
        const auto importStart = timings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto planes = context_->create_tensor_nv12(height, width, surface);
        if (timings)
            timings->tensorImport += std::chrono::steady_clock::now() - importStart;
        const auto bindingStart = timings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        request_.set_tensor(inputName + "/y", planes.first);
        request_.set_tensor(inputName + "/uv", planes.second);
        if (timings)
            timings->tensorBinding += std::chrono::steady_clock::now() - bindingStart;
        const auto inferStart = timings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        request_.infer();
        if (timings)
            timings->inferCall += std::chrono::steady_clock::now() - inferStart;
        const auto outputStart = timings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const ov::Tensor output = request_.get_output_tensor();
        const float* values = output.data<const float>();
        int detections = 0;
        for (size_t result = 0; result + 7 <= output.get_size(); result += 7) {
            if (values[result + 2] >= 0.5f)
                ++detections;
        }
        if (timings)
            timings->outputHandling += std::chrono::steady_clock::now() - outputStart;
        return detections;
    }

    const ov::CompiledModel& compiledModel() const {
        return compiled_;
    }

private:
    ov::Core core_;
    void* display_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    std::unique_ptr<ov::intel_gpu::ocl::VAContext> context_;
    ov::CompiledModel compiled_;
    ov::InferRequest request_;
};
}  // namespace mediapipe
#endif