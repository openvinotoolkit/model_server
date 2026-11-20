//*****************************************************************************
// Copyright 2026 Intel Corporation
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
//*****************************************************************************
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "classification.hpp"
#include "src/logging.hpp"
#include "src/mediapipe_internal/graph_side_packets.hpp"
#include "src/mediapipe_internal/live_graph_config.hpp"
#include "src/mediapipe_internal/node_initializer.hpp"
#include "src/movinet/movinet_calculator.pb.h"

namespace ovms {
namespace {
constexpr char CalculatorName[] = "MovinetCalculator";
constexpr char ResourcesTag[] = "MOVINET_RESOURCES";
constexpr std::array<size_t, 11> TemporalDepths{2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4};
constexpr size_t PoolInputStart = 1 + std::accumulate(TemporalDepths.begin(), TemporalDepths.end(), size_t{0});
constexpr size_t PoolOutputStart = TemporalDepths.size() + 1;
constexpr size_t PoolCount = 16;
constexpr size_t PoolingWindowFrames = 32;
constexpr size_t PoolScaleInput = PoolInputStart + PoolCount;
constexpr size_t HistoryScaleInput = PoolScaleInput + 1;
constexpr size_t ColorChannels = 3;
constexpr size_t ModelImageSize = 172;
constexpr float PixelMax = 255.0f;
constexpr int InferenceThreads = 2;
constexpr auto StatisticsInterval = std::chrono::seconds(1);
}  // namespace

class MovinetResources {
public:
    ov::CompiledModel model;
    std::vector<std::string> labels;

    MovinetResources(const std::string& path, const std::string& labelsPath) {
        std::ifstream input(labelsPath);
        if (!input)
            throw std::runtime_error("Cannot open MoViNet labels: " + labelsPath);
        std::string label;
        while (std::getline(input, label)) {
            if (!label.empty() && label.back() == '\r')
                label.pop_back();
            labels.push_back(label);
        }
        ov::Core core;
        model = core.compile_model(path, "CPU",
            ov::inference_num_threads(InferenceThreads), ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::hint::inference_precision(ov::element::f32));
        if (model.inputs().size() != HistoryScaleInput + 1 ||
            model.outputs().size() != PoolOutputStart + PoolCount ||
            model.output(0).get_shape() != ov::Shape{1, labels.size()} || labels.empty() ||
            model.input(0).get_shape() != ov::Shape{1, ColorChannels, ModelImageSize, ModelImageSize})
            throw std::runtime_error("Unsupported MoViNet export signature; expected the custom 47-input A0 model");
        size_t inputIndex = 1;
        for (size_t output = 1; output <= TemporalDepths.size(); ++output) {
            for (size_t history = 0; history < TemporalDepths[output - 1]; ++history) {
                if (model.input(inputIndex++).get_shape() != model.output(output).get_shape())
                    throw std::runtime_error("MoViNet temporal state shape mismatch");
            }
        }
        for (size_t offset = 0; offset < PoolCount; ++offset) {
            if (model.input(PoolInputStart + offset).get_shape() != model.output(PoolOutputStart + offset).get_shape())
                throw std::runtime_error("MoViNet pooling state shape mismatch");
        }
        for (const auto& port : model.inputs()) {
            if (port.get_element_type() != ov::element::f32)
                throw std::runtime_error("MoViNet requires FP32 inputs");
        }
        for (const auto& port : model.outputs()) {
            if (port.get_element_type() != ov::element::f32)
                throw std::runtime_error("MoViNet requires FP32 outputs");
        }
        for (size_t index : {PoolScaleInput, HistoryScaleInput}) {
            if (ov::shape_size(model.input(index).get_shape()) != 1)
                throw std::runtime_error("MoViNet pooling/history scales must be scalars");
        }
    }
};

class MovinetCalculator : public mediapipe::CalculatorBase {
    std::shared_ptr<MovinetResources> resources_;
    ov::InferRequest request_;
    std::vector<ov::Tensor> inputs_;
    std::array<std::vector<float>, PoolCount> poolHistory_;
    size_t frames_ = 0;
    std::chrono::steady_clock::time_point statisticsStart_ = std::chrono::steady_clock::now();
    size_t statisticsFrames_ = 0;
    std::chrono::steady_clock::duration inferenceTime_{};

    VideoClassification infer(const mediapipe::ImageFrame& image) {
        if (image.Format() != mediapipe::ImageFormat::SRGB || image.Width() <= 0 || image.Height() <= 0)
            throw std::runtime_error("MoViNet requires a nonempty RGB ImageFrame");
        cv::Mat rgb(image.Height(), image.Width(), CV_8UC3,
            const_cast<uint8_t*>(image.PixelData()), image.WidthStep());
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(ModelImageSize, ModelImageSize));
        constexpr size_t Pixels = ModelImageSize * ModelImageSize;
        float* tensor = inputs_[0].data<float>();
        for (size_t pixel = 0; pixel < Pixels; ++pixel) {
            for (size_t channel = 0; channel < ColorChannels; ++channel)
                tensor[channel * Pixels + pixel] = resized.data[pixel * ColorChannels + channel] / PixelMax;
        }
        const size_t poolSlot = frames_ % PoolingWindowFrames;
        // Evict before inference: the model adds the current frame to each pooling sum.
        if (frames_ >= PoolingWindowFrames) {
            for (size_t state = 0; state < PoolCount; ++state) {
                float* accumulator = inputs_[PoolInputStart + state].data<float>();
                const size_t elements = inputs_[PoolInputStart + state].get_size();
                const float* oldest = poolHistory_[state].data() + poolSlot * elements;
                for (size_t index = 0; index < elements; ++index)
                    accumulator[index] -= oldest[index];
            }
        }
        *inputs_[PoolScaleInput].data<float>() = 1.0f / static_cast<float>(std::min(frames_ + 1, PoolingWindowFrames));
        *inputs_[HistoryScaleInput].data<float>() = 1.0f;
        const auto inferenceStart = std::chrono::steady_clock::now();
        request_.infer();
        inferenceTime_ += std::chrono::steady_clock::now() - inferenceStart;
        const auto logits = request_.get_output_tensor(0);
        const float* values = logits.data<const float>();
        const size_t classes = logits.get_size();
        for (size_t index = 0; index < classes; ++index) {
            if (!std::isfinite(values[index]))
                throw std::runtime_error("MoViNet returned non-finite logits");
        }
        const size_t best = std::max_element(values, values + classes) - values;
        float sum = 0;
        for (size_t index = 0; index < classes; ++index)
            sum += std::exp(values[index] - values[best]);

        size_t start = 1;
        for (size_t output = 1; output <= TemporalDepths.size(); ++output) {
            const size_t depth = TemporalDepths[output - 1];
            for (size_t index = start; index < start + depth - 1; ++index)
                std::memcpy(inputs_[index].data(), inputs_[index + 1].data(), inputs_[index].get_byte_size());
            const auto state = request_.get_output_tensor(output);
            std::memcpy(inputs_[start + depth - 1].data(), state.data(), state.get_byte_size());
            start += depth;
        }
        for (size_t state = 0; state < PoolCount; ++state) {
            float* accumulator = inputs_[PoolInputStart + state].data<float>();
            const auto update = request_.get_output_tensor(PoolOutputStart + state);
            const float* increment = update.data<const float>();
            std::memcpy(poolHistory_[state].data() + poolSlot * update.get_size(), increment, update.get_byte_size());
            for (size_t index = 0; index < update.get_size(); ++index)
                accumulator[index] += increment[index];
        }
        ++frames_;
        ++statisticsFrames_;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = now - statisticsStart_;
        if (elapsed >= StatisticsInterval) {
            const double seconds = std::chrono::duration<double>(elapsed).count();
            const double inferenceMs = std::chrono::duration<double, std::milli>(inferenceTime_).count();
            SPDLOG_INFO(
                "MoViNet stream={} prediction_fps={:.2f} avg_inference_ms={:.2f} window_frames={} window_seconds={:.2f} total_frames={} pooling_window_frames={}",
                static_cast<const void*>(this), statisticsFrames_ / seconds, inferenceMs / statisticsFrames_,
                statisticsFrames_, seconds, frames_, PoolingWindowFrames);
            statisticsStart_ = now;
            statisticsFrames_ = 0;
            inferenceTime_ = {};
        }
        return {resources_->labels[best], 1.0f / sum};
    }

public:
    static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
        cc->Inputs().Tag("IMAGE").Set<mediapipe::ImageFrame>();
        cc->Outputs().Tag("CLASSIFICATION").Set<VideoClassification>();
        cc->InputSidePackets().Tag(ResourcesTag).Set<MovinetResourcesMap>();
        return absl::OkStatus();
    }

    absl::Status Open(mediapipe::CalculatorContext* cc) override {
        const auto& resources = cc->InputSidePackets().Tag(ResourcesTag).Get<MovinetResourcesMap>();
        const auto found = resources.find(cc->NodeName());
        if (found == resources.end() || !found->second)
            return absl::InvalidArgumentError("Missing initialized MoViNet resources");
        try {
            resources_ = found->second;
            request_ = resources_->model.create_infer_request();
            for (const auto& port : resources_->model.inputs()) {
                inputs_.emplace_back(ov::element::f32, port.get_shape());
                std::memset(inputs_.back().data(), 0, inputs_.back().get_byte_size());
                request_.set_input_tensor(inputs_.size() - 1, inputs_.back());
            }
            for (size_t state = 0; state < PoolCount; ++state)
                poolHistory_[state].resize(PoolingWindowFrames * inputs_[PoolInputStart + state].get_size());
            cc->SetOffset(mediapipe::TimestampDiff(0));
            return absl::OkStatus();
        } catch (const std::exception& exception) {
            SPDLOG_ERROR("Cannot start MoViNet graph: {}", exception.what());
            return absl::InternalError(exception.what());
        }
    }

    absl::Status Process(mediapipe::CalculatorContext* cc) override {
        if (cc->Inputs().Tag("IMAGE").IsEmpty())
            return absl::OkStatus();
        try {
            cc->Outputs().Tag("CLASSIFICATION").AddPacket(
                mediapipe::MakePacket<VideoClassification>(infer(cc->Inputs().Tag("IMAGE").Get<mediapipe::ImageFrame>()))
                    .At(cc->InputTimestamp()));
            return absl::OkStatus();
        } catch (const std::exception& exception) {
            SPDLOG_ERROR("MoViNet graph inference failed: {}", exception.what());
            return absl::InternalError(exception.what());
        }
    }
};
REGISTER_CALCULATOR(MovinetCalculator);

namespace {
class MovinetNodeInitializer final : public NodeInitializer {
public:
    bool matches(const std::string& calculatorName) const override {
        return calculatorName == CalculatorName;
    }

    Status initialize(const mediapipe::CalculatorGraphConfig_Node& node,
        const std::string& graphName, const std::string& basePath,
        GraphSidePackets& sidePackets, GraphMetadata&,
        PythonBackend*, const MetricProvider*) override {
        mediapipe::MovinetCalculatorOptions options;
        if (node.name().empty() || node.node_options_size() != 1 ||
            !node.node_options(0).UnpackTo(&options) ||
            options.model_path().empty() || options.labels_path().empty())
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID,
                "MoViNet requires a named node and model_path/labels_path options");
        if (sidePackets.movinetResourcesMap.count(node.name()))
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Duplicate MoViNet node: " + node.name());
        try {
            auto resources = std::make_shared<MovinetResources>(
                (std::filesystem::path(basePath) / options.model_path()).string(),
                (std::filesystem::path(basePath) / options.labels_path()).string());
            sidePackets.movinetResourcesMap.emplace(node.name(), std::move(resources));
            return StatusCode::OK;
        } catch (const std::exception& exception) {
            SPDLOG_ERROR("Cannot load MoViNet graph {}: {}", graphName, exception.what());
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, exception.what());
        }
    }

    Status prepareLiveGraph(const mediapipe::CalculatorGraphConfig& config,
        GraphSidePackets& sidePackets) override {
        if (config.node_size() != 1 || config.node(0).calculator() != CalculatorName)
            return StatusCode::OK;
        if (!sidePackets.movinetResourcesMap.count(config.node(0).name()))
            return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Missing initialized MoViNet resources");
        auto profile = makeLiveGraphConfig(config, LiveGraphVideoInputQueueSize,
            {{"video", LiveGraphStreamCodec::imageFrame()}},
            {{"classification", LiveGraphStreamCodec::typed<VideoClassification>()}});
        auto* node = profile->graph.mutable_node(0);
        node->clear_input_stream();
        node->add_input_stream("IMAGE:video");
        node->clear_output_stream();
        node->add_output_stream("CLASSIFICATION:classification");
        return registerLiveGraphProfile(VideoClassificationLiveGraphProfile, std::move(profile), sidePackets);
    }
};

const bool movinetNodeInitializerRegistered = [] {
    NodeInitializerRegistry::instance().add(std::make_unique<MovinetNodeInitializer>());
    return true;
}();
}  // namespace
}  // namespace ovms
