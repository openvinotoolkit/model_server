// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.
#include "movinet_video_stream.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vpx/vp8dx.h>
#include <vpx/vpx_decoder.h>

#include "src/logging.hpp"

namespace ovms {
namespace {
constexpr std::array<size_t, 11> TemporalDepths{2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4};
constexpr size_t PoolInputStart = 1 + std::accumulate(TemporalDepths.begin(), TemporalDepths.end(), size_t{0});
constexpr size_t PoolOutputStart = TemporalDepths.size() + 1;
constexpr size_t PoolCount = 16;
constexpr size_t PoolingWindowFrames = 32;
constexpr size_t PoolScaleInput = PoolInputStart + PoolCount;
constexpr size_t HistoryScaleInput = PoolScaleInput + 1;
constexpr size_t MaxQueuedFrames = 16;
constexpr size_t MaxEncodedFrameBytes = 2 * 1024 * 1024;
constexpr unsigned int MaxVideoDimension = 1920;
constexpr size_t ColorChannels = 3;
constexpr float PixelMax = 255.0f;
constexpr size_t ModelImageSize = 172;
constexpr auto StatisticsInterval = std::chrono::seconds(1);
}  // namespace

struct MovinetVideoModel::Impl {
    ov::CompiledModel model;
    std::vector<std::string> labels;

    Impl(const std::string& path, const std::string& labelsPath) {
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
            ov::inference_num_threads(2), ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
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
        for (size_t index : {PoolScaleInput, HistoryScaleInput}) {
            if (ov::shape_size(model.input(index).get_shape()) != 1)
                throw std::runtime_error("MoViNet pooling/history scales must be scalars");
        }
    }
};

MovinetVideoModel::MovinetVideoModel(const std::string& path, const std::string& labels) :
    impl_(std::make_unique<Impl>(path, labels)) {}
MovinetVideoModel::~MovinetVideoModel() = default;

struct MovinetVideoStream::Impl {
    struct Frame {
        std::vector<std::byte> data;
        uint32_t timestamp;
    };

    std::shared_ptr<MovinetVideoModel> model;
    PredictionCallback prediction;
    ErrorCallback error;
    ov::InferRequest request;
    std::vector<ov::Tensor> inputs;
    std::array<std::vector<float>, PoolCount> poolHistory;
    vpx_codec_ctx_t decoder{};
    bool decoderInitialized = false;
    size_t frames = 0;
    std::chrono::steady_clock::time_point statisticsStart;
    size_t statisticsFrames = 0;
    std::chrono::steady_clock::duration inferenceTime{};
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Frame> queue;
    bool stopping = false;
    std::string failure;
    std::thread worker;

    Impl(std::shared_ptr<MovinetVideoModel> model, PredictionCallback prediction, ErrorCallback error) :
        model(std::move(model)),
        prediction(std::move(prediction)),
        error(std::move(error)),
        request(this->model->impl_->model.create_infer_request()) {
        for (const auto& port : this->model->impl_->model.inputs()) {
            inputs.emplace_back(ov::element::f32, port.get_shape());
            std::memset(inputs.back().data(), 0, inputs.back().get_byte_size());
        }
        for (size_t state = 0; state < PoolCount; ++state)
            poolHistory[state].resize(PoolingWindowFrames * inputs[PoolInputStart + state].get_size());
        vpx_codec_dec_cfg_t config{};
        config.threads = 1;
        if (vpx_codec_dec_init(&decoder, vpx_codec_vp8_dx(), &config, 0) != VPX_CODEC_OK)
            throw std::runtime_error("Could not initialize VP8 decoder");
        decoderInitialized = true;
        try {
            worker = std::thread(&Impl::run, this);
        } catch (const std::exception&) {
            vpx_codec_destroy(&decoder);
            throw;
        }
    }

    ~Impl() {
        stop();
        if (decoderInitialized)
            vpx_codec_destroy(&decoder);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            queue.clear();
        }
        ready.notify_all();
        if (worker.joinable())
            worker.join();
    }

    void infer(const vpx_image_t& image, uint32_t timestamp) {
        if (frames == 0)
            statisticsStart = std::chrono::steady_clock::now();
        if (image.fmt != VPX_IMG_FMT_I420 || image.d_w % 2 || image.d_h % 2)
            throw std::runtime_error("VP8 POC requires even-sized I420 frames");
        const int width = static_cast<int>(image.d_w);
        const int height = static_cast<int>(image.d_h);
        cv::Mat yuv(height + height / 2, width, CV_8UC1);
        size_t offset = 0;
        for (size_t plane = 0; plane < ColorChannels; ++plane) {
            const size_t planeWidth = plane == 0 ? image.d_w : image.d_w / 2;
            const size_t planeHeight = plane == 0 ? image.d_h : image.d_h / 2;
            for (size_t row = 0; row < planeHeight; ++row) {
                std::memcpy(yuv.data + offset, image.planes[plane] + row * image.stride[plane], planeWidth);
                offset += planeWidth;
            }
        }
        cv::Mat rgb;
        cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_I420);
        const auto& shape = inputs[0].get_shape();
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2])));
        const size_t pixels = shape[2] * shape[3];
        float* tensor = inputs[0].data<float>();
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            for (size_t channel = 0; channel < ColorChannels; ++channel)
                tensor[channel * pixels + pixel] = resized.data[pixel * ColorChannels + channel] / PixelMax;
        }
        const size_t poolSlot = frames % PoolingWindowFrames;
        // Evict before inference: the graph adds the current frame to each pooling sum.
        if (frames >= PoolingWindowFrames) {
            for (size_t state = 0; state < PoolCount; ++state) {
                float* accumulator = inputs[PoolInputStart + state].data<float>();
                const size_t elements = inputs[PoolInputStart + state].get_size();
                const float* oldest = poolHistory[state].data() + poolSlot * elements;
                for (size_t index = 0; index < elements; ++index)
                    accumulator[index] -= oldest[index];
            }
        }
        *inputs[PoolScaleInput].data<float>() = 1.0f / static_cast<float>(std::min(frames + 1, PoolingWindowFrames));
        *inputs[HistoryScaleInput].data<float>() = 1.0f;
        for (size_t index = 0; index < inputs.size(); ++index)
            request.set_input_tensor(index, inputs[index]);
        const auto inferenceStart = std::chrono::steady_clock::now();
        request.infer();
        inferenceTime += std::chrono::steady_clock::now() - inferenceStart;
        const auto logits = request.get_output_tensor(0);
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
                std::memcpy(inputs[index].data(), inputs[index + 1].data(), inputs[index].get_byte_size());
            const auto state = request.get_output_tensor(output);
            std::memcpy(inputs[start + depth - 1].data(), state.data(), state.get_byte_size());
            start += depth;
        }
        for (size_t state = 0; state < PoolCount; ++state) {
            float* accumulator = inputs[PoolInputStart + state].data<float>();
            const auto update = request.get_output_tensor(PoolOutputStart + state);
            const float* increment = update.data<const float>();
            std::memcpy(poolHistory[state].data() + poolSlot * update.get_size(), increment, update.get_byte_size());
            for (size_t index = 0; index < update.get_size(); ++index)
                accumulator[index] += increment[index];
        }
        prediction(++frames, timestamp, model->impl_->labels[best], 1.0f / sum);
        ++statisticsFrames;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = now - statisticsStart;
        if (elapsed >= StatisticsInterval) {
            const double seconds = std::chrono::duration<double>(elapsed).count();
            const double inferenceMs = std::chrono::duration<double, std::milli>(inferenceTime).count();
            SPDLOG_LOGGER_INFO(webrtc_logger,
                "MoViNet stream={} prediction_fps={:.2f} avg_inference_ms={:.2f} window_frames={} window_seconds={:.2f} total_frames={} pooling_window_frames={}",
                static_cast<const void*>(this), statisticsFrames / seconds, inferenceMs / statisticsFrames,
                statisticsFrames, seconds, frames, PoolingWindowFrames);
            statisticsStart = now;
            statisticsFrames = 0;
            inferenceTime = {};
        }
    }

    void run() {
        try {
            while (true) {
                Frame frame;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready.wait(lock, [this] { return stopping || !queue.empty() || !failure.empty(); });
                    if (stopping)
                        return;
                    if (!failure.empty())
                        throw std::runtime_error(failure);
                    frame = std::move(queue.front());
                    queue.pop_front();
                }
                vpx_codec_stream_info_t info{};
                info.sz = sizeof(info);
                const auto* data = reinterpret_cast<const uint8_t*>(frame.data.data());
                if (vpx_codec_peek_stream_info(vpx_codec_vp8_dx(), data, frame.data.size(), &info) == VPX_CODEC_OK &&
                    (info.w > MaxVideoDimension || info.h > MaxVideoDimension))
                    throw std::runtime_error("VP8 POC supports video dimensions up to 1920");
                if (vpx_codec_decode(&decoder, data, frame.data.size(), nullptr, 0) != VPX_CODEC_OK)
                    throw std::runtime_error(std::string("VP8 decode failed: ") + vpx_codec_error(&decoder));
                vpx_codec_iter_t iterator = nullptr;
                while (vpx_image_t* image = vpx_codec_get_frame(&decoder, &iterator))
                    infer(*image, frame.timestamp);
            }
        } catch (const std::exception& exception) {
            SPDLOG_LOGGER_ERROR(webrtc_logger, "WebRTC video stopped: {}", exception.what());
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
                queue.clear();
            }
            error(exception.what());
        }
    }
};

MovinetVideoStream::MovinetVideoStream(std::shared_ptr<MovinetVideoModel> model, PredictionCallback prediction, ErrorCallback error) :
    impl_(std::make_unique<Impl>(std::move(model), std::move(prediction), std::move(error))) {}
MovinetVideoStream::~MovinetVideoStream() = default;

void MovinetVideoStream::push(std::vector<std::byte> frame, uint32_t timestamp) {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->stopping)
            return;
        if (frame.empty() || frame.size() > MaxEncodedFrameBytes || impl_->queue.size() >= MaxQueuedFrames)
            impl_->failure = "Invalid VP8 frame or video input queue overflow; create a new call";
        else
            impl_->queue.push_back({std::move(frame), timestamp});
    }
    impl_->ready.notify_one();
}

void MovinetVideoStream::stop() {
    impl_->stop();
}

}  // namespace ovms
