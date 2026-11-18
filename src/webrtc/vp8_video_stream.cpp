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
#include "vp8_video_stream.hpp"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vpx/vp8dx.h>
#include <vpx/vpx_decoder.h>

#include "mediapipe/framework/formats/image_frame.h"
#include "src/logging.hpp"

namespace ovms {
namespace {
constexpr size_t MaxQueuedFrames = 16;
constexpr size_t MaxEncodedFrameBytes = 2 * 1024 * 1024;
constexpr unsigned int MaxVideoDimension = 1920;
constexpr size_t ColorChannels = 3;
}  // namespace

struct Vp8VideoStream::Impl {
    struct Frame {
        std::vector<std::byte> data;
        uint32_t timestamp;
    };

    ImageFrameCallback imageFrame;
    ErrorCallback error;
    vpx_codec_ctx_t decoder{};
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Frame> queue;
    bool stopping = false;
    std::string failure;
    std::thread worker;

    Impl(ImageFrameCallback imageFrame, ErrorCallback error) :
        imageFrame(std::move(imageFrame)),
        error(std::move(error)) {
        vpx_codec_dec_cfg_t config{};
        config.threads = 1;
        if (vpx_codec_dec_init(&decoder, vpx_codec_vp8_dx(), &config, 0) != VPX_CODEC_OK)
            throw std::runtime_error("Could not initialize VP8 decoder");
        try {
            worker = std::thread(&Impl::run, this);
        } catch (const std::exception&) {
            vpx_codec_destroy(&decoder);
            throw;
        }
    }

    ~Impl() {
        stop();
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

    void deliver(const vpx_image_t& image, uint32_t timestamp) {
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
        auto frame = std::make_shared<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, width, height);
        cv::Mat rgb(height, width, CV_8UC3, frame->MutablePixelData(), frame->WidthStep());
        cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_I420);
        imageFrame(std::move(frame), timestamp);
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
                    deliver(*image, frame.timestamp);
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

Vp8VideoStream::Vp8VideoStream(ImageFrameCallback imageFrame, ErrorCallback error) :
    impl_(std::make_unique<Impl>(std::move(imageFrame), std::move(error))) {}
Vp8VideoStream::~Vp8VideoStream() = default;

void Vp8VideoStream::push(std::vector<std::byte> frame, uint32_t timestamp) {
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

void Vp8VideoStream::stop() {
    impl_->stop();
}

}  // namespace ovms
