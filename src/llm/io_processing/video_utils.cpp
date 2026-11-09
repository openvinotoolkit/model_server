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

#include "video_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define MINIMP4_IMPLEMENTATION
#include <minimp4.h>
#include <codec_api.h>

#include "image_utils.hpp"
#include "src/logging.hpp"
#include "src/predict_request_validation_utils_impl.hpp"

namespace ovms {

namespace {

struct EncodedVideoFile {
    std::FILE* input;
    size_t size;
};

struct FileCloser {
    void operator()(std::FILE* file) const {
        if (file != nullptr) {
            std::fclose(file);
        }
    }
};

struct Mp4DemuxGuard {
    MP4D_demux_t& demux;

    ~Mp4DemuxGuard() {
        MP4D_close(&demux);
    }
};

struct OpenH264DecoderGuard {
    ISVCDecoder* decoder = nullptr;
    bool initialized = false;

    ~OpenH264DecoderGuard() {
        if (decoder != nullptr) {
            if (initialized) {
                decoder->Uninitialize();
            }
            WelsDestroyDecoder(decoder);
        }
    }
};

int readVideoBytes(int64_t offset, void* output, size_t size, void* token) {
    auto* file = static_cast<EncodedVideoFile*>(token);
    if (offset < 0 || static_cast<uint64_t>(offset) > file->size || size > file->size - static_cast<size_t>(offset)) {
        return 1;
    }
    if (std::fseek(file->input, offset, SEEK_SET) != 0) {
        return 1;
    }
    return std::fread(output, 1, size, file->input) == size ? 0 : 1;
}

bool isRemoteMediaSource(const std::string& mediaSource) {
    return mediaSource.rfind("http://", 0) == 0 || mediaSource.rfind("https://", 0) == 0 ||
           mediaSource.rfind("ftp://", 0) == 0 || mediaSource.rfind("sftp://", 0) == 0;
}

void appendStartCode(std::vector<uint8_t>& annexB, const uint8_t* nal, size_t size) {
    annexB.insert(annexB.end(), {0, 0, 0, 1});
    annexB.insert(annexB.end(), nal, nal + size);
}

bool appendLengthPrefixedSample(std::vector<uint8_t>& annexB, const uint8_t* sample, size_t size, size_t lengthPrefixBytes) {
    if (lengthPrefixBytes != 1 && lengthPrefixBytes != 2 && lengthPrefixBytes != 4) {
        return false;
    }
    size_t position = 0;
    while (position < size) {
        if (size - position < lengthPrefixBytes) {
            return false;
        }
        uint32_t nalSize = 0;
        for (size_t byteIndex = 0; byteIndex < lengthPrefixBytes; ++byteIndex) {
            nalSize = (nalSize << 8) | sample[position + byteIndex];
        }
        position += lengthPrefixBytes;
        if (nalSize > size - position) {
            return false;
        }
        appendStartCode(annexB, sample + position, nalSize);
        position += nalSize;
    }
    return true;
}

uint8_t clampToByte(int value) {
    return static_cast<uint8_t>(std::clamp(value, 0, 255));
}

bool multiplicationExceeds(size_t left, size_t right, size_t limit) {
    return left != 0 && right > limit / left;
}

void copyI420ToRgb(const uint8_t* const planes[3], int width, int height, const int strides[2], uint8_t* destination);

absl::Status storeDecodedFrameInTensor(uint8_t* const planes[3], const SBufferInfo& output,
    size_t maxAllowedVideoPixels, size_t& totalAllocatedPixels, ov::Shape& frameShape,
    size_t& frameBytes, size_t selectedFrameLimit, size_t selectedFrames, ov::Tensor& video) {
    const int width = output.UsrData.sSystemBuffer.iWidth;
    const int height = output.UsrData.sSystemBuffer.iHeight;
    if (width <= 0 || height <= 0 || multiplicationExceeds(static_cast<size_t>(height), static_cast<size_t>(width), std::numeric_limits<size_t>::max())) {
        return absl::InvalidArgumentError("H.264 decoder returned an invalid frame size");
    }
    const size_t framePixels = static_cast<size_t>(height) * static_cast<size_t>(width);
    const ov::Shape decodedFrameShape{1, static_cast<size_t>(height), static_cast<size_t>(width), 3};
    if (framePixels > maxAllowedVideoPixels - totalAllocatedPixels) {
        return absl::InvalidArgumentError("Video exceeds maximum decoded size");
    }
    if (frameShape.empty()) {
        frameShape = decodedFrameShape;
        if (multiplicationExceeds(framePixels, 3, std::numeric_limits<size_t>::max())) {
            return absl::InvalidArgumentError("H.264 decoder returned an invalid frame size");
        }
        frameBytes = framePixels * 3;
        video = ov::Tensor(ov::element::u8, ov::Shape{selectedFrameLimit, frameShape[1], frameShape[2], frameShape[3]});
    } else if (decodedFrameShape != frameShape) {
        return absl::InvalidArgumentError("All video frames must have the same height, width and channel count");
    }
    auto* const destination = static_cast<uint8_t*>(video.data()) + selectedFrames * frameBytes;
    copyI420ToRgb(planes, width, height, output.UsrData.sSystemBuffer.iStride, destination);
    totalAllocatedPixels += framePixels;
    return absl::OkStatus();
}

void copyI420ToRgb(const uint8_t* const planes[3], int width, int height, const int strides[2], uint8_t* destination) {
    for (int row = 0; row < height; ++row) {
        const uint8_t* const y = planes[0] + row * strides[0];
        const uint8_t* const u = planes[1] + (row / 2) * strides[1];
        const uint8_t* const v = planes[2] + (row / 2) * strides[1];
        for (int column = 0; column < width; ++column) {
            const int luminance = static_cast<int>(y[column]) - 16;
            const int chromaU = static_cast<int>(u[column / 2]) - 128;
            const int chromaV = static_cast<int>(v[column / 2]) - 128;
            uint8_t* const rgb = destination + (static_cast<size_t>(row) * width + column) * 3;
            rgb[0] = clampToByte((298 * luminance + 409 * chromaV + 128) >> 8);
            rgb[1] = clampToByte((298 * luminance - 100 * chromaU - 208 * chromaV + 128) >> 8);
            rgb[2] = clampToByte((298 * luminance + 516 * chromaU + 128) >> 8);
        }
    }
}

}  // namespace

absl::StatusOr<ov::Tensor> loadVideoFrames(const std::vector<std::string>& frameSources,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains,
    size_t& totalAllocatedPixels) {
    if (frameSources.empty()) {
        return absl::InvalidArgumentError("Video must contain at least one frame");
    }
    if (static_cast<int64_t>(frameSources.size()) > MAX_VIDEO_FRAMES) {
        return absl::InvalidArgumentError("Number of video frames exceeds the allowed maximum of " + std::to_string(MAX_VIDEO_FRAMES));
    }
    const size_t numFrames = frameSources.size();

    const size_t maxAllowedImagePixels = request_validation_utils::getMaxImageDecodePixels();

    auto firstResult = fetchAndDecodeImage(frameSources[0], allowedLocalMediaPath, allowedMediaDomains, totalAllocatedPixels, maxAllowedImagePixels);
    if (!firstResult.ok()) {
        return firstResult.status();
    }
    // Move out of the StatusOr so resetting firstFrame below releases its buffer.
    ov::Tensor firstFrame = std::move(firstResult).value();
    const ov::Shape frameShape = firstFrame.get_shape();
    const size_t height = frameShape[1];
    const size_t width = frameShape[2];
    const size_t channels = frameShape[3];
    const size_t frameBytes = height * width * channels * firstFrame.get_element_type().size();

    // Reject against the remaining budget before allocating the stacked tensor;
    // division keeps the comparison overflow-safe.
    const size_t framePixels = height * width;
    const size_t remainingPixels = maxAllowedImagePixels - totalAllocatedPixels;
    if (numFrames > 1 && framePixels > remainingPixels / (numFrames - 1)) {
        return absl::InvalidArgumentError("Image exceeds maximum decoded size");
    }

    ov::Tensor video(firstFrame.get_element_type(), ov::Shape{numFrames, height, width, channels});
    auto* dst = static_cast<uint8_t*>(video.data());
    std::memcpy(dst, firstFrame.data(), frameBytes);
    firstFrame = ov::Tensor();  // release the first frame

    for (size_t i = 1; i < numFrames; i++) {
        auto frameResult = fetchAndDecodeImage(frameSources[i], allowedLocalMediaPath, allowedMediaDomains, totalAllocatedPixels, maxAllowedImagePixels);
        if (!frameResult.ok()) {
            return frameResult.status();
        }
        const ov::Tensor& frame = frameResult.value();
        // memcpy below relies on every frame being exactly frameBytes.
        if (frame.get_shape() != frameShape) {
            return absl::InvalidArgumentError("All video frames must have the same height, width and channel count");
        }
        std::memcpy(dst + i * frameBytes, frame.data(), frameBytes);
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Loaded video with {} frames of shape [{}, {}, {}]", numFrames, height, width, channels);
    return video;
}

absl::StatusOr<ov::Tensor> decodeVideoSource(const std::string& videoSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains,
    float requestedFps, uint32_t maxDecodedFrames, size_t& totalAllocatedPixels) {
    if (!std::isfinite(requestedFps) || requestedFps <= 0.0F || maxDecodedFrames == 0) {
        return absl::InvalidArgumentError("Requested video FPS and maximum frame count must be valid and greater than zero");
    }
    auto validationStatus = validateMediaSource(videoSource, allowedLocalMediaPath, allowedMediaDomains);
    if (!validationStatus.ok()) {
        return validationStatus;
    }

    const size_t maxFileSize = request_validation_utils::getMaxEncodedVideoFileSizeBytes();
    std::unique_ptr<std::FILE, FileCloser> input(nullptr);
    size_t fileSize = 0;
    if (isRemoteMediaSource(videoSource)) {
        input.reset(std::tmpfile());
        if (!input) {
            return absl::InternalError("Unable to create temporary video storage");
        }
        auto downloadStatus = downloadMediaToFile(videoSource, input.get(), maxFileSize, fileSize);
        if (!downloadStatus.ok()) {
            return downloadStatus;
        }
    } else {
        std::error_code fileSizeError;
        const uintmax_t localFileSize = std::filesystem::file_size(videoSource, fileSizeError);
        if (fileSizeError || localFileSize == 0) {
            return absl::InvalidArgumentError("Unable to read video source");
        }
        if (localFileSize > maxFileSize || localFileSize > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()) ||
            localFileSize > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
            return absl::InvalidArgumentError("Encoded video source exceeds the allowed file size");
        }
        fileSize = static_cast<size_t>(localFileSize);
        input.reset(std::fopen(videoSource.c_str(), "rb"));
        if (!input) {
            return absl::InvalidArgumentError("Unable to read video source");
        }
    }
    if (fileSize == 0 || fileSize > maxFileSize || fileSize > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return absl::InvalidArgumentError("Encoded video source exceeds the allowed file size");
    }
    if (std::fseek(input.get(), 0, SEEK_SET) != 0) {
        return absl::InvalidArgumentError("Unable to read video source");
    }
    EncodedVideoFile file{input.get(), fileSize};
    MP4D_demux_t mp4{};
    if (MP4D_open(&mp4, readVideoBytes, &file, static_cast<int64_t>(fileSize)) == 0) {
        return absl::InvalidArgumentError("Unable to parse MP4 video source");
    }
    Mp4DemuxGuard mp4Guard{mp4};

    unsigned trackIndex = mp4.track_count;
    for (unsigned index = 0; index < mp4.track_count; ++index) {
        if (mp4.track[index].handler_type == MP4D_HANDLER_TYPE_VIDE) {
            trackIndex = index;
            break;
        }
    }
    if (trackIndex == mp4.track_count) {
        return absl::InvalidArgumentError("MP4 source does not contain a video track");
    }

    std::vector<uint8_t> configuration;
    for (int index = 0, bytes = 0; const auto *sps = static_cast<const uint8_t*>(MP4D_read_sps(&mp4, trackIndex, index, &bytes)); ++index) {
        appendStartCode(configuration, sps, bytes);
    }
    for (int index = 0, bytes = 0; const auto *pps = static_cast<const uint8_t*>(MP4D_read_pps(&mp4, trackIndex, index, &bytes)); ++index) {
        appendStartCode(configuration, pps, bytes);
    }
    if (configuration.empty()) {
        return absl::InvalidArgumentError("MP4 video track is not H.264 AVC");
    }

    const auto& videoTrack = mp4.track[trackIndex];
    if (videoTrack.timescale == 0) {
        return absl::InvalidArgumentError("MP4 video track has an invalid time scale");
    }
    const uint64_t durationUnits = (static_cast<uint64_t>(videoTrack.duration_hi) << 32) | videoTrack.duration_lo;
    const long double requestedFrameCount = std::ceil(
        static_cast<long double>(durationUnits) * requestedFps / static_cast<long double>(videoTrack.timescale));
    if (!std::isfinite(requestedFrameCount) || requestedFrameCount < 1.0L ||
        requestedFrameCount > static_cast<long double>(std::numeric_limits<size_t>::max())) {
        return absl::InvalidArgumentError("MP4 video duration cannot be sampled at the requested FPS");
    }
    const size_t estimatedRequestedFrames = static_cast<size_t>(requestedFrameCount);
    if (estimatedRequestedFrames > maxDecodedFrames) {
        return absl::InvalidArgumentError("Requested video FPS exceeds maximum decoded frame count");
    }
    const size_t selectedFrameLimit = std::min({estimatedRequestedFrames, static_cast<size_t>(maxDecodedFrames),
        static_cast<size_t>(MAX_VIDEO_FRAMES), static_cast<size_t>(videoTrack.sample_count)});
    const size_t estimatedWidth = mp4.track[trackIndex].SampleDescription.video.width;
    const size_t estimatedHeight = mp4.track[trackIndex].SampleDescription.video.height;
    const size_t maxAllowedVideoPixels = request_validation_utils::getMaxVideoDecodePixels();
    if (selectedFrameLimit == 0 || estimatedWidth == 0 || estimatedHeight == 0 ||
        multiplicationExceeds(estimatedWidth, estimatedHeight, std::numeric_limits<size_t>::max())) {
        return absl::InvalidArgumentError("MP4 source has invalid video dimensions");
    }
    const size_t estimatedFramePixels = estimatedWidth * estimatedHeight;
    if (totalAllocatedPixels > maxAllowedVideoPixels ||
        multiplicationExceeds(estimatedFramePixels, selectedFrameLimit, maxAllowedVideoPixels - totalAllocatedPixels)) {
        return absl::InvalidArgumentError("Video exceeds maximum decoded size");
    }

    OpenH264DecoderGuard decoderGuard;
    if (WelsCreateDecoder(&decoderGuard.decoder) != 0 || decoderGuard.decoder == nullptr) {
        return absl::InternalError("Unable to create H.264 decoder");
    }
    SDecodingParam parameters{};
    parameters.sVideoProperty.size = sizeof(SVideoProperty);
    parameters.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC;
    if (decoderGuard.decoder->Initialize(&parameters) != 0) {
        return absl::InternalError("Unable to initialize H.264 decoder");
    }
    decoderGuard.initialized = true;

    ov::Shape frameShape;
    size_t frameBytes = 0;
    size_t selectedFrames = 0;
    ov::Tensor video;
    const long double frameInterval = static_cast<long double>(videoTrack.timescale) / requestedFps;
    long double nextSelectedTimestamp = 0.0L;
    absl::Status decodeStatus = absl::OkStatus();
    for (unsigned sampleIndex = 0; sampleIndex < mp4.track[trackIndex].sample_count; ++sampleIndex) {
        unsigned sampleBytes = 0;
        unsigned timestamp = 0;
        unsigned duration = 0;
        const MP4D_file_offset_t offset = MP4D_frame_offset(&mp4, trackIndex, sampleIndex, &sampleBytes, &timestamp, &duration);
        if (offset < 0 || static_cast<uint64_t>(offset) > file.size || sampleBytes > file.size - static_cast<size_t>(offset) ||
            sampleBytes > static_cast<unsigned>(std::numeric_limits<int>::max()) - configuration.size()) {
            decodeStatus = absl::InvalidArgumentError("MP4 source has an invalid video sample");
            break;
        }
        std::vector<uint8_t> annexB;
        if (sampleIndex == 0) {
            annexB = configuration;
        }
        std::vector<uint8_t> sample(sampleBytes);
        if (readVideoBytes(offset, sample.data(), sample.size(), &file) != 0 ||
            !appendLengthPrefixedSample(annexB, sample.data(), sample.size(), videoTrack.nal_length_size)) {
            decodeStatus = absl::InvalidArgumentError("MP4 source has an invalid H.264 sample");
            break;
        }

        uint8_t* planes[3]{};
        SBufferInfo output{};
        if (decoderGuard.decoder->DecodeFrameNoDelay(annexB.data(), static_cast<int>(annexB.size()), planes, &output) != dsErrorFree) {
            decodeStatus = absl::InvalidArgumentError("Unable to decode H.264 video frame");
            break;
        }
        if (output.iBufferStatus != 1) {
            continue;
        }
        // TODO: Sample using presentation timestamps after preserving MP4 ctts offsets
        // and passing timestamps through a timestamp-aware OpenH264 decoder API.
        if (selectedFrames == maxDecodedFrames || static_cast<long double>(timestamp) < nextSelectedTimestamp) {
            continue;
        }
        nextSelectedTimestamp = (std::floor(static_cast<long double>(timestamp) / frameInterval) + 1.0L) * frameInterval;

        decodeStatus = storeDecodedFrameInTensor(planes, output, maxAllowedVideoPixels, totalAllocatedPixels,
            frameShape, frameBytes, selectedFrameLimit, selectedFrames, video);
        if (!decodeStatus.ok()) {
            break;
        }
        selectedFrames++;
        if (selectedFrames == selectedFrameLimit) {
            break;
        }
    }

    while (decodeStatus.ok() && selectedFrames < selectedFrameLimit) {
        uint8_t* planes[3]{};
        SBufferInfo output{};
        if (decoderGuard.decoder->FlushFrame(planes, &output) != dsErrorFree || output.iBufferStatus != 1) {
            break;
        }
        decodeStatus = storeDecodedFrameInTensor(planes, output, maxAllowedVideoPixels, totalAllocatedPixels,
            frameShape, frameBytes, selectedFrameLimit, selectedFrames, video);
        if (!decodeStatus.ok()) {
            break;
        }
        selectedFrames++;
    }

    if (!decodeStatus.ok()) {
        return decodeStatus;
    }
    if (selectedFrames == 0) {
        return absl::InvalidArgumentError("Video must contain at least one decodable frame");
    }
    video.set_shape(ov::Shape{selectedFrames, frameShape[1], frameShape[2], frameShape[3]});
    return video;
}

}  // namespace ovms
