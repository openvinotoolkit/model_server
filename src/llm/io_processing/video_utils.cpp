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
#pragma warning(push)
#pragma warning(disable : 6246)
#include <minimp4.h>
#pragma warning(pop)
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

struct OpenedVideoSource {
    std::unique_ptr<std::FILE, FileCloser> input;
    size_t fileSize;
};

struct VideoSamplingPlan {
    size_t selectedFrameLimit;
    size_t maxAllowedPixels;
    std::optional<long double> frameInterval;
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

class H264BitReader {
public:
    explicit H264BitReader(const std::vector<uint8_t>& bytes) : bytes(bytes) {
    }

    bool readBit(uint32_t& value) {
        if (bitOffset >= bytes.size() * 8) {
            return false;
        }
        value = (bytes[bitOffset / 8] >> (7 - bitOffset % 8)) & 1;
        bitOffset++;
        return true;
    }

    bool readBits(uint32_t count, uint32_t& value) {
        if (count > 32 || count > bytes.size() * 8 - bitOffset) {
            return false;
        }
        value = 0;
        for (uint32_t bitIndex = 0; bitIndex < count; ++bitIndex) {
            uint32_t bit = 0;
            if (!readBit(bit)) {
                return false;
            }
            value = (value << 1) | bit;
        }
        return true;
    }

    bool readUnsignedExpGolomb(uint32_t& value) {
        uint32_t leadingZeroBits = 0;
        uint32_t bit = 0;
        while (true) {
            if (!readBit(bit) || leadingZeroBits == 32) {
                return false;
            }
            if (bit != 0) {
                break;
            }
            leadingZeroBits++;
        }
        uint32_t suffix = 0;
        if (!readBits(leadingZeroBits, suffix)) {
            return false;
        }
        value = ((uint32_t{1} << leadingZeroBits) - 1) + suffix;
        return true;
    }

    bool readSignedExpGolomb(int32_t& value) {
        uint32_t unsignedValue = 0;
        if (!readUnsignedExpGolomb(unsignedValue) || unsignedValue == std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        value = unsignedValue % 2 == 0 ? -static_cast<int32_t>(unsignedValue / 2) : static_cast<int32_t>((unsignedValue + 1) / 2);
        return true;
    }

private:
    const std::vector<uint8_t>& bytes;
    size_t bitOffset = 0;
};

bool skipH264ScalingList(H264BitReader& reader, uint32_t size) {
    int32_t lastScale = 8;
    int32_t nextScale = 8;
    for (uint32_t index = 0; index < size; ++index) {
        if (nextScale != 0) {
            int32_t deltaScale = 0;
            if (!reader.readSignedExpGolomb(deltaScale)) {
                return false;
            }
            nextScale = (lastScale + deltaScale + 256) % 256;
        }
        lastScale = nextScale == 0 ? lastScale : nextScale;
    }
    return true;
}

bool isHighProfile(uint32_t profileIdc) {
    return profileIdc == 44 || profileIdc == 83 || profileIdc == 86 || profileIdc == 100 || profileIdc == 110 ||
           profileIdc == 118 || profileIdc == 122 || profileIdc == 128 || profileIdc == 134 || profileIdc == 135 ||
           profileIdc == 138 || profileIdc == 139 || profileIdc == 244;
}

absl::Status validateH264SpsDimensions(const uint8_t* nal, size_t nalSize, size_t maxAllowedPixels) {
    if (nalSize < 2 || (nal[0] & 0x1f) != 7) {
        return absl::InvalidArgumentError("Invalid H.264 SPS");
    }

    std::vector<uint8_t> rbsp;
    rbsp.reserve(nalSize - 1);
    uint32_t consecutiveZeroBytes = 0;
    for (size_t index = 1; index < nalSize; ++index) {
        const uint8_t byte = nal[index];
        if (consecutiveZeroBytes >= 2 && byte == 3) {
            consecutiveZeroBytes = 0;
            continue;
        }
        rbsp.push_back(byte);
        consecutiveZeroBytes = byte == 0 ? consecutiveZeroBytes + 1 : 0;
    }

    H264BitReader reader(rbsp);
    uint32_t profileIdc = 0;
    uint32_t ignored = 0;
    uint32_t chromaFormatIdc = 1;
    uint32_t separateColourPlaneFlag = 0;
    if (!reader.readBits(8, profileIdc) || !reader.readBits(8, ignored) || !reader.readBits(8, ignored) ||
        !reader.readUnsignedExpGolomb(ignored)) {
        return absl::InvalidArgumentError("Invalid H.264 SPS");
    }
    if (isHighProfile(profileIdc)) {
        if (!reader.readUnsignedExpGolomb(chromaFormatIdc) || chromaFormatIdc > 3) {
            return absl::InvalidArgumentError("Invalid H.264 SPS");
        }
        if (chromaFormatIdc == 3 && !reader.readBit(separateColourPlaneFlag)) {
            return absl::InvalidArgumentError("Invalid H.264 SPS");
        }
        uint32_t scalingMatrixPresent = 0;
        if (!reader.readUnsignedExpGolomb(ignored) || !reader.readUnsignedExpGolomb(ignored) || !reader.readBit(ignored) ||
            !reader.readBit(scalingMatrixPresent)) {
            return absl::InvalidArgumentError("Invalid H.264 SPS");
        }
        if (scalingMatrixPresent != 0) {
            const uint32_t scalingListCount = chromaFormatIdc == 3 ? 12 : 8;
            for (uint32_t index = 0; index < scalingListCount; ++index) {
                uint32_t scalingListPresent = 0;
                if (!reader.readBit(scalingListPresent) || (scalingListPresent != 0 && !skipH264ScalingList(reader, index < 6 ? 16 : 64))) {
                    return absl::InvalidArgumentError("Invalid H.264 SPS");
                }
            }
        }
    }

    uint32_t pictureOrderCountType = 0;
    if (!reader.readUnsignedExpGolomb(ignored) || !reader.readUnsignedExpGolomb(pictureOrderCountType)) {
        return absl::InvalidArgumentError("Invalid H.264 SPS");
    }
    if (pictureOrderCountType == 0) {
        if (!reader.readUnsignedExpGolomb(ignored)) {
            return absl::InvalidArgumentError("Invalid H.264 SPS");
        }
    } else if (pictureOrderCountType == 1) {
        uint32_t cycleLength = 0;
        int32_t signedValue = 0;
        if (!reader.readBit(ignored) || !reader.readSignedExpGolomb(signedValue) || !reader.readSignedExpGolomb(signedValue) ||
            !reader.readUnsignedExpGolomb(cycleLength)) {
            return absl::InvalidArgumentError("Invalid H.264 SPS");
        }
        for (uint32_t index = 0; index < cycleLength; ++index) {
            if (!reader.readSignedExpGolomb(signedValue)) {
                return absl::InvalidArgumentError("Invalid H.264 SPS");
            }
        }
    } else if (pictureOrderCountType > 2) {
        return absl::InvalidArgumentError("Invalid H.264 SPS");
    }

    uint32_t widthInMbsMinusOne = 0;
    uint32_t heightInMapUnitsMinusOne = 0;
    uint32_t frameMbsOnlyFlag = 0;
    uint32_t frameCroppingFlag = 0;
    if (!reader.readUnsignedExpGolomb(ignored) || !reader.readBit(ignored) || !reader.readUnsignedExpGolomb(widthInMbsMinusOne) ||
        !reader.readUnsignedExpGolomb(heightInMapUnitsMinusOne) || !reader.readBit(frameMbsOnlyFlag) ||
        (frameMbsOnlyFlag == 0 && !reader.readBit(ignored)) || !reader.readBit(ignored) || !reader.readBit(frameCroppingFlag)) {
        return absl::InvalidArgumentError("Invalid H.264 SPS");
    }

    uint32_t cropLeft = 0;
    uint32_t cropRight = 0;
    uint32_t cropTop = 0;
    uint32_t cropBottom = 0;
    if (frameCroppingFlag != 0 && (!reader.readUnsignedExpGolomb(cropLeft) || !reader.readUnsignedExpGolomb(cropRight) ||
                                  !reader.readUnsignedExpGolomb(cropTop) || !reader.readUnsignedExpGolomb(cropBottom))) {
        return absl::InvalidArgumentError("Invalid H.264 SPS");
    }

    const uint32_t chromaArrayType = separateColourPlaneFlag != 0 ? 0 : chromaFormatIdc;
    const size_t cropUnitX = chromaArrayType == 0 || chromaArrayType == 3 ? 1 : 2;
    const size_t cropUnitY = (chromaArrayType == 0 ? 1 : chromaArrayType == 1 ? 2 : 1) * (2 - frameMbsOnlyFlag);
    const size_t codedWidth = (static_cast<size_t>(widthInMbsMinusOne) + 1) * 16;
    const size_t codedHeight = (static_cast<size_t>(heightInMapUnitsMinusOne) + 1) * 16 * (2 - frameMbsOnlyFlag);
    const size_t croppedWidth = (static_cast<size_t>(cropLeft) + cropRight) * cropUnitX;
    const size_t croppedHeight = (static_cast<size_t>(cropTop) + cropBottom) * cropUnitY;
    if (croppedWidth >= codedWidth || croppedHeight >= codedHeight) {
        return absl::InvalidArgumentError("Invalid H.264 SPS");
    }
    const size_t width = codedWidth - croppedWidth;
    const size_t height = codedHeight - croppedHeight;
    if (multiplicationExceeds(width, height, maxAllowedPixels)) {
        return absl::InvalidArgumentError("Video exceeds maximum decoded size");
    }
    return absl::OkStatus();
}

absl::Status validateH264SpsDimensions(const std::vector<uint8_t>& annexB, size_t maxAllowedPixels) {
    size_t start = 0;
    while (start < annexB.size()) {
        if (annexB.size() - start < 5 || std::memcmp(annexB.data() + start, "\0\0\0\1", 4) != 0) {
            return absl::InvalidArgumentError("Invalid H.264 sample");
        }
        const size_t nalStart = start + 4;
        size_t nextStart = nalStart;
        while (nextStart + 4 <= annexB.size() && std::memcmp(annexB.data() + nextStart, "\0\0\0\1", 4) != 0) {
            ++nextStart;
        }
        const size_t nalEnd = nextStart + 4 <= annexB.size() ? nextStart : annexB.size();
        if ((annexB[nalStart] & 0x1f) == 7) {
            auto status = validateH264SpsDimensions(annexB.data() + nalStart, nalEnd - nalStart, maxAllowedPixels);
            if (!status.ok()) {
                return status;
            }
        }
        start = nalEnd == annexB.size() ? annexB.size() : nextStart;
    }
    return absl::OkStatus();
}

}  // namespace

bool videoFrameAllocationFitsBudget(size_t framePixels, size_t frameCount,
    size_t totalAllocatedPixels, size_t maxAllowedPixels) {
    return totalAllocatedPixels <= maxAllowedPixels &&
           !multiplicationExceeds(framePixels, frameCount, maxAllowedPixels - totalAllocatedPixels);
}

absl::StatusOr<size_t> calculateSelectedVideoFrameCount(uint64_t durationUnits,
    uint32_t timescale, uint32_t sampleCount, std::optional<float> requestedFps, uint32_t maxDecodedFrames) {
    if (timescale == 0) {
        return absl::InvalidArgumentError("MP4 video track has an invalid time scale");
    }
    if (!requestedFps.has_value()) {
        return std::min({static_cast<size_t>(sampleCount), static_cast<size_t>(maxDecodedFrames),
            static_cast<size_t>(MAX_VIDEO_FRAMES)});
    }
    const float fps = requestedFps.value();
    const long double requestedFrameCount = std::ceil(
        static_cast<long double>(durationUnits) * fps / static_cast<long double>(timescale));
    if (!std::isfinite(requestedFrameCount) || requestedFrameCount < 1.0L ||
        requestedFrameCount > static_cast<long double>(std::numeric_limits<size_t>::max())) {
        return absl::InvalidArgumentError("MP4 video duration cannot be sampled at the requested FPS");
    }
    const size_t estimatedRequestedFrames = static_cast<size_t>(requestedFrameCount);
    if (estimatedRequestedFrames > maxDecodedFrames) {
        return absl::InvalidArgumentError("Requested video FPS exceeds maximum decoded frame count");
    }
    return std::min({estimatedRequestedFrames, static_cast<size_t>(maxDecodedFrames),
        static_cast<size_t>(MAX_VIDEO_FRAMES), static_cast<size_t>(sampleCount)});
}

VideoFrameSelector::VideoFrameSelector(std::optional<long double> frameInterval, size_t maxSelectedFrames) :
    frameInterval(frameInterval),
    maxSelectedFrames(maxSelectedFrames) {
}

bool VideoFrameSelector::shouldSelect(uint64_t timestamp) {
    if (selectedFrameCount >= maxSelectedFrames) {
        return false;
    }
    if (frameInterval.has_value()) {
        if (static_cast<long double>(timestamp) < nextSelectedTimestamp) {
            return false;
        }
        nextSelectedTimestamp = (std::floor(static_cast<long double>(timestamp) / frameInterval.value()) + 1.0L) * frameInterval.value();
    }
    selectedFrameCount++;
    return true;
}

namespace {

void copyI420ToRgb(const uint8_t* const planes[3], int width, int height, const int strides[2], uint8_t* destination);

absl::StatusOr<OpenedVideoSource> openVideoSource(const std::string& videoSource, size_t maxFileSize) {
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
    return OpenedVideoSource{std::move(input), fileSize};
}

absl::StatusOr<unsigned> findVideoTrackIndex(const MP4D_demux_t& mp4) {
    for (unsigned index = 0; index < mp4.track_count; ++index) {
        if (mp4.track[index].handler_type == MP4D_HANDLER_TYPE_VIDE) {
            return index;
        }
    }
    return absl::InvalidArgumentError("MP4 source does not contain a video track");
}

absl::StatusOr<std::vector<uint8_t>> readH264ParameterSets(MP4D_demux_t& mp4, unsigned videoTrackIndex) {
    std::vector<uint8_t> parameterSets;
    for (int index = 0, bytes = 0; const auto *sps = static_cast<const uint8_t*>(MP4D_read_sps(&mp4, videoTrackIndex, index, &bytes)); ++index) {
        appendStartCode(parameterSets, sps, bytes);
    }
    for (int index = 0, bytes = 0; const auto *pps = static_cast<const uint8_t*>(MP4D_read_pps(&mp4, videoTrackIndex, index, &bytes)); ++index) {
        appendStartCode(parameterSets, pps, bytes);
    }
    if (parameterSets.empty()) {
        return absl::InvalidArgumentError("MP4 video track is not H.264 AVC");
    }
    return parameterSets;
}

absl::StatusOr<VideoSamplingPlan> createVideoSamplingPlan(const MP4D_demux_t& mp4, unsigned videoTrackIndex,
    std::optional<float> requestedFps, uint32_t maxDecodedFrames, size_t totalAllocatedPixels) {
    const auto& videoTrack = mp4.track[videoTrackIndex];
    const uint64_t durationUnits = (static_cast<uint64_t>(videoTrack.duration_hi) << 32) | videoTrack.duration_lo;
    auto selectedFrameLimitResult = calculateSelectedVideoFrameCount(durationUnits, videoTrack.timescale,
        videoTrack.sample_count, requestedFps, maxDecodedFrames);
    if (!selectedFrameLimitResult.ok()) {
        return selectedFrameLimitResult.status();
    }
    const size_t selectedFrameLimit = selectedFrameLimitResult.value();
    const size_t estimatedWidth = videoTrack.SampleDescription.video.width;
    const size_t estimatedHeight = videoTrack.SampleDescription.video.height;
    const size_t maxAllowedPixels = request_validation_utils::getMaxVideoDecodePixels();
    if (selectedFrameLimit == 0 || estimatedWidth == 0 || estimatedHeight == 0 ||
        multiplicationExceeds(estimatedWidth, estimatedHeight, std::numeric_limits<size_t>::max())) {
        return absl::InvalidArgumentError("MP4 source has invalid video dimensions");
    }
    const size_t estimatedFramePixels = estimatedWidth * estimatedHeight;
    if (totalAllocatedPixels > maxAllowedPixels ||
        multiplicationExceeds(estimatedFramePixels, selectedFrameLimit, maxAllowedPixels - totalAllocatedPixels)) {
        return absl::InvalidArgumentError("Video exceeds maximum decoded size");
    }
    const std::optional<long double> frameInterval = requestedFps.has_value()
                                                         ? std::optional<long double>{static_cast<long double>(videoTrack.timescale) / requestedFps.value()}
                                                         : std::nullopt;
    return VideoSamplingPlan{selectedFrameLimit, maxAllowedPixels, frameInterval};
}

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
    if (!videoFrameAllocationFitsBudget(framePixels, selectedFrameLimit - selectedFrames,
            totalAllocatedPixels, maxAllowedVideoPixels)) {
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

absl::StatusOr<ov::Tensor> decodeH264Video(MP4D_demux_t& mp4, unsigned videoTrackIndex,
    EncodedVideoFile& file, const std::vector<uint8_t>& parameterSets,
    const VideoSamplingPlan& samplingPlan, size_t& totalAllocatedPixels) {
    const auto& videoTrack = mp4.track[videoTrackIndex];
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
    VideoFrameSelector frameSelector(samplingPlan.frameInterval, samplingPlan.selectedFrameLimit);
    absl::Status decodeStatus = absl::OkStatus();
    for (unsigned sampleIndex = 0; sampleIndex < videoTrack.sample_count; ++sampleIndex) {
        unsigned sampleBytes = 0;
        unsigned timestamp = 0;
        unsigned duration = 0;
        const MP4D_file_offset_t offset = MP4D_frame_offset(&mp4, videoTrackIndex, sampleIndex, &sampleBytes, &timestamp, &duration);
        if (offset < 0 || static_cast<uint64_t>(offset) > file.size || sampleBytes > file.size - static_cast<size_t>(offset) ||
            sampleBytes > static_cast<unsigned>(std::numeric_limits<int>::max()) - parameterSets.size()) {
            decodeStatus = absl::InvalidArgumentError("MP4 source has an invalid video sample");
            break;
        }
        std::vector<uint8_t> annexB;
        if (sampleIndex == 0) {
            annexB = parameterSets;
        }
        std::vector<uint8_t> sample(sampleBytes);
        if (readVideoBytes(offset, sample.data(), sample.size(), &file) != 0 ||
            !appendLengthPrefixedSample(annexB, sample.data(), sample.size(), videoTrack.nal_length_size)) {
            decodeStatus = absl::InvalidArgumentError("MP4 source has an invalid H.264 sample");
            break;
        }
        decodeStatus = validateH264SpsDimensions(annexB, samplingPlan.maxAllowedPixels);
        if (!decodeStatus.ok()) {
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
        if (!frameSelector.shouldSelect(timestamp)) {
            continue;
        }

        decodeStatus = storeDecodedFrameInTensor(planes, output, samplingPlan.maxAllowedPixels, totalAllocatedPixels,
            frameShape, frameBytes, samplingPlan.selectedFrameLimit, selectedFrames, video);
        if (!decodeStatus.ok()) {
            break;
        }
        selectedFrames++;
        if (selectedFrames == samplingPlan.selectedFrameLimit) {
            break;
        }
    }

    while (decodeStatus.ok() && selectedFrames < samplingPlan.selectedFrameLimit) {
        uint8_t* planes[3]{};
        SBufferInfo output{};
        if (decoderGuard.decoder->FlushFrame(planes, &output) != dsErrorFree || output.iBufferStatus != 1) {
            break;
        }
        decodeStatus = storeDecodedFrameInTensor(planes, output, samplingPlan.maxAllowedPixels, totalAllocatedPixels,
            frameShape, frameBytes, samplingPlan.selectedFrameLimit, selectedFrames, video);
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
    std::optional<float> requestedFps, uint32_t maxDecodedFrames, size_t& totalAllocatedPixels) {
    if ((requestedFps.has_value() && (!std::isfinite(requestedFps.value()) || requestedFps.value() <= 0.0F)) || maxDecodedFrames == 0) {
        return absl::InvalidArgumentError("Requested video FPS and maximum frame count must be valid and greater than zero");
    }
    auto validationStatus = validateMediaSource(videoSource, allowedLocalMediaPath, allowedMediaDomains);
    if (!validationStatus.ok()) {
        return validationStatus;
    }

    auto sourceResult = openVideoSource(videoSource, request_validation_utils::getMaxEncodedVideoFileSizeBytes());
    if (!sourceResult.ok()) {
        return sourceResult.status();
    }
    OpenedVideoSource source = std::move(sourceResult).value();
    EncodedVideoFile file{source.input.get(), source.fileSize};
    MP4D_demux_t mp4{};
    if (MP4D_open(&mp4, readVideoBytes, &file, static_cast<int64_t>(source.fileSize)) == 0) {
        return absl::InvalidArgumentError("Unable to parse MP4 video source");
    }
    Mp4DemuxGuard mp4Guard{mp4};

    auto videoTrackIndexResult = findVideoTrackIndex(mp4);
    if (!videoTrackIndexResult.ok()) {
        return videoTrackIndexResult.status();
    }
    const unsigned videoTrackIndex = videoTrackIndexResult.value();
    auto parameterSetsResult = readH264ParameterSets(mp4, videoTrackIndex);
    if (!parameterSetsResult.ok()) {
        return parameterSetsResult.status();
    }
    auto samplingPlanResult = createVideoSamplingPlan(mp4, videoTrackIndex, requestedFps, maxDecodedFrames, totalAllocatedPixels);
    if (!samplingPlanResult.ok()) {
        return samplingPlanResult.status();
    }
    return decodeH264Video(mp4, videoTrackIndex, file, parameterSetsResult.value(), samplingPlanResult.value(), totalAllocatedPixels);
}

}  // namespace ovms
