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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <curl/curl.h>

#include "image_utils.hpp"
#include "src/filesystem/filesystem.hpp"
#include "../../logging.hpp"
#include "../../predict_request_validation_utils_impl.hpp"
#include "src/mpi/intel_mpi.h"

namespace ovms {

namespace {

constexpr int64_t MAX_REMOTE_VIDEO_SIZE_BYTES = 20000000;

struct VideoDownloadContext {
    std::string* data;
    size_t sizeLimit;
    size_t bytesReceived = 0;
    bool sizeLimitExceeded = false;
    bool allocationFailed = false;

    size_t append(const void* downloadedChunk, size_t size, size_t nmemb) noexcept {
        if (size != 0 && nmemb > std::numeric_limits<size_t>::max() / size) {
            sizeLimitExceeded = true;
            return 0;
        }
        const size_t realSize = size * nmemb;
        if (bytesReceived > sizeLimit || realSize > sizeLimit - bytesReceived) {
            sizeLimitExceeded = true;
            return 0;
        }
        try {
            data->append(static_cast<const char*>(downloadedChunk), realSize);
            bytesReceived += realSize;
        } catch (...) {
            allocationFailed = true;
            return 0;
        }
        return realSize;
    }
};

class TemporaryVideoDirectory {
public:
    ~TemporaryVideoDirectory() {
        if (!path.empty()) {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    }

    absl::Status create() {
        std::string temporaryPath;
        if (FileSystem::createTempPath(&temporaryPath) != StatusCode::OK) {
            return absl::InternalError("Failed to create temporary video directory");
        }
        path = std::move(temporaryPath);
        return absl::OkStatus();
    }

    std::filesystem::path videoPath() const {
        return path / "video.mp4";
    }

private:
    std::filesystem::path path;
};

bool isPathInsideDirectory(const std::filesystem::path& testedPath, const std::filesystem::path& allowedDirectory) {
    const auto mismatch = std::mismatch(
        allowedDirectory.begin(), allowedDirectory.end(),
        testedPath.begin(), testedPath.end());
    return mismatch.first == allowedDirectory.end();
}

absl::StatusOr<std::string> resolveLocalVideoPath(const std::string& videoSource,
    const std::optional<std::string>& allowedLocalMediaPath) {
    if (!allowedLocalMediaPath.has_value()) {
        return absl::InvalidArgumentError("Loading videos from local filesystem is disabled.");
    }
    if (FileSystem::isPathEscaped(videoSource)) {
        return absl::InvalidArgumentError("Video path escape with .. is forbidden.");
    }

    const std::filesystem::path resolvedAllowedPath = FileSystem::normalizeConfiguredPath(allowedLocalMediaPath.value());
    const std::string resolvedVideoPath = FileSystem::normalizeConfiguredPath(videoSource);
    if (!isPathInsideDirectory(resolvedVideoPath, resolvedAllowedPath)) {
        return absl::InvalidArgumentError("Given filepath is not subpath of allowed_local_media_path");
    }
    return resolvedVideoPath;
}

bool isRemoteVideoUrl(const std::string& source) {
    return source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0;
}

bool isDomainAllowed(const std::vector<std::string>& allowedDomains, const std::string& url) {
    if (allowedDomains.size() == 1 && allowedDomains[0] == "all") {
        return true;
    }
    CURLU* parsedUrl = curl_url();
    if (parsedUrl == nullptr) {
        return false;
    }
    const CURLUcode setResult = curl_url_set(parsedUrl, CURLUPART_URL, url.c_str(), 0);
    char* host = nullptr;
    const CURLUcode getResult = setResult == CURLUE_OK ? curl_url_get(parsedUrl, CURLUPART_HOST, &host, 0) : CURLUE_BAD_HANDLE;
    if (getResult != CURLUE_OK) {
        curl_url_cleanup(parsedUrl);
        return false;
    }
    const bool allowed = std::find(allowedDomains.begin(), allowedDomains.end(), host) != allowedDomains.end();
    curl_free(host);
    curl_url_cleanup(parsedUrl);
    return allowed;
}

size_t appendVideoChunk(void* downloadedChunk, size_t size, size_t nmemb, void* context) noexcept {
    return static_cast<VideoDownloadContext*>(context)->append(downloadedChunk, size, nmemb);
}

absl::Status downloadRemoteVideo(const std::string& url, const std::filesystem::path& destination) {
    CURL* handle = curl_easy_init();
    if (handle == nullptr) {
        return absl::InternalError("Video download initialization failed");
    }
    auto handleGuard = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(handle, curl_easy_cleanup);
    std::string videoData;
    VideoDownloadContext context{&videoData, static_cast<size_t>(MAX_REMOTE_VIDEO_SIZE_BYTES)};

    CURLcode status = curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    if (status == CURLE_OK) {
        status = curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, appendVideoChunk);
    }
    if (status == CURLE_OK) {
        status = curl_easy_setopt(handle, CURLOPT_WRITEDATA, &context);
    }
    if (status == CURLE_OK) {
        status = curl_easy_setopt(handle, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
    }
    if (std::getenv("OVMS_MEDIA_URL_ALLOW_REDIRECTS") != nullptr &&
        std::strcmp(std::getenv("OVMS_MEDIA_URL_ALLOW_REDIRECTS"), "1") == 0 && status == CURLE_OK) {
        status = curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
    }
    if (status == CURLE_OK) {
        status = curl_easy_setopt(handle, CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(MAX_REMOTE_VIDEO_SIZE_BYTES));
    }
    if (status != CURLE_OK) {
        return absl::InternalError("Video download configuration failed");
    }
    status = curl_easy_perform(handle);
    if (status != CURLE_OK) {
        if (context.sizeLimitExceeded || status == CURLE_FILESIZE_EXCEEDED) {
            return absl::InvalidArgumentError("Remote video exceeds maximum size");
        }
        if (context.allocationFailed) {
            return absl::ResourceExhaustedError("Memory allocation failed while downloading video");
        }
        return absl::InvalidArgumentError("Video download failed");
    }

    std::ofstream file(destination, std::ios::binary);
    if (!file) {
        return absl::InternalError("Failed to create temporary video file");
    }
    file.write(videoData.data(), static_cast<std::streamsize>(videoData.size()));
    if (!file) {
        return absl::InternalError("Failed to write temporary video file");
    }
    return absl::OkStatus();
}

uint8_t clampToByte(int value) {
    return static_cast<uint8_t>(std::clamp(value, 0, 255));
}

void convertNv12ToRgb(const uint8_t* yPlane, const uint8_t* uvPlane,
    size_t width, size_t height, std::vector<uint8_t>& rgb) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const int luminance = std::max(0, static_cast<int>(yPlane[y * width + x]) - 16);
            const size_t uvOffset = (y / 2) * width + (x & ~static_cast<size_t>(1));
            const int chromaBlue = static_cast<int>(uvPlane[uvOffset]) - 128;
            const int chromaRed = static_cast<int>(uvPlane[uvOffset + 1]) - 128;
            const size_t rgbOffset = (y * width + x) * 3;
            rgb[rgbOffset] = clampToByte((298 * luminance + 409 * chromaRed + 128) >> 8);
            rgb[rgbOffset + 1] = clampToByte((298 * luminance - 100 * chromaBlue - 208 * chromaRed + 128) >> 8);
            rgb[rgbOffset + 2] = clampToByte((298 * luminance + 516 * chromaBlue + 128) >> 8);
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

absl::StatusOr<ov::Tensor> loadEncodedVideo(const std::string& videoSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains,
    size_t& totalAllocatedPixels) {
    std::string sourcePath;
    TemporaryVideoDirectory temporaryDirectory;
    if (isRemoteVideoUrl(videoSource)) {
        if (!allowedMediaDomains.has_value() || !isDomainAllowed(allowedMediaDomains.value(), videoSource)) {
            return absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains");
        }
        const auto temporaryDirectoryStatus = temporaryDirectory.create();
        if (!temporaryDirectoryStatus.ok()) {
            return temporaryDirectoryStatus;
        }
        const auto downloadStatus = downloadRemoteVideo(videoSource, temporaryDirectory.videoPath());
        if (!downloadStatus.ok()) {
            return downloadStatus;
        }
        sourcePath = temporaryDirectory.videoPath().string();
    } else {
        auto resolvedPath = resolveLocalVideoPath(videoSource, allowedLocalMediaPath);
        if (!resolvedPath.ok()) {
            return resolvedPath.status();
        }
        sourcePath = std::move(resolvedPath).value();
    }

    imp_context_t* rawContext = nullptr;
    if (imp_context_create(&rawContext, nullptr) != IMP_OK) {
        return absl::InternalError("Failed to create video decode context");
    }
    std::unique_ptr<imp_context_t, decltype(&imp_context_destroy)> context(rawContext, imp_context_destroy);

    imp_video_source_t* rawSource = nullptr;
    if (imp_video_source_create(&rawSource, IMP_SOURCE_FILE) != IMP_OK || rawSource == nullptr) {
        return absl::InternalError("Failed to create video source");
    }
    std::unique_ptr<imp_video_source_t, decltype(&imp_video_source_destroy)> source(rawSource, imp_video_source_destroy);
    if (imp_video_source_set(source.get(), "path", sourcePath.c_str()) != IMP_OK) {
        return absl::InvalidArgumentError("Failed to configure video source");
    }

    imp_video_decode_opts_t options{};
    options.output_memory = IMP_VIDEO_MEMORY_SYSTEM;
    imp_video_stream_t* rawStream = nullptr;
    const imp_status_t openStatus = imp_video_open(&rawStream, source.get(), context.get(), &options);
    if (openStatus != IMP_OK || rawStream == nullptr) {
        return absl::InvalidArgumentError(std::string("Unable to open video source: ") + imp_context_get_error(context.get()));
    }
    std::unique_ptr<imp_video_stream_t, decltype(&imp_video_close)> stream(rawStream, imp_video_close);

    const size_t maxAllowedPixels = request_validation_utils::getMaxImageDecodePixels();
    std::vector<std::vector<uint8_t>> rgbFrames;
    rgbFrames.reserve(MAX_ENCODED_VIDEO_FRAMES);
    size_t width = 0;
    size_t height = 0;

    while (rgbFrames.size() < static_cast<size_t>(MAX_ENCODED_VIDEO_FRAMES)) {
        imp_tensor_t* rawFrame = nullptr;
        const imp_status_t readStatus = imp_video_read_frame(&rawFrame, stream.get(), 0);
        std::unique_ptr<imp_tensor_t, decltype(&imp_tensor_release)> frame(rawFrame, imp_tensor_release);
        if (readStatus == IMP_ERROR_STREAM_END) {
            break;
        }
        if (readStatus != IMP_OK || frame == nullptr) {
            return absl::InvalidArgumentError("Failed to decode video frame");
        }

        const uint8_t* yPlane = nullptr;
        const uint8_t* uvPlane = nullptr;
        int frameWidth = 0;
        int frameHeight = 0;
        if (imp_tensor_get_nv12_planes(frame.get(), &yPlane, &uvPlane, &frameWidth, &frameHeight) != IMP_OK ||
            frameWidth <= 0 || frameHeight <= 0) {
            return absl::InternalError("Video decoder did not return system-memory NV12 frame");
        }
        const size_t decodedWidth = static_cast<size_t>(frameWidth);
        const size_t decodedHeight = static_cast<size_t>(frameHeight);
        if (decodedWidth > std::numeric_limits<size_t>::max() / decodedHeight) {
            return absl::InvalidArgumentError("Video frame dimensions are too large");
        }
        const size_t framePixels = decodedWidth * decodedHeight;
        const size_t remainingPixels = totalAllocatedPixels >= maxAllowedPixels ? 0 : maxAllowedPixels - totalAllocatedPixels;
        if (framePixels > remainingPixels || framePixels > std::numeric_limits<size_t>::max() / 3) {
            return absl::InvalidArgumentError("Video exceeds maximum decoded size");
        }
        if (rgbFrames.empty()) {
            width = decodedWidth;
            height = decodedHeight;
        } else if (decodedWidth != width || decodedHeight != height) {
            return absl::InvalidArgumentError("All video frames must have the same height and width");
        }

        std::vector<uint8_t> rgb(framePixels * 3);
        convertNv12ToRgb(yPlane, uvPlane, width, height, rgb);
        totalAllocatedPixels += framePixels;
        rgbFrames.push_back(std::move(rgb));
    }

    if (rgbFrames.empty()) {
        return absl::InvalidArgumentError("Video contains no frames");
    }

    const size_t frameBytes = width * height * 3;
    ov::Tensor video(ov::element::u8, ov::Shape{rgbFrames.size(), height, width, 3});
    auto* destination = static_cast<uint8_t*>(video.data());
    for (size_t index = 0; index < rgbFrames.size(); ++index) {
        std::memcpy(destination + index * frameBytes, rgbFrames[index].data(), frameBytes);
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Decoded video source into {} RGB frames of shape [{}, {}, 3]", rgbFrames.size(), height, width);
    return video;
}

}  // namespace ovms
