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

#include "image_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <string.h>

#include "src/logging.hpp"
#include "src/filesystem/filesystem.hpp"
#include "src/image_utils/decoded_image_size.hpp"
#include "src/image_utils/image_conversion.hpp"
#include "src/predict_request_validation_utils_impl.hpp"

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#pragma warning(pop)

#include <curl/curl.h>

namespace ovms {

size_t ImageDownloadContext::append(const void* downloadedChunk, size_t size, size_t nmemb) noexcept {
    if (size != 0 && nmemb > std::numeric_limits<size_t>::max() / size) {
        sizeLimitExceeded = true;
        return 0;
    }
    const size_t realsize = size * nmemb;
    if (bytesReceived > sizeLimit || realsize > sizeLimit - bytesReceived) {
        sizeLimitExceeded = true;
        return 0;
    }
    try {
        image->append(static_cast<const char*>(downloadedChunk), realsize);
        bytesReceived += realsize;
    } catch (...) {
        allocationFailed = true;
        return 0;
    }
    return realsize;
}

namespace {

bool isPathInsideDirectory(const std::filesystem::path& testedPath, const std::filesystem::path& allowedDirectory) {
    const auto mismatch = std::mismatch(
        allowedDirectory.begin(), allowedDirectory.end(),
        testedPath.begin(), testedPath.end());
    return mismatch.first == allowedDirectory.end();
}

static size_t appendChunkCallback(void* downloadedChunk, size_t size, size_t nmemb,
    void* context) noexcept {
    return static_cast<ImageDownloadContext*>(context)->append(downloadedChunk, size, nmemb);
}

struct MediaFileDownloadContext {
    std::FILE* output;
    size_t sizeLimit;
    size_t bytesReceived = 0;
    bool sizeLimitExceeded = false;
};

constexpr int64_t MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 120;

size_t appendMediaFileChunk(void* downloadedChunk, size_t size, size_t nmemb, void* context) noexcept {
    auto* downloadContext = static_cast<MediaFileDownloadContext*>(context);
    if (size != 0 && nmemb > std::numeric_limits<size_t>::max() / size) {
        downloadContext->sizeLimitExceeded = true;
        return 0;
    }
    const size_t realSize = size * nmemb;
    if (realSize > downloadContext->sizeLimit - downloadContext->bytesReceived) {
        downloadContext->sizeLimitExceeded = true;
        return 0;
    }
    const size_t written = std::fwrite(downloadedChunk, 1, realSize, downloadContext->output);
    if (written != realSize) {
        return 0;
    }
    downloadContext->bytesReceived += realSize;
    return realSize;
}

const char* getDownloadFailureReason(CURLcode status) {
    switch (status) {
    case CURLE_URL_MALFORMAT:
        return "invalid_url";
    case CURLE_UNSUPPORTED_PROTOCOL:
        return "unsupported_protocol";
    case CURLE_COULDNT_RESOLVE_HOST:
        return "dns_resolution_failed";
    case CURLE_COULDNT_RESOLVE_PROXY:
        return "proxy_resolution_failed";
    case CURLE_COULDNT_CONNECT:
        return "connect_failed";
    case CURLE_OPERATION_TIMEDOUT:
        return "timeout";
    case CURLE_SSL_CONNECT_ERROR:
        return "tls_connection_failed";
    case CURLE_PEER_FAILED_VERIFICATION:
        return "tls_verification_failed";
    case CURLE_LOGIN_DENIED:
        return "authentication_failed";
    case CURLE_REMOTE_ACCESS_DENIED:
        return "remote_access_denied";
    case CURLE_TOO_MANY_REDIRECTS:
        return "redirect_limit_exceeded";
    case CURLE_SEND_ERROR:
    case CURLE_RECV_ERROR:
        return "io_failed";
    default:
        return "transfer_failed";
    }
}

#define CURL_SETOPT(setopt)   \
    if (status == CURLE_OK) { \
        status = setopt;      \
    }

}  // namespace

absl::Status validateHttpDownloadResponse(int64_t httpCode, bool redirectsAllowed, std::string_view mediaType) {
    if (httpCode >= 300 && httpCode < 400 && !redirectsAllowed) {
        return absl::InvalidArgumentError("Failed to download " + std::string(mediaType) + " from external URL because redirects are not allowed. Set OVMS_MEDIA_URL_ALLOW_REDIRECTS=1 environment variable to allow redirects.");
    }
    if (httpCode < 200 || httpCode >= 300) {
        return absl::InvalidArgumentError("Failed to download " + std::string(mediaType) + " from external URL: HTTP status " + std::to_string(httpCode) + ".");
    }
    return absl::OkStatus();
}

namespace {

absl::Status downloadImage(const char* url, std::string& image, const int64_t& sizeLimit) {
    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
            "External image download failed: reason=curl_initialization");
        return absl::InternalError("Image downloading failed");
    }
    auto handleGuard = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(curl_handle, curl_easy_cleanup);
    if (sizeLimit < 0) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
            "External image download failed: reason=invalid_size_limit size_limit_bytes={}", sizeLimit);
        return absl::InvalidArgumentError("Image size limit cannot be negative");
    }
    ImageDownloadContext downloadContext{&image, static_cast<size_t>(sizeLimit)};

    auto status = curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, appendChunkCallback))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &downloadContext))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA))
    const char* envAllowRedirects = std::getenv("OVMS_MEDIA_URL_ALLOW_REDIRECTS");
    const bool redirectsAllowed = envAllowRedirects != nullptr && std::strcmp(envAllowRedirects, "1") == 0;
    if (redirectsAllowed) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "URL redirects allowed");
        CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L))
    }
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(sizeLimit)))

    if (status != CURLE_OK) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
            "External image download failed: reason=curl_configuration curl_code={} curl_error={}",
            static_cast<int>(status), curl_easy_strerror(status));
        return absl::InternalError("Failed to configure image download.");
    }

    status = curl_easy_perform(curl_handle);
    if (status != CURLE_OK) {
        if (downloadContext.sizeLimitExceeded || status == CURLE_FILESIZE_EXCEEDED) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
                "External image download failed: reason=size_limit_exceeded size_limit_bytes={} bytes_received={}",
                sizeLimit, downloadContext.bytesReceived);
            return absl::InvalidArgumentError("Downloaded image exceeds the size limit");
        }
        if (downloadContext.allocationFailed) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
                "External image download failed: reason=memory_allocation_failed bytes_received={}",
                downloadContext.bytesReceived);
            return absl::ResourceExhaustedError("Memory allocation failed while downloading image");
        }
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
            "External image download failed: reason={} curl_code={} curl_error={}",
            getDownloadFailureReason(status), static_cast<int>(status), curl_easy_strerror(status));
        return absl::InvalidArgumentError("Failed to download image from external URL.");
    }

    const bool isHttpUrl = std::strncmp(url, "http://", 7) == 0 || std::strncmp(url, "https://", 8) == 0;
    if (isHttpUrl) {
        long httpCode = 0;  // NOLINT(runtime/int): CURLINFO_RESPONSE_CODE requires long*.
        status = curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &httpCode);
        if (status != CURLE_OK) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
                "External image download failed: reason=response_status_unavailable curl_code={} curl_error={}",
                static_cast<int>(status), curl_easy_strerror(status));
            return absl::InternalError("Failed to get image download response status.");
        }
        if (httpCode >= 300 && httpCode < 400 && !redirectsAllowed) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
                "External image download failed: reason=redirect_disallowed http_status={} redirects_allowed=false",
                httpCode);
        } else if (httpCode < 200 || httpCode >= 300) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
                "External image download failed: reason=http_status http_status={}", httpCode);
        }
        return validateHttpDownloadResponse(static_cast<int64_t>(httpCode), redirectsAllowed, "image");
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Downloading image succeeded, {} bytes retrieved", image.size());
    return absl::OkStatus();
}

absl::Status readLocalImage(const std::filesystem::path& path, std::string& image, const int64_t& sizeLimit) {
    std::error_code ec;
    auto fileSize = std::filesystem::file_size(path, ec);
    if (ec) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to get file size for {}: {}", path.string(), ec.message());
        return absl::InvalidArgumentError("Image loading failed");
    }
    if (fileSize > static_cast<uint64_t>(sizeLimit)) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Local image file too large to inspect. Path: {}, Size: {}", path.string(), fileSize);
        return absl::InvalidArgumentError("Image too large");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to open local image file: {}", path.string());
        return absl::InvalidArgumentError("Image loading failed");
    }

    image.resize(fileSize);
    file.read(image.data(), static_cast<std::streamsize>(fileSize));
    if (file.fail()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to read local image file: {}", path.string());
        return absl::InvalidArgumentError("Image loading failed");
    }
    return absl::OkStatus();
}

bool isDomainAllowed(const std::vector<std::string>& allowedDomains, const char* url) {
    if (allowedDomains.size() == 1 && allowedDomains[0] == "all") {
        return true;
    }
    CURLUcode rc;
    CURLU* parsedUrl = curl_url();
    if (!parsedUrl) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsing url {} failed", url);
        return false;
    }
    rc = curl_url_set(parsedUrl, CURLUPART_URL, url, 0);
    if (rc) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsing url {} failed", url);
        curl_url_cleanup(parsedUrl);
        return false;
    }
    char* host;
    rc = curl_url_get(parsedUrl, CURLUPART_HOST, &host, 0);
    if (rc) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsing url {} hostname failed", url);
        curl_url_cleanup(parsedUrl);
        return false;
    }
    bool allowed = false;
    for (const auto& allowedDomain : allowedDomains) {
        if (allowedDomain.compare(host) == 0) {
            allowed = true;
            break;
        }
    }
    curl_free(host);
    curl_url_cleanup(parsedUrl);
    return allowed;
}

}  // namespace

absl::Status downloadMediaToFile(const std::string& mediaSource, std::FILE* output,
    size_t sizeLimit, size_t& downloadedBytes) {
    if (output == nullptr) {
        return absl::InternalError("Unable to create temporary video storage");
    }
    CURL* curlHandle = curl_easy_init();
    if (!curlHandle) {
        return absl::InternalError("Video downloading failed");
    }
    auto handleGuard = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(curlHandle, curl_easy_cleanup);
    MediaFileDownloadContext downloadContext{output, sizeLimit};
    CURLcode status = curl_easy_setopt(curlHandle, CURLOPT_URL, mediaSource.c_str());
    CURL_SETOPT(curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, appendMediaFileChunk))
    CURL_SETOPT(curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &downloadContext))
    CURL_SETOPT(curl_easy_setopt(curlHandle, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA))
    const char* envAllowRedirects = std::getenv("OVMS_MEDIA_URL_ALLOW_REDIRECTS");
    if (envAllowRedirects != nullptr && std::strcmp(envAllowRedirects, "1") == 0) {
        CURL_SETOPT(curl_easy_setopt(curlHandle, CURLOPT_FOLLOWLOCATION, 1L))
    }
    CURL_SETOPT(curl_easy_setopt(curlHandle, CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(sizeLimit)))
    CURL_SETOPT(curl_easy_setopt(curlHandle, CURLOPT_TIMEOUT, MEDIA_DOWNLOAD_TIMEOUT_SECONDS))
    if (status != CURLE_OK) {
        return absl::InternalError("Video downloading failed");
    }

    status = curl_easy_perform(curlHandle);
    if (status != CURLE_OK) {
        if (downloadContext.sizeLimitExceeded || status == CURLE_FILESIZE_EXCEEDED) {
            return absl::InvalidArgumentError("Encoded video source exceeds the allowed file size");
        }
        return absl::InvalidArgumentError("Video downloading failed");
    }
    const bool isHttpUrl = mediaSource.rfind("http://", 0) == 0 || mediaSource.rfind("https://", 0) == 0;
    if (isHttpUrl) {
        long httpCode = 0;  // NOLINT(runtime/int): CURLINFO_RESPONSE_CODE requires long*.
        status = curl_easy_getinfo(curlHandle, CURLINFO_RESPONSE_CODE, &httpCode);
        if (status != CURLE_OK) {
            return absl::InternalError("Failed to get video download response status.");
        }
        const bool redirectsAllowed = envAllowRedirects != nullptr && std::strcmp(envAllowRedirects, "1") == 0;
        const auto responseStatus = validateHttpDownloadResponse(static_cast<int64_t>(httpCode), redirectsAllowed, "video");
        if (!responseStatus.ok()) {
            return responseStatus;
        }
    }
    if (std::fflush(output) != 0) {
        return absl::InternalError("Unable to write downloaded video source");
    }
    downloadedBytes = downloadContext.bytesReceived;
    return absl::OkStatus();
}

absl::Status validateMediaSource(const std::string& mediaSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains) {
    const bool isRemoteUrl = mediaSource.rfind("http://", 0) == 0 || mediaSource.rfind("https://", 0) == 0 ||
                             mediaSource.rfind("ftp://", 0) == 0 || mediaSource.rfind("sftp://", 0) == 0;
    if (isRemoteUrl) {
        if (!allowedMediaDomains.has_value() || !isDomainAllowed(allowedMediaDomains.value(), mediaSource.c_str())) {
            return absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains");
        }
        return absl::OkStatus();
    }

    if (!allowedLocalMediaPath.has_value()) {
        return absl::InvalidArgumentError("Loading media from local filesystem is disabled.");
    }
    if (FileSystem::isPathEscaped(mediaSource)) {
        std::stringstream ss;
        ss << "Path " << mediaSource.c_str() << " escape with .. is forbidden.";
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
        return absl::InvalidArgumentError(ss.str());
    }
    const std::filesystem::path resolvedAllowedPath = FileSystem::normalizeConfiguredPath(allowedLocalMediaPath.value());
    const std::string resolvedMediaPathStr = FileSystem::normalizeConfiguredPath(mediaSource);
    if (!isPathInsideDirectory(resolvedMediaPathStr, resolvedAllowedPath)) {
        return absl::InvalidArgumentError("Given filepath is not subpath of allowed_local_media_path");
    }
    return absl::OkStatus();
}

absl::StatusOr<ov::Tensor> fetchAndDecodeImage(const std::string& imageSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains,
    size_t& totalAllocatedPixels, size_t maxAllowedImagePixels) {
    std::size_t pos = imageSource.find(BASE64_PREFIX);
    std::string decoded;
    // Part 1: fetch the encoded image bytes into `decoded` (base64, URL, or local file).
    if (pos != std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from base64 string");
        size_t offset = pos + BASE64_PREFIX.length();
        if (!absl::Base64Unescape(std::string_view(imageSource.data() + offset, imageSource.size() - offset), &decoded)) {
            return absl::InvalidArgumentError("Invalid base64 string in request");
        }
    } else if (imageSource.rfind("http://", 0) == 0 || imageSource.rfind("https://", 0) == 0 ||
               imageSource.rfind("ftp://", 0) == 0 || imageSource.rfind("sftp://", 0) == 0) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image using curl");
        auto validationStatus = validateMediaSource(imageSource, allowedLocalMediaPath, allowedMediaDomains);
        if (!validationStatus.ok()) {
            return validationStatus;
        }
        auto status = downloadImage(imageSource.c_str(), decoded, MAX_IMAGE_SIZE_BYTES);
        if (status != absl::OkStatus()) {
            return status;
        }
    } else {
        auto validationStatus = validateMediaSource(imageSource, allowedLocalMediaPath, allowedMediaDomains);
        if (!validationStatus.ok()) {
            return validationStatus;
        }
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from local filesystem");
        const std::string resolvedImagePathStr = FileSystem::normalizeConfiguredPath(imageSource);
        const std::filesystem::path resolvedImagePath = resolvedImagePathStr;
        auto status = readLocalImage(resolvedImagePath, decoded, MAX_IMAGE_SIZE_BYTES);
        if (status != absl::OkStatus()) {
            return status;
        }
    }

    // Part 2: guard against decompression bombs, then decode the in-memory bytes exactly once.
    uint64_t estimatedDecodedPixels = 0;
    auto estimate = image_utils::estimateDecodedImageSize(decoded, estimatedDecodedPixels);
    if (estimate == image_utils::DecodedSizeEstimate::InputTooLarge) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image binary payload too large to inspect. Size: {}", decoded.size());
        return absl::InvalidArgumentError("Image too large");
    }
    if (estimate == image_utils::DecodedSizeEstimate::UnsupportedFormat &&
        !request_validation_utils::allowUnestimatableImageFormats()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image decoded size could not be estimated and unestimatable formats are not allowed");
        return absl::InvalidArgumentError("Image format decoded size cannot be verified");
    }
    size_t remainingBudget = totalAllocatedPixels >= maxAllowedImagePixels ? 0 : maxAllowedImagePixels - totalAllocatedPixels;
    if (estimate == image_utils::DecodedSizeEstimate::Estimated &&
        estimatedDecodedPixels > remainingBudget) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Estimated decoded image pixels {} exceeds remaining budget {}",
            estimatedDecodedPixels, remainingBudget);
        return absl::InvalidArgumentError("Image exceeds maximum decoded size");
    }
    ov::Tensor imageTensor;
    try {
        imageTensor = loadImageStbiFromMemory(decoded);
    } catch (const std::runtime_error& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image parsing failed: {}", e.what());
        return absl::InvalidArgumentError("Image parsing failed: " + std::string(e.what()));
    }

    // Part 3: bound the actual decoded pixel count against the remaining per-request budget and
    // advance the running total so subsequent images in the same request see the reduced budget.
    const auto& shape = imageTensor.get_shape();
    if (shape.size() != 4) {
        return absl::InternalError("Decoded image tensor has unexpected shape");
    }
    if (shape[1] != 0 && shape[2] > remainingBudget / shape[1]) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Decoded image dimensions exceed remaining budget {}", remainingBudget);
        return absl::InvalidArgumentError("Image exceeds maximum decoded size");
    }
    size_t imagePixels = shape[1] * shape[2];
    if (imagePixels > remainingBudget) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Decoded image pixels {} exceeds remaining budget {}",
            imagePixels, remainingBudget);
        return absl::InvalidArgumentError("Image exceeds maximum decoded size");
    }
    totalAllocatedPixels += imagePixels;
    return imageTensor;
}

}  // namespace ovms
