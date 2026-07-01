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
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <regex>
#include <sstream>
#include <string.h>

#include "../../logging.hpp"
#include "../../filesystem/filesystem.hpp"
#include "../../image_conversion.hpp"

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#pragma warning(pop)

#include <curl/curl.h>

namespace ovms {

namespace {

bool isPathInsideDirectory(const std::filesystem::path& testedPath, const std::filesystem::path& allowedDirectory) {
    const auto mismatch = std::mismatch(
        allowedDirectory.begin(), allowedDirectory.end(),
        testedPath.begin(), testedPath.end());
    return mismatch.first == allowedDirectory.end();
}

static size_t appendChunkCallback(void* downloadedChunk, size_t size, size_t nmemb,
    void* image) {
    size_t realsize = size * nmemb;
    auto& mem = *static_cast<std::string*>(image);
    mem.append(static_cast<char*>(downloadedChunk), realsize);
    return realsize;
}

#define CURL_SETOPT(setopt)   \
    if (status == CURLE_OK) { \
        status = setopt;      \
    }

absl::Status downloadImage(const char* url, std::string& image, const int64_t& sizeLimit) {
    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to initialize curl handle");
        return absl::InternalError("Image downloading failed");
    }
    auto handleGuard = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(curl_handle, curl_easy_cleanup);

    auto status = curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, appendChunkCallback))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &image))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA))
    const char* envAllowRedirects = std::getenv("OVMS_MEDIA_URL_ALLOW_REDIRECTS");
    if (envAllowRedirects != nullptr && (std::strcmp(envAllowRedirects, "1") == 0)) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "URL redirects allowed");
        CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L))
    }
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_MAXFILESIZE, sizeLimit))

    if (status != CURLE_OK) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Setting curl opts failed: {}", curl_easy_strerror(status));
        return absl::InvalidArgumentError("Image downloading failed");
    }

    status = curl_easy_perform(curl_handle);
    if (status != CURLE_OK) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Downloading image failed: {}", curl_easy_strerror(status));
        return absl::InvalidArgumentError("Image downloading failed");
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Downloading image succeeded, {} bytes retrieved", image.size());
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

absl::StatusOr<ov::Tensor> loadImage(const std::string& imageSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains) {
    std::size_t pos = imageSource.find(BASE64_PREFIX);
    std::string decoded;
    ov::Tensor tensor;
    if (pos != std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from base64 string");
        size_t offset = pos + BASE64_PREFIX.length();
        if (!absl::Base64Unescape(std::string_view(imageSource.data() + offset, imageSource.size() - offset), &decoded)) {
            return absl::InvalidArgumentError("Invalid base64 string in request");
        }
        try {
            tensor = loadImageStbiFromMemory(decoded);
        } catch (std::runtime_error& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image parsing failed: {}", e.what());
            return absl::InvalidArgumentError("Image parsing failed");
        }
    } else if (imageSource.rfind("http://", 0) == 0 || imageSource.rfind("https://", 0) == 0 ||
               imageSource.rfind("ftp://", 0) == 0 || imageSource.rfind("sftp://", 0) == 0) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image using curl");
        if (!allowedMediaDomains.has_value() || !isDomainAllowed(allowedMediaDomains.value(), imageSource.c_str())) {
            return absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains");
        }
        auto status = downloadImage(imageSource.c_str(), decoded, MAX_IMAGE_SIZE_BYTES);
        if (status != absl::OkStatus()) {
            return status;
        }
        try {
            tensor = loadImageStbiFromMemory(decoded);
        } catch (std::runtime_error& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image parsing failed: {}", e.what());
            return absl::InvalidArgumentError("Image parsing failed");
        }
    } else {
        if (!allowedLocalMediaPath.has_value()) {
            return absl::InvalidArgumentError("Loading images from local filesystem is disabled.");
        }
        if (FileSystem::isPathEscaped(imageSource)) {
            std::stringstream ss;
            ss << "Path " << imageSource.c_str() << " escape with .. is forbidden.";
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
            return absl::InvalidArgumentError(ss.str());
        }
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from local filesystem");
        const std::filesystem::path resolvedAllowedPath = FileSystem::normalizeConfiguredPath(allowedLocalMediaPath.value());
        const std::string resolvedImagePathStr = FileSystem::normalizeConfiguredPath(imageSource);
        const std::filesystem::path resolvedImagePath = resolvedImagePathStr;
        if (!isPathInsideDirectory(resolvedImagePath, resolvedAllowedPath)) {
            return absl::InvalidArgumentError("Given filepath is not subpath of allowed_local_media_path");
        }
        try {
            tensor = loadImageStbiFromFile(resolvedImagePathStr.c_str());
        } catch (std::runtime_error& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image file {} parsing failed: {}", resolvedImagePathStr, e.what());
            return absl::InvalidArgumentError("Image file parsing failed");
        }
    }
    return tensor;
}

}  // namespace ovms
