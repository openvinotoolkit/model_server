//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "gguf_downloader.hpp"

#include <string>
#include <memory>
#include <iostream>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "../capi_frontend/server_settings.hpp"
#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../stringutils.hpp"
#include "../status.hpp"
#include "../version.hpp"

namespace ovms {
std::string GGUFDownloader::getGraphDirectory() {
    return this->downloadPath;
}
static Status checkIfOverwriteAndRemove(const HFSettingsImpl& hfSettings, const std::string& path) {
    auto lfstatus = StatusCode::OK;
    if (hfSettings.overwriteModels && std::filesystem::is_directory(path)) {
        LocalFileSystem lfs;
        lfstatus = lfs.deleteFileFolder(path);
        if (lfstatus != StatusCode::OK) {
            SPDLOG_ERROR("Error occurred while deleting path: {} reason: {}",
                path,
                lfstatus);
        } else {
            SPDLOG_DEBUG("Path deleted: {}", path);
        }
    }
    return lfstatus;
}

GGUFDownloader::GGUFDownloader(const std::string& hfEndpoint, const HFSettingsImpl& hfSettings) :
    hfSettings(hfSettings),
    hfEndpoint(hfEndpoint) {
    // TODO shared logic across all pullers
    this->downloadPath = FileSystem::joinPath({this->hfSettings.downloadPath, this->hfSettings.sourceModel});
}

Status GGUFDownloader::downloadModel() {
    if (this->hfSettings.downloadType != GGUF_DOWNLOAD) {
        SPDLOG_ERROR("Wrong download type selected. Expected GGUF download type.");
        return StatusCode::INTERNAL_ERROR;
    }
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    if (std::filesystem::is_directory(this->downloadPath) && !this->hfSettings.overwriteModels) {
        SPDLOG_DEBUG("Path already exists on local filesystem. Not downloading to path: {}", this->downloadPath);
        return StatusCode::OK;
    }
    std::filesystem::create_directories(this->downloadPath);
    ovms::Status status;
    if (!status.ok()) {
        return status;
    }
    status = checkIfOverwriteAndRemove(this->hfSettings, this->downloadPath);
    if (!status.ok()) {
        return status;
    }
    status = downloadWithCurl(this->hfEndpoint, this->hfSettings.sourceModel, "/resolve/main/", this->hfSettings.ggufFilename.value(), this->downloadPath);
    if (!status.ok()) {
        SPDLOG_ERROR("Error occurred while downloading GGUF model: {} reason: {}", this->hfSettings.sourceModel, status.string());
        return status;
    }
    return StatusCode::OK;
}

static const char* rate_units[] = {"B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s", NULL};

static void print_download_speed_info(size_t received_size, size_t elapsed_time) {
    double recv_len = (double)received_size;
    uint64_t elapsed = (uint64_t)elapsed_time;
    double rate;
    rate = elapsed ? recv_len / elapsed : received_size;

    size_t rate_unit_idx = 0;
    while (rate > 1024 && rate_units[rate_unit_idx + 1]) {
        rate /= 1024.0;
        rate_unit_idx++;
    }
    printf(" [%.2f %s] ", rate, rate_units[rate_unit_idx]);
}

static const char* sizeUnits[] = {"B", "KB", "MB", "GB", "TB", NULL};
void print_progress(size_t count, size_t max, bool first_run, size_t elapsed_time) {
    float progress = (float)count / max;
    if (!first_run && progress < 0.01 && count > 0)
        return;

    const int bar_width = 50;
    int bar_length = progress * bar_width;

    printf("\rProgress: [");
    int i;
    for (i = 0; i < bar_length; ++i) {
        printf("#");
    }
    for (i = bar_length; i < bar_width; ++i) {
        printf(" ");
    }
    size_t totalSizeUnitId = 0;
    double totalSize = max;
    while (totalSize > 1000 && sizeUnits[totalSizeUnitId + 1]) {
        totalSize /= 1000.0;
        totalSizeUnitId++;
    }
    printf("] %.2f%% of %.2f %s", progress * 100, totalSize, sizeUnits[totalSizeUnitId]);
    print_download_speed_info(count, elapsed_time);
    if (progress == 1.0)
        printf("\n");
    fflush(stdout);
}

struct FtpFile {
    const char* filename;
    FILE* stream;
};

void fileClose(FILE* file) {
    if (file) {
        fclose(file);
    }
}

static size_t file_write_callback(void* buffer, size_t size, size_t nmemb, void* stream) {
    struct FtpFile* out = (struct FtpFile*)stream;
    if (!out->stream) {
        out->stream = fopen(out->filename, "wb");
        if (!out->stream) {
            fprintf(stderr, "failure, cannot open file to write: %s\n",
                out->filename);
            return 0;
        }
    }
    return fwrite(buffer, size, nmemb, out->stream);
}

#define CHECK_CURL_CALL(call)                                                                            \
    do {                                                                                                 \
        CURLcode curlCode = call;                                                                        \
        if (curlCode != CURLE_OK) {                                                                      \
            SPDLOG_ERROR("curl error: {}. Error code: {}", curl_easy_strerror(curlCode), (int)curlCode); \
            return StatusCode::INTERNAL_ERROR;                                                           \
        }                                                                                                \
    } while (0)

struct ProgressData {
    time_t started_download;
    time_t last_print_time;
    bool fullDownloadPrinted = false;
};
int progress_callback(void* clientp,
    curl_off_t dltotal,
    curl_off_t dlnow,
    curl_off_t ultotal,
    curl_off_t ulnow) {
    ProgressData* pcs = reinterpret_cast<ProgressData*>(clientp);
    if (dlnow == 0) {
        pcs->started_download = time(NULL);
        pcs->last_print_time = time(NULL);
    }
    time_t currentTime = time(NULL);
    if (currentTime - pcs->last_print_time < 1) {
        return 0;
    }
    // SPDLOG_DEBUG("Progress callback called with dltotal: {}, dlnow: {}, ultotal: {}, ulnow: {}", dltotal, dlnow, ultotal, ulnow);
    if ((dltotal == dlnow) && dltotal < 1000) {
        // Usually with first messages we don't get the full size and we don't want to print progress bar
        // so we assume that until dltotal is less than 1000 we don't have full size
        // otherwise we would print 100% progress bar
        pcs->last_print_time = currentTime;
        return 0;
    }
    // called multiple times, so we want to print progress bar only once reached 100%
    if (pcs->fullDownloadPrinted) {
        return 0;
    }
    print_progress(dlnow, dltotal, (dlnow == 0), currentTime - pcs->started_download);
    std::cout.flush();
    pcs->fullDownloadPrinted = (dltotal == dlnow);
    return 0;
}

Status GGUFDownloader::downloadWithCurl(const std::string& hfEndpoint, const std::string& modelName, const std::string& filenamePrefix, const std::string& ggufFilename, const std::string& downloadPath) {
    // construct url
    SPDLOG_TRACE("hfEndpoint: {} modelName: {} filenamePrefix: {} ggufFilename: {}, downloadPath:{}", hfEndpoint, modelName, filenamePrefix, ggufFilename, downloadPath);
    std::string url = hfEndpoint + modelName + filenamePrefix + ggufFilename;
    SPDLOG_TRACE("Constructed URL: {}", url);

    // construct filepath
    auto filePath = FileSystem::joinPath({downloadPath, ggufFilename});

    // agent string required to avoid 403 Forbidden error on modelscope
    std::string agentString = std::string(PROJECT_NAME) + "/" + std::string(PROJECT_VERSION);

    CURL* curl = nullptr;
    CHECK_CURL_CALL(curl_global_init(CURL_GLOBAL_DEFAULT));
    auto globalCurlGuard = std::unique_ptr<void, void (*)(void*)>(
        nullptr, [](void*) { curl_global_cleanup(); });
    curl = curl_easy_init();
    if (!curl) {
        SPDLOG_ERROR("Failed to initialize cURL.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto handleGuard = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(curl, curl_easy_cleanup);
    // set impl options
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_URL, url.c_str()));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, file_write_callback));
    struct FtpFile ftpFile = {filePath.c_str(), NULL};
    auto fileCloseGuard = std::unique_ptr<FILE, decltype(&fileClose)>(ftpFile.stream, fileClose);
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ftpFile));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_USERAGENT, agentString.c_str()));
    // progress bar options
    ProgressData progressData;
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressData));
    // other options
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL));
    CHECK_CURL_CALL(curl_easy_perform(curl));
    SPDLOG_TRACE("cURL download completed for model: {} to path: {}", modelName, filePath);
    return StatusCode::OK;
}
}  // namespace ovms
