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
#include "curl_downloader.hpp"

#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <curl/curl.h>
#include <stdio.h>

#include "src/logging.hpp"
#include "src/status.hpp"
#include "src/version.hpp"

namespace ovms {

static const char* sizeUnits[] = {"B", "KB", "MB", "GB", "TB", NULL};

static void print_download_speed_info(size_t received_size, size_t elapsed_time) {
    double recv_len = (double)received_size;
    uint64_t elapsed = (uint64_t)elapsed_time;
    double rate;
    rate = elapsed ? recv_len / elapsed : received_size;

    size_t rate_unit_idx = 0;
    while (rate > 1000 && sizeUnits[rate_unit_idx + 1]) {
        rate /= 1000.0;
        rate_unit_idx++;
    }
    printf(" [%.2f %s/s] ", rate, sizeUnits[rate_unit_idx]);
}

static void print_progress(size_t count, size_t max, bool first_run, size_t elapsed_time) {
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

struct CurlDownloadFile {
    const char* filename;
    FILE* stream;
    CurlDownloadFile() = delete;
    CurlDownloadFile(const CurlDownloadFile&) = delete;
    CurlDownloadFile& operator=(const CurlDownloadFile&) = delete;
    CurlDownloadFile(const char* fname, FILE* str) :
        filename(fname),
        stream(str) {}
    ~CurlDownloadFile() {
        if (stream) {
            fclose(stream);
        }
        if (!success) {
            std::filesystem::remove(filename);
        }
    }
    bool success = false;
};

static size_t file_write_callback(void* buffer, size_t size, size_t nmemb, void* stream) {
    CurlDownloadFile* out = static_cast<CurlDownloadFile*>(stream);
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

static int progress_callback(void* clientp,
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
    bool shouldPrintDueToTime = (currentTime - pcs->last_print_time >= 1);
    if ((dltotal == dlnow) && dltotal < 10000) {
        return 0;
    }
    if (pcs->fullDownloadPrinted) {
        return 0;
    }
    if (!shouldPrintDueToTime && (dltotal != dlnow)) {
        return 0;
    }
    pcs->fullDownloadPrinted = (dltotal == dlnow);
    pcs->last_print_time = currentTime;
    print_progress(dlnow, dltotal, (dlnow == 0), currentTime - pcs->started_download);
    std::cout.flush();
    return 0;
}

Status downloadFileWithCurl(const std::string& url, const std::string& filePath) {
    return downloadFileWithCurl(url, filePath, "");
}

Status downloadFileWithCurl(const std::string& url, const std::string& filePath, const std::string& authTokenHF) {
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
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_URL, url.c_str()));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, file_write_callback));
    CurlDownloadFile downloadFile{filePath.c_str(), NULL};
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_WRITEDATA, &downloadFile));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_USERAGENT, agentString.c_str()));
    struct curl_slist* headers = nullptr;
    std::string authHeader;
    if (!authTokenHF.empty()) {
        authHeader = "Authorization: Bearer " + authTokenHF;
        headers = curl_slist_append(headers, authHeader.c_str());
        CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers));
    }
    auto headersGuard = std::unique_ptr<struct curl_slist, decltype(&curl_slist_free_all)>(headers, curl_slist_free_all);
    ProgressData progressData;
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressData));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L));
    CHECK_CURL_CALL(curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL));
    CHECK_CURL_CALL(curl_easy_perform(curl));
    int32_t http_code = 0;
    CHECK_CURL_CALL(curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code));
    SPDLOG_TRACE("HTTP response code: {}", http_code);
    if (http_code != 200) {
        SPDLOG_ERROR("Failed to download file from URL: {} HTTP response code: {}", url, http_code);
        return StatusCode::PATH_INVALID;
    }
    downloadFile.success = true;
    return StatusCode::OK;
}

#undef CHECK_CURL_CALL

}  // namespace ovms
