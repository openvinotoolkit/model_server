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

#include <stdio.h>

namespace ovms {
std::string GGUFDownloader::getGraphDirectory() {
    return this->downloadPath;
}
static Status checkIfOverwriteAndRemove(const HFSettingsImpl& hfSettings, const std::string& path) {
    // improve so that only specific files are deleted not all quantizations FIXME
    auto lfstatus = StatusCode::OK;
    if (hfSettings.overwriteModels && std::filesystem::is_directory(path)) {
        auto allSpecifiedQuantizationPartsFilenamesOrStatus = GGUFDownloader::createGGUFFilenamesToDownload(hfSettings.ggufFilename.value());
        if (std::holds_alternative<Status>(allSpecifiedQuantizationPartsFilenamesOrStatus)) {
            return std::get<Status>(allSpecifiedQuantizationPartsFilenamesOrStatus);
        }
        auto& allSpecifiedQuantizationPartsFilenames = std::get<std::vector<std::string>>(allSpecifiedQuantizationPartsFilenamesOrStatus);
        LocalFileSystem lfs;
        for (const auto& file : allSpecifiedQuantizationPartsFilenames) {
            auto filePath = FileSystem::joinPath({path, file});
            SPDLOG_TRACE("Checking if model file exists for overwrite: {}", filePath);
            if (std::filesystem::exists(filePath)) {
                SPDLOG_TRACE("Model file already exists and will be removed due to overwrite flag: {}", filePath);
                lfstatus = lfs.deleteFileFolder(filePath);
                if (lfstatus != StatusCode::OK) {
                    SPDLOG_ERROR("Error occurred while deleting path: {} reason: {}", path,lfstatus);
                } else {
                    SPDLOG_TRACE("Path deleted: {}", path);
                }
            }
        }
    }
    return lfstatus;
}

std::variant<Status, bool> GGUFDownloader::checkIfAlreadyExists(const HFSettingsImpl& hfSettings, const std::string& path) {
    auto ggufFilesOrStatus = createGGUFFilenamesToDownload(hfSettings.ggufFilename.value());
    if (std::holds_alternative<Status>(ggufFilesOrStatus)) {
        SPDLOG_ERROR("Could not create GGUF filenames to download for checking existing files");
        return std::get<Status>(ggufFilesOrStatus);
    }
    auto& ggufFiles = std::get<std::vector<std::string>>(ggufFilesOrStatus);
    for (const auto& file : ggufFiles) {
        auto filePath = FileSystem::joinPath({path, file});
        SPDLOG_DEBUG("Checking if model file exists: {}", filePath);
        bool exist = false;
        auto status = LocalFileSystem::exists(filePath, &exist);
        if (!status.ok()) return status;
        if (exist) return exist;
    }
    return false;
}

GGUFDownloader::GGUFDownloader(const std::string& hfEndpoint, const HFSettingsImpl& hfSettings) :
    hfSettings(hfSettings),
    hfEndpoint(hfEndpoint),
    downloadPath(FileSystem::joinPath({this->hfSettings.downloadPath, this->hfSettings.sourceModel})) {}

Status GGUFDownloader::downloadModel() {
    if (this->hfSettings.downloadType != GGUF_DOWNLOAD) {
        SPDLOG_ERROR("Wrong download type selected. Expected GGUF download type.");
        return StatusCode::INTERNAL_ERROR;
    }
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }
    if (!this->hfSettings.ggufFilename.has_value() || this->hfSettings.ggufFilename->empty()) {
        SPDLOG_ERROR("GGUF filename must be specified for GGUF download type, and shouldn't be empty.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto status = checkIfOverwriteAndRemove(this->hfSettings, this->downloadPath);
    if (!status.ok()) {
        return status;
    }
    // now we want to check if model directory already exists
    // if not we will create one
    if (!std::filesystem::is_directory(this->downloadPath)) {
        if (!std::filesystem::create_directories(this->downloadPath)) {
            SPDLOG_ERROR("Failed to create model directory: {}", this->downloadPath);
            return StatusCode::PATH_INVALID;
        }
    }
    if (!this->hfSettings.overwriteModels) {
        auto statusOrBool = checkIfAlreadyExists(this->hfSettings, this->downloadPath);
        if (std::holds_alternative<Status>(statusOrBool)) {
            return std::get<Status>(statusOrBool);
        }
        if (std::get<bool>(statusOrBool)) {
            SPDLOG_DEBUG("Model files already exist and overwrite is disabled, skipping download for model: {}", this->hfSettings.sourceModel);
            return StatusCode::OK;
        }
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
    FtpFile(const char* fname, FILE* str) :
        filename(fname),
        stream(str) {}
    ~FtpFile() {
        if (stream) {
            fclose(stream);
        }
        if (!success) {
            std::filesystem::remove(filename);
        }
    }
    bool success = false;
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
    //fwrite(buffer, size, nmemb, stdout);
    //fwrite("\n", 1, 1, stdout);
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
    time_t currentTime = time(NULL);
    if (dlnow == 0) {
        pcs->started_download = time(NULL);
        pcs->last_print_time = time(NULL);
    }

    if ((dltotal == dlnow) && dltotal < 1000) {
        // Usually with first messages we don't get the full size and we don't want to print progress bar
        // so we assume that until dltotal is less than 1000 we don't have full size
        // otherwise we would print 100% progress bar
        return 0;
    }
    // called multiple times, so we want to print progress bar only once reached 100%
    if (pcs->fullDownloadPrinted) {
        return 0;
    }
    if ((currentTime - pcs->last_print_time < 1) && (dltotal != dlnow)) {
        // we dont want to skip printing progress bar for the 100% but we don't want to spam stdout either
        return 0;
    }
    // FIXME no proper speed
    print_progress(dlnow, dltotal, (dlnow == 0), currentTime - pcs->started_download);
    std::cout.flush();
    pcs->fullDownloadPrinted = (dltotal == dlnow);
    pcs->last_print_time = currentTime;
    return 0;
}

static Status downloadSingleFileWithCurl(const std::string& filePath, const std::string& url) {
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
    //autddo fileCloseGuard = std::unique_ptr<FILE, decltype(&fileClose)>(ftpFile.stream, fileClose);
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
    long http_code = 0;
    CHECK_CURL_CALL(curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code));
    SPDLOG_TRACE("HTTP response code: {}", http_code);
    if (http_code != 200) {
        SPDLOG_ERROR("Failed to download file from URL: {} HTTP response code: {}", url, http_code);
        return StatusCode::PATH_INVALID;
    }
    ftpFile.success = true;
    return StatusCode::OK;
}

std::variant<Status, std::vector<std::string>> GGUFDownloader::createGGUFFilenamesToDownload(const std::string& ggufFilename) {
    std::vector<std::string> filesToDownload;
    // we need to check if ggufFilename is of multipart type (contains 00001-of-N string)
    // we should fail if it is 0000M-of-0000N where M != 1 as we need 1 here
    // we need to extract N
    // note that it could be 00001-of-00002, it could be 00001-of-00212 etc
    // note that it has to be exactly 5 digits for part number and 5 digits for total parts
    std::string multipartExactPattern = R"(.*-(\d{5})-of-(\d{5})\.gguf$)";
    std::smatch match;
    if (std::regex_match(ggufFilename, match, std::regex(multipartExactPattern))) {
        SPDLOG_TRACE("Detected multipart gguf filename: {}", ggufFilename);
        int totalParts = 1;
        if (match.size() != 3) {
            SPDLOG_ERROR("Regex match for multipart filename failed for filename: {}", ggufFilename);
            return StatusCode::INTERNAL_ERROR;
        }
        try {
            if (match[1].str() != "00001") {
                SPDLOG_ERROR("Multipart gguf filename must start with part 00001, got: {} for filename: {}", match[1].str(), ggufFilename);
                return StatusCode::PATH_INVALID;
            }
            auto totalPartsOpt = stoi32(match[2].str());
            if (!totalPartsOpt.has_value() || totalPartsOpt.value() <= 0) {
                SPDLOG_ERROR("Error converting total parts to integer for filename: {}, match: {}", ggufFilename, match[2].str());
                return StatusCode::INTERNAL_ERROR;
            }
            totalParts = totalPartsOpt.value();
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Error converting total parts to integer for filename: {} error: {}", ggufFilename, e.what());
            return StatusCode::INTERNAL_ERROR;
        }
        // now write a loop that will replace the part number in the filename and download all parts from 1 to N eg. 00001 to N where N is totalParts and we have to pad with zeros
        for (int part = 1; part <= totalParts; part++) {
            // create part filename
            std::string partNumberStr = std::to_string(part);
            auto partFilename = preparePartFilename(ggufFilename, part, totalParts);
            filesToDownload.push_back(partFilename);
        }
    } else {
        filesToDownload.push_back(ggufFilename);
    }
    return filesToDownload;
}

Status GGUFDownloader::downloadWithCurl(const std::string& hfEndpoint, const std::string& modelName, const std::string& filenamePrefix, const std::string& ggufFilename, const std::string& downloadPath) {
    auto filesOrStatus = createGGUFFilenamesToDownload(ggufFilename);
    if (std::holds_alternative<Status>(filesOrStatus)) {
        return std::get<Status>(filesOrStatus);
    }
    auto& filesToDownload = std::get<std::vector<std::string>>(filesOrStatus);
    size_t partNo = 1;
    for (const auto& file : filesToDownload) {
        // construct url
        SPDLOG_TRACE("hfEndpoint: {} modelName: {} filenamePrefix: {} file: {}, downloadPath:{}", hfEndpoint, modelName, filenamePrefix, file, downloadPath);
        std::string url = hfEndpoint + modelName + filenamePrefix + file;
        // construct filepath
        auto filePath = FileSystem::joinPath({downloadPath, file});
        SPDLOG_DEBUG("Downloading part {}/{} filename: {} url:{}", partNo, filesToDownload.size(), file, url);
        auto status = downloadSingleFileWithCurl(filePath, url);
        if (!status.ok()) {
            return status;
        }
        SPDLOG_TRACE("cURL download completed for model: {} part: {}/{} to path: {}", partNo, filesToDownload.size(), modelName, filePath);
        ++partNo;
    }
    SPDLOG_TRACE("cURL download completed for model: {}", modelName);
    return StatusCode::OK;
}
std::string GGUFDownloader::preparePartFilename(const std::string& ggufFilename, int part, int totalParts) {
    if (part <= 0 || totalParts <= 1 || part > totalParts || totalParts > 99999 || part > 99999) {
        throw std::invalid_argument("Invalid part or totalParts values");
    }
    // example of strings
    // ggufFilename qwen2.5-3b-instruct-fp16-00001-of-00002.gguf
    // ggufFilename qwen2.5-b-instruct-fp16-00001-of-23232.gguf
    // ggufFilename qwen3-b-instruct-fp16-00001-of-00232.gguf
    // so we want to replace 00001-of-[0-9]{4}[2-9] part with appropriate part number
    // we need to pad part number with zeros to match the length of 5
    std::string partNumberStr = std::to_string(part);
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << partNumberStr;
    std::string numberPadded = oss.str();
    std::string constructedFilename = ggufFilename;
    auto it = ggufFilename.find("-00001-");
    if (it == std::string::npos) {
        throw std::invalid_argument("Invalid ggufFilename format, cannot find -00001- part");
    }
    constructedFilename.replace(ggufFilename.find("-00001-"), 7, "-" + numberPadded + "-");
    return constructedFilename;
}

}  // namespace ovms
