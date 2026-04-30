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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../capi_frontend/server_settings.hpp"
#include "src/filesystem/filesystem.hpp"
#include "src/filesystem/localfilesystem.hpp"
#include "../logging.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "curl_downloader.hpp"

namespace ovms {

Status GGUFDownloader::checkIfOverwriteAndRemove() {
    auto lfstatus = StatusCode::OK;
    if (this->overwriteModels && std::filesystem::is_directory(this->downloadPath)) {
        auto allSpecifiedQuantizationPartsFilenamesOrStatus = GGUFDownloader::createGGUFFilenamesToDownload(this->ggufFilename.value());
        if (std::holds_alternative<Status>(allSpecifiedQuantizationPartsFilenamesOrStatus)) {
            return std::get<Status>(allSpecifiedQuantizationPartsFilenamesOrStatus);
        }
        auto& allSpecifiedQuantizationPartsFilenames = std::get<std::vector<std::string>>(allSpecifiedQuantizationPartsFilenamesOrStatus);
        LocalFileSystem lfs;
        for (const auto& file : allSpecifiedQuantizationPartsFilenames) {
            auto filePath = FileSystem::joinPath({this->downloadPath, file});
            SPDLOG_TRACE("Checking if model file exists for overwrite: {}", filePath);
            if (std::filesystem::exists(filePath)) {
                SPDLOG_TRACE("Model file already exists and will be removed due to overwrite flag: {}", filePath);
                lfstatus = lfs.deleteFileFolder(filePath);
                if (lfstatus != StatusCode::OK) {
                    SPDLOG_ERROR("Error occurred while deleting path: {} reason: {}", this->downloadPath, lfstatus);
                } else {
                    SPDLOG_TRACE("Path deleted: {}", this->downloadPath);
                }
            }
        }
    }
    return lfstatus;
}

std::variant<Status, bool> GGUFDownloader::checkIfAlreadyExists(const std::optional<std::string> ggufFilename, const std::string& path) {
    auto ggufFilesOrStatus = createGGUFFilenamesToDownload(ggufFilename.value());
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
        if (!status.ok())
            return status;
        if (exist)
            return exist;
    }
    return false;
}

GGUFDownloader::GGUFDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, bool inOverwrite, const std::optional<std::string> inGgufFilename, const std::string& hfEndpoint) :
    IModelDownloader(inSourceModel, inDownloadPath, inOverwrite),
    ggufFilename(inGgufFilename),
    hfEndpoint(hfEndpoint) {}

Status GGUFDownloader::downloadModel() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }
    if (!this->ggufFilename.has_value() || this->ggufFilename->empty()) {
        SPDLOG_ERROR("GGUF filename must be specified for GGUF download type, and shouldn't be empty.");
        return StatusCode::INTERNAL_ERROR;
    }
    auto status = this->checkIfOverwriteAndRemove();
    if (!status.ok()) {
        return status;
    }
    bool exists = false;
    status = LocalFileSystem::exists(this->downloadPath, &exists);
    if (!exists || !status.ok()) {
        if (!std::filesystem::create_directories(this->downloadPath)) {
            SPDLOG_ERROR("Failed to create model directory: {}", this->downloadPath);
            return StatusCode::DIRECTORY_NOT_CREATED;
        }
    } else if (!std::filesystem::is_directory(this->downloadPath)) {
        SPDLOG_ERROR("Model path exists and is not a directory: {}", this->downloadPath);
        return StatusCode::DIRECTORY_NOT_CREATED;
    }
    if (!this->overwriteModels) {
        auto statusOrBool = checkIfAlreadyExists(this->ggufFilename, this->downloadPath);
        if (std::holds_alternative<Status>(statusOrBool)) {
            return std::get<Status>(statusOrBool);
        }
        if (std::get<bool>(statusOrBool)) {
            SPDLOG_DEBUG("Model files already exist and overwrite is disabled, skipping download for model: {}", this->sourceModel);
            return StatusCode::OK;
        }
    }
    status = downloadWithCurl(this->hfEndpoint, this->sourceModel, "/resolve/main/", this->ggufFilename.value(), this->downloadPath);
    if (!status.ok()) {
        SPDLOG_ERROR("Error occurred while downloading GGUF model: {} reason: {}", this->sourceModel, status.string());
        return status;
    }
    return StatusCode::OK;
}

std::variant<Status, std::vector<std::string>> GGUFDownloader::createGGUFFilenamesToDownload(const std::string& ggufFilename) {
    std::vector<std::string> filesToDownload;
    // we need to check if ggufFilename is of multipart type (contains 00001-of-N string)
    // we should fail if it is 0000M-of-0000N where M != 1 as we need 1 here
    // we need to extract N
    // note that it could be 00001-of-00002, it could be 00001-of-00212 etc
    // note that it has to be exactly 5 digits for part number and 5 digits for total parts
    // we first should check if this at least tries to be multipart with regex "-of-" as not to
    // try to use web with curl unnecessarily
    std::string multipartPreCheckPattern = R"(.*-of-.*$)";
    std::smatch preCheckMatch;
    if (!std::regex_match(ggufFilename, preCheckMatch, std::regex(multipartPreCheckPattern))) {
        // not multipart
        filesToDownload.push_back(ggufFilename);
        return filesToDownload;
    }
    std::string multipartExactPattern = R"(.*-(\d{5})-of-(\d{5})\.gguf$)";
    std::smatch match;
    if (std::regex_match(ggufFilename, match, std::regex(multipartExactPattern))) {
        SPDLOG_TRACE("Detected multipart gguf filename: {}", ggufFilename);
        int totalParts = 1;
        if (match.size() != 3) {
            SPDLOG_ERROR("Regex match for multipart filename failed for filename: {}", ggufFilename);
            return StatusCode::INTERNAL_ERROR;
        }
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
        for (int part = 1; part <= totalParts; part++) {
            std::string partNumberStr = std::to_string(part);
            auto partFilenameOrStatus = preparePartFilename(ggufFilename, part, totalParts);
            if (std::holds_alternative<Status>(partFilenameOrStatus)) {
                // shouldn't happen as we already validated regex
                return std::get<Status>(partFilenameOrStatus);
            }
            filesToDownload.push_back(std::get<std::string>(partFilenameOrStatus));
        }
    } else {
        SPDLOG_ERROR("Invalid multipart gguf filename format: {}", ggufFilename);
        return StatusCode::PATH_INVALID;
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
        auto status = downloadFileWithCurl(url, filePath);
        if (!status.ok()) {
            return status;
        }
        SPDLOG_TRACE("cURL download completed for model: {} part: {}/{} to path: {}", partNo, filesToDownload.size(), modelName, filePath);
        ++partNo;
    }
    SPDLOG_TRACE("cURL download completed for model: {}", modelName);
    return StatusCode::OK;
}
std::variant<Status, std::string> GGUFDownloader::preparePartFilename(const std::string& ggufFilename, int part, int totalParts) {
    if (part <= 0 || totalParts <= 1 || part > totalParts || totalParts > 99999 || part > 99999) {
        return Status(StatusCode::INTERNAL_ERROR, "Invalid part or totalParts values");
    }
    // example of strings
    // ggufFilename qwen2.5-3b-instruct-fp16-00001-of-00002.gguf
    // ggufFilename qwen2.5-b-instruct-fp16-00001-of-23232.gguf
    // so we want to replace 00001-of-[0-9]{4}[2-9] part with appropriate part number
    // we need to pad part number with zeros to match the length of 5
    std::string partNumberStr = std::to_string(part);
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << partNumberStr;
    std::string numberPadded = oss.str();
    std::string constructedFilename = ggufFilename;
    auto it = ggufFilename.find("-00001-");
    if (it == std::string::npos) {
        return Status(StatusCode::INTERNAL_ERROR, "Invalid ggufFilename format, cannot find -00001- part");
    }
    constructedFilename.replace(ggufFilename.find("-00001-"), 7, "-" + numberPadded + "-");
    return constructedFilename;
}
}  // namespace ovms
