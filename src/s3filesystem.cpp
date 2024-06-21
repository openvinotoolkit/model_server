// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "s3filesystem.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "logging.hpp"
#include "stringutils.hpp"

namespace ovms {

namespace fs = std::filesystem;

StatusCode S3FileSystem::parsePath(const std::string& path, std::string* bucket, std::string* object) {
    return StatusCode::OK;
}

S3FileSystem::S3FileSystem(const std::string& s3_path) :
    s3_regex_(FileSystem::S3_URL_PREFIX + "([0-9a-zA-Z-.]+):([0-9]+)/([0-9a-z.-]+)(((/"
                                          "[0-9a-zA-Z.-_]+)*)?)"),
    proxy_regex_("^(https?)://(([^:]{1,128}):([^@]{1,256})@)?([^:/]{1,255})(:([0-9]{1,5}))?/?") {
}

S3FileSystem::~S3FileSystem() {
}

StatusCode S3FileSystem::fileExists(const std::string& path, bool* exists) {
    return StatusCode::OK;
}

StatusCode S3FileSystem::isDirectory(const std::string& path, bool* is_dir) {
    return StatusCode::OK;
}

StatusCode S3FileSystem::getDirectoryContents(const std::string& path, std::set<std::string>* contents) {
    return StatusCode::OK;
}

StatusCode S3FileSystem::getDirectorySubdirs(const std::string& path, std::set<std::string>* subdirs) {
    return StatusCode::OK;
}

StatusCode S3FileSystem::getDirectoryFiles(const std::string& path, std::set<std::string>* files) {
    return StatusCode::OK;
}

StatusCode S3FileSystem::readTextFile(const std::string& path, std::string* contents) {
    return StatusCode::OK;
}

StatusCode S3FileSystem::downloadFileFolder(const std::string& path, const std::string& local_path) {
    return StatusCode::OK;
}

StatusCode S3FileSystem::downloadModelVersions(const std::string& path, std::string* b, const std::vector<long int>& a){
    return StatusCode::OK;
}

StatusCode S3FileSystem::deleteFileFolder(const std::string& path) {
    return StatusCode::OK;
}
}  // namespace ovms
