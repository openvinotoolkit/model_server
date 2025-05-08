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
#pragma once
#ifndef SRC_LIBGT2_LIBGT2_HPP_
#define SRC_LIBGT2_LIBGT2_HPP_
#include <string>
#include <memory>

#include <assert.h>
#include <fcntl.h>
#include <git2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

namespace ovms {
class Status;

class Libgt2InitGuard {
public:
    int status;
    std::string errMsg;
    Libgt2InitGuard() {
        this->status = git_libgit2_init();
        if (this->status < 0) {
            const git_error* err = git_error_last();
            const char* msg = err ? err->message : "unknown failure";
            errMsg = std::string(msg);
        } else {
            errMsg = "";
        }
    }
    ~Libgt2InitGuard() {
        git_libgit2_shutdown();
    }
};

class HfDownloader {
public:
    HfDownloader(const std::string& sourceModel, const std::string& downloadPath, const std::string& hfEndpoint, const std::string& hfToken, const std::string& httpProxy);
    Status cloneRepository();
    std::string getGraphDirectory();

protected:
    std::string sourceModel;
    std::string downloadPath;
    std::string hfEndpoint;
    std::string hfToken;
    std::string httpProxy;

    HfDownloader();
    std::string GetRepoUrl();
    std::string GetRepositoryUrlWithPassword();
    std::string getGraphDirectory(const std::string& inDownloadPath, const std::string& inSourceModel);
    bool CheckIfProxySet();
};

}  // namespace ovms
#ifdef __cplusplus
}
#endif
#endif  // SRC_LIBGT2_LIBGT2_HPP_
