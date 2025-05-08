#pragma once
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

namespace ovms {
class Status;

/*
 * libgit2 options. 0 is the default value
 */
struct Libgit2Options {
    int serverConnectTimeoutMs = 0;
    int serverTimeoutMs = 0;
};

struct Libgt2InitGuard {
    int status;
    std::string errMsg;
    Libgt2InitGuard(const Libgit2Options& opts);
    ~Libgt2InitGuard();
};

class HfDownloader {
public:
    HfDownloader(const std::string& sourceModel, const std::string& downloadPath, const std::string& hfEndpoint, const std::string& hfToken, const std::string& httpProxy);
    Status cloneRepository();

protected:
    std::string sourceModel;
    std::string downloadPath;
    std::string hfEndpoint;
    std::string hfToken;
    std::string httpProxy;

    HfDownloader();
    std::string GetRepoUrl();
    std::string GetRepositoryUrlWithPassword();
    bool CheckIfProxySet();
};
}  // namespace ovms
