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
#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <assert.h>
#include <fcntl.h>
#include <git2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "model_downloader.hpp"

namespace ovms {
class Status;
namespace fs = std::filesystem;

/*
 * libgit2 options. 0 is the default value
 */
struct Libgit2Options {
    int serverConnectTimeoutMs = 0;
    int serverTimeoutMs = 0;
    std::string sslCertificateLocation = "";
};

struct Libgt2InitGuard {
    int status;
    std::string errMsg;
    bool countedAsInitialized = false;
    Libgt2InitGuard(const Libgit2Options& opts);
    ~Libgt2InitGuard();
};

class HfDownloader : public IModelDownloader {
public:
    HfDownloader(const std::string& sourceModel, const std::string& downloadPath, const std::string& hfEndpoint, const std::string& hfToken, const std::string& httpProxy, bool inOverwrite);
    Status downloadModel() override;

protected:
    const std::string hfEndpoint;
    const std::string hfToken;
    const std::string httpProxy;

    std::string GetRepoUrl();
    std::string GetRepositoryUrlWithPassword();
    bool CheckIfProxySet();
    Status RemoveReadonlyFileAttributeFromDir(const std::string& directoryPath);
    Status CheckRepositoryStatus(bool checkUntracked);
};

namespace libgit2 {
    void rtrimCrLfWhitespace(std::string& s);
    bool containsCaseInsensitive(const std::string& hay, const std::string& needle);
    bool readFirstThreeLines(const fs::path& p, std::vector<std::string>& outLines);
    bool fileHasLfsKeywordsFirst3Positional(const fs::path& p);
    fs::path makeRelativeToBase(const fs::path& path, const fs::path& base);
    std::vector<fs::path> findLfsLikeFiles(const std::string& directory, bool recursive = true);
}
}  // namespace ovms
