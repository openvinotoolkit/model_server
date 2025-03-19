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
#ifndef SRC_LIBGT2_LIBGT2_HPP_
#define SRC_LIBGT2_LIBGT2_HPP_
#include <string>

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

class HfDownloader {
public:
    int cloneRepository();
    void setSourceModel(std::string inSourceModel);
    void setRepositoryPath(std::string inRepoPath);
    void setPullHfModelMode(bool isOn);
    bool isPullHfModelModeOn();

private:
    std::string sourceModel;
    std::string repoPath;
    bool pullHfModelMode;
    std::string hfEndpoint;
    std::string repoUrl;

    void UpdateRepoUrl();
    void SetHfEndpoint();
    void GetRepositoryUrlWithPassword(std::string& passRepoUrl);
    bool CheckIfProxySet();
    bool CheckIfTokenSet();
};

}  // namespace ovms
#ifdef __cplusplus
}
#endif
#endif  // SRC_LIBGT2_LIBGT2_HPP_
