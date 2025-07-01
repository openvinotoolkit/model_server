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
#include "libgit2.hpp"

#include <iostream>
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

#include "cmd_exec.hpp"
#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../stringutils.hpp"
#include "../status.hpp"

#ifndef PRIuZ
/* Define the printf format specifier to use for size_t output */
#if defined(_MSC_VER) || defined(__MINGW32__)
#define PRIuZ "Iu"
#else
#define PRIuZ "zu"
#endif
#endif

namespace ovms {

// C-style callback functions section used in libgt2 library STARTS ********************************
typedef struct progress_data {
    git_indexer_progress fetch_progress;
    size_t completed_steps;
    size_t total_steps;
    const char* path;
} progress_data;

static void print_progress(const progress_data* pd) {
    int network_percent = pd->fetch_progress.total_objects > 0 ? (100 * pd->fetch_progress.received_objects) / pd->fetch_progress.total_objects : 0;
    int index_percent = pd->fetch_progress.total_objects > 0 ? (100 * pd->fetch_progress.indexed_objects) / pd->fetch_progress.total_objects : 0;

    int checkout_percent = pd->total_steps > 0
                               ? (int)((100 * pd->completed_steps) / pd->total_steps)
                               : 0;
    size_t kbytes = pd->fetch_progress.received_bytes / 1024;

    if (pd->fetch_progress.total_objects &&
        pd->fetch_progress.received_objects == pd->fetch_progress.total_objects) {
        printf("Resolving deltas %u/%u\r",
            pd->fetch_progress.indexed_deltas,
            pd->fetch_progress.total_deltas);
    } else {
        printf("net %3d%% (%4" PRIuZ " kb, %5u/%5u)  /  idx %3d%% (%5u/%5u)  /  chk %3d%% (%4" PRIuZ "/%4" PRIuZ ")%s\n",
            network_percent, kbytes,
            pd->fetch_progress.received_objects, pd->fetch_progress.total_objects,
            index_percent, pd->fetch_progress.indexed_objects, pd->fetch_progress.total_objects,
            checkout_percent,
            pd->completed_steps, pd->total_steps,
            pd->path);
    }
}

static int sideband_progress(const char* str, int len, void* payload) {
    (void)payload;  //  unused

    printf("remote: %.*s", len, str);
    fflush(stdout);
    return 0;
}

static int fetch_progress(const git_indexer_progress* stats, void* payload) {
    progress_data* pd = (progress_data*)payload;
    pd->fetch_progress = *stats;
    print_progress(pd);
    return 0;
}

static void checkout_progress(const char* path, size_t cur, size_t tot, void* payload) {
    progress_data* pd = (progress_data*)payload;
    pd->completed_steps = cur;
    pd->total_steps = tot;
    pd->path = path;
    print_progress(pd);
}

// Callback for clone authentication - will be used when password is not set in repo_url
// Does not work with LFS download as it requires additional authentication when password is not set in repository url
int cred_acquire_cb(git_credential** out,
    const char* url,
    const char* username_from_url,
    unsigned int allowed_types,
    void* payload) {
    char *username = NULL, *password = NULL;
    int error = -1;

    fprintf(stdout, "Authentication is required for repository clone.\n");
    if (allowed_types & GIT_CREDENTIAL_USERPASS_PLAINTEXT) {
        const char* env_cred = std::getenv("HF_TOKEN");
        if (env_cred) {
#ifdef __linux__
            username = strdup(env_cred);
            password = strdup(username);
#elif _WIN32
            username = _strdup(env_cred);
            password = _strdup(username);
#endif
        } else {
            fprintf(stderr, "HF_TOKEN env variable is not set.\n");
            return -1;
        }
        error = git_credential_userpass_plaintext_new(out, username, password);
        if (error < 0) {
            fprintf(stderr, "Creating credentials failed.\n");
            error = -1;
        }
    } else {
        fprintf(stderr, "Only USERPASS_PLAINTEXT supported in OVMS.\n");
        return 1;
    }

    free(username);
    free(password);
    return error;
}

// C-style callback functions section used in libgt2 library ENDS ********************************

#define IF_ERROR_SET_MSG_AND_RETURN()                                 \
    do {                                                              \
        if (this->status < 0) {                                       \
            const git_error* err = git_error_last();                  \
            const char* msg = err ? err->message : "unknown failure"; \
            errMsg = std::string(msg);                                \
            return;                                                   \
        } else {                                                      \
            errMsg = "";                                              \
        }                                                             \
    } while (0)

Libgt2InitGuard::Libgt2InitGuard(const Libgit2Options& opts) {
    SPDLOG_DEBUG("Initializing libgit2");
    this->status = git_libgit2_init();
    IF_ERROR_SET_MSG_AND_RETURN();
    SPDLOG_TRACE("Setting libgit2 server connection timeout:{}", opts.serverConnectTimeoutMs);
    this->status = git_libgit2_opts(GIT_OPT_SET_SERVER_CONNECT_TIMEOUT, opts.serverConnectTimeoutMs);
    IF_ERROR_SET_MSG_AND_RETURN();
    SPDLOG_TRACE("Setting libgit2 server timeout:{}", opts.serverTimeoutMs);
    this->status = git_libgit2_opts(GIT_OPT_SET_SERVER_TIMEOUT, opts.serverTimeoutMs);
    IF_ERROR_SET_MSG_AND_RETURN();
}

Libgt2InitGuard::~Libgt2InitGuard() {
    SPDLOG_DEBUG("Shutdown libgit2");
    git_libgit2_shutdown();
}

const std::string PROTOCOL_SEPARATOR = "://";

bool HfDownloader::CheckIfProxySet() {
    if (this->httpProxy != "")
        return true;
    return false;
}

std::string HfDownloader::GetRepositoryUrlWithPassword() {
    std::string repoPass = "";
    if (this->hfToken != "") {
        repoPass += this->hfToken + ":" + this->hfToken + "@";
    } else {
        SPDLOG_DEBUG("HF_TOKEN environment variable not set");
        return this->hfEndpoint + this->sourceModel;
    }

    std::string outputWithPass = "";
    size_t match = this->hfEndpoint.find(PROTOCOL_SEPARATOR);
    if (match != std::string::npos) {
        // https://huggingface.co
        // protocol[match]//address
        std::string protocol = this->hfEndpoint.substr(0, match);
        std::string address = this->hfEndpoint.substr(match + PROTOCOL_SEPARATOR.size());
        outputWithPass = protocol + PROTOCOL_SEPARATOR + repoPass + address + this->sourceModel;
    } else {
        outputWithPass = repoPass + this->hfEndpoint + this->sourceModel;
    }

    return outputWithPass;
}

std::string HfDownloader::GetRepoUrl() {
    std::string repoUrl = "";
    repoUrl += this->hfEndpoint + this->sourceModel;
    return repoUrl;
}

HfDownloader::HfDownloader() {
    this->sourceModel = "";
    this->downloadPath = "";
    this->hfEndpoint = "";
    this->hfToken = "";
    this->httpProxy = "";
    this->overwriteModels = false;
}

std::string HfDownloader::getGraphDirectory() {
    return this->downloadPath;
}

std::string HfDownloader::getGraphDirectory(const std::string& inDownloadPath, const std::string& inSourceModel) {
    std::string fullPath = FileSystem::joinPath({inDownloadPath, inSourceModel});
    return fullPath;
}

HfDownloader::HfDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const std::string& inHfEndpoint, const std::string& inHfToken, const std::string& inHttpProxy, bool inOverwrite) {
    this->sourceModel = inSourceModel;
    this->downloadPath = HfDownloader::getGraphDirectory(inDownloadPath, inSourceModel);
    this->hfEndpoint = inHfEndpoint;
    this->hfToken = inHfToken;
    this->httpProxy = inHttpProxy;
    this->overwriteModels = inOverwrite;
}

Status HfDownloader::RemoveReadonlyFileAttributeFromDir(const std::string& directoryPath) {
    for (const std::filesystem::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(directoryPath)) {
        try {
            std::filesystem::permissions(dir_entry, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Failed to set permission for: {} .Exception caught: {}", dir_entry.path().string(), e.what());
            return StatusCode::PATH_INVALID;
        }
    }

    return StatusCode::OK;
}

Status HfDownloader::checkIfOverwriteAndRemove(const std::string& path) {
    auto lfstatus = StatusCode::OK;
    if (this->overwriteModels && std::filesystem::is_directory(path)) {
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

Status HfDownloader::checkRequiredToolsArePresent() {
    std::string cmd = "git --version";
    int retCode = -1;
    std::string output = exec_cmd(cmd, retCode);
    if (retCode != 0 || output.find("git version ") == std::string::npos) {
        SPDLOG_DEBUG("Command output {}", output);
        SPDLOG_ERROR("Required git executable is not present. Please add git from ovms package to PATH.");
        return StatusCode::HF_FAILED_TO_INIT_GIT;
    }

    cmd = "git-lfs --version";
    retCode = -1;
    output = exec_cmd(cmd, retCode);
    if (retCode != 0 || output.find("git-lfs/") == std::string::npos) {
        SPDLOG_DEBUG("Command output {}", output);
        SPDLOG_ERROR("Required git-lfs executable is not present. Please add git-lfs from ovms package to PATH.");
        return StatusCode::HF_FAILED_TO_INIT_GIT_LFS;
    }

    return StatusCode::OK;
}

Status HfDownloader::cloneRepository() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    // Repository exists and we do not want to overwrite
    if (std::filesystem::is_directory(this->downloadPath) && !this->overwriteModels) {
        std::cout << "Path already exists on local filesystem. Skipping download to path: " << this->downloadPath << std::endl;
        return StatusCode::OK;
    }

    auto status = checkRequiredToolsArePresent();
    if (!status.ok()) {
        return status;
    }

    status = checkIfOverwriteAndRemove(this->downloadPath);
    if (!status.ok()) {
        return status;
    }

    SPDLOG_DEBUG("Downloading to path: {}", this->downloadPath);
    progress_data pd = {{0}};
    git_repository* cloned_repo = NULL;
    git_clone_options clone_opts = GIT_CLONE_OPTIONS_INIT;
    git_checkout_options checkout_opts = GIT_CHECKOUT_OPTIONS_INIT;

    /* Set up options */
    checkout_opts.checkout_strategy = GIT_CHECKOUT_SAFE;
    checkout_opts.progress_cb = checkout_progress;
    checkout_opts.progress_payload = &pd;
    clone_opts.checkout_opts = checkout_opts;
    clone_opts.fetch_opts.callbacks.sideband_progress = sideband_progress;
    clone_opts.fetch_opts.callbacks.transfer_progress = &fetch_progress;
    clone_opts.fetch_opts.callbacks.credentials = cred_acquire_cb;
    clone_opts.fetch_opts.callbacks.payload = &pd;

    // Use proxy
    if (CheckIfProxySet()) {
        clone_opts.fetch_opts.proxy_opts.type = GIT_PROXY_SPECIFIED;
        clone_opts.fetch_opts.proxy_opts.url = this->httpProxy.c_str();
        SPDLOG_DEBUG("Download using https_proxy settings");
    } else {
        SPDLOG_DEBUG("Download with https_proxy not set");
    }

    std::string repoUrl = GetRepoUrl();
    SPDLOG_DEBUG("Downloading from url: {}", repoUrl.c_str());
    std::string passRepoUrl = GetRepositoryUrlWithPassword();
    const char* url = passRepoUrl.c_str();
    const char* path = this->downloadPath.c_str();
    SPDLOG_TRACE("Starting git clone to: {}", path);
    int error = git_clone(&cloned_repo, url, path, &clone_opts);
    SPDLOG_TRACE("Ended git clone");
    if (error != 0) {
        const git_error* err = git_error_last();
        if (err)
            SPDLOG_ERROR("Libgit2 clone error: {} message: {}", err->klass, err->message);
        else
            SPDLOG_ERROR("Libgit2 clone error: {}", error);

        return StatusCode::HF_GIT_CLONE_FAILED;
    } else if (cloned_repo) {
        git_repository_free(cloned_repo);
    }

    // libgit2 clone sets readonly attributes
    status = RemoveReadonlyFileAttributeFromDir(this->downloadPath);
    if (!status.ok()) {
        return status;
    }

    return StatusCode::OK;
}

}  // namespace ovms
