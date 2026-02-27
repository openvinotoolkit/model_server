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

// Callback for clone authentication - will be used when password is not set in repo_url
// Does not work with LFS download as it requires additional authentication when password is not set in repository url
int cred_acquire_cb(git_credential** out,
    const char* url,
    const char* username_from_url,
    unsigned int allowed_types,
    void* payload) {
    char *username = NULL, *password = NULL;
    int error = -1;

    fprintf(stdout, "Authentication is required for repository clone or model is missing.\n");
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
    if (opts.sslCertificateLocation != "") {
        SPDLOG_TRACE("Setting libgit2 ssl certificate location:{}", opts.sslCertificateLocation);
        this->status = git_libgit2_opts(GIT_OPT_SET_SSL_CERT_LOCATIONS, NULL, opts.sslCertificateLocation.c_str());
        IF_ERROR_SET_MSG_AND_RETURN();
    }
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

HfDownloader::HfDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const std::string& inHfEndpoint, const std::string& inHfToken, const std::string& inHttpProxy, bool inOverwrite) :
    IModelDownloader(inSourceModel, inDownloadPath, inOverwrite),
    hfEndpoint(inHfEndpoint),
    hfToken(inHfToken),
    httpProxy(inHttpProxy) {}

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

Status HfDownloader::CheckRepositoryStatus() {
    git_repository *repo = NULL;
    int error = git_repository_open_ext(&repo, this->downloadPath.c_str(), 0, NULL);
    if (error < 0) {
        const git_error *err = git_error_last();
        if (err)
            SPDLOG_ERROR("Repository open failed: {} {}", err->klass, err->message);
        else
            SPDLOG_ERROR("Repository open failed: {}", error);
        if (repo) git_repository_free(repo);

        return StatusCode::HF_GIT_STATUS_FAILED;
    }
    // HEAD state info
    bool is_detached = git_repository_head_detached(repo) == 1;
    bool is_unborn   = git_repository_head_unborn(repo) == 1;
    
    // Collect status (staged/unstaged/untracked)
    git_status_options opts = GIT_STATUS_OPTIONS_INIT;
    
    opts.show  = GIT_STATUS_SHOW_INDEX_AND_WORKDIR;
    opts.flags = GIT_STATUS_OPT_INCLUDE_UNTRACKED        // include untracked files // | GIT_STATUS_OPT_RENAMES_HEAD_TO_INDEX    // detect renames HEAD->index - not required currently and impacts performance
               | GIT_STATUS_OPT_SORT_CASE_SENSITIVELY;   

    
    git_status_list* status_list = nullptr;
    error = git_status_list_new(&status_list, repo, &opts);
    if (error != 0) {
        return StatusCode::HF_GIT_STATUS_FAILED;
    }

    size_t staged = 0, unstaged = 0, untracked = 0, conflicted = 0;
    const size_t n = git_status_list_entrycount(status_list); // iterate entries
    for (size_t i = 0; i < n; ++i) {
        const git_status_entry* e = git_status_byindex(status_list, i);
        unsigned s = e->status;

        // Staged (index) changes
        if (s & (GIT_STATUS_INDEX_NEW     |
                 GIT_STATUS_INDEX_MODIFIED|
                 GIT_STATUS_INDEX_DELETED |
                 GIT_STATUS_INDEX_RENAMED |
                 GIT_STATUS_INDEX_TYPECHANGE))
            ++staged;

        // Unstaged (workdir) changes
        if (s & (GIT_STATUS_WT_MODIFIED   |
                 GIT_STATUS_WT_DELETED    |
                 GIT_STATUS_WT_RENAMED    |
                 GIT_STATUS_WT_TYPECHANGE))
            ++unstaged;

        // Untracked
        if (s & GIT_STATUS_WT_NEW)
            ++untracked;

        // libgit2 will also flag conflicted entries via status/diff machinery
        if (s & GIT_STATUS_CONFLICTED)
            ++conflicted;
    }

    std::stringstream ss;
    ss << "HEAD state      : "
              << (is_unborn ? "unborn (no commits)" : (is_detached ? "detached" : "attached"))
              << "\n";
    ss << "Staged changes  : " << staged     << "\n";
    ss << "Unstaged changes: " << unstaged   << "\n";
    ss << "Untracked files : " << untracked  << "\n";
    if (conflicted) ss << " (" << conflicted << " paths flagged)";

    SPDLOG_DEBUG(ss.str());
    git_status_list_free(status_list);

    if (is_unborn || is_detached || staged || unstaged || untracked || conflicted) {
        return StatusCode::HF_GIT_STATUS_UNCLEAN; 
    }
    return StatusCode::OK;
}

static int print_changed_and_untracked(git_repository *repo) {
    int error = 0;
    git_status_list *statuslist = NULL;

    git_status_options opts;
    error = git_status_options_init(&opts, GIT_STATUS_OPTIONS_VERSION);
    if (error < 0) return error;

    // Choose what to include
    opts.show  = GIT_STATUS_SHOW_INDEX_AND_WORKDIR; // consider both index and working dir
    opts.flags =
        GIT_STATUS_OPT_INCLUDE_UNTRACKED |        // include untracked files
        GIT_STATUS_OPT_RECURSE_UNTRACKED_DIRS |   // recurse into untracked dirs
        GIT_STATUS_OPT_INCLUDE_IGNORED |          // (optional) include ignored if you want to see them
        GIT_STATUS_OPT_RENAMES_HEAD_TO_INDEX |    // detect renames in index
        GIT_STATUS_OPT_RENAMES_INDEX_TO_WORKDIR | // detect renames in workdir
        GIT_STATUS_OPT_SORT_CASE_SENSITIVELY;     // stable ordering

    // If you want to limit to certain paths/patterns, set opts.pathspec here.

    if ((error = git_status_list_new(&statuslist, repo, &opts)) < 0)
        return error;

    size_t count = git_status_list_entrycount(statuslist);
    for (size_t i = 0; i < count; i++) {
        const git_status_entry *e = git_status_byindex(statuslist, i);
        if (!e) continue;

        unsigned int s = e->status;

        // Consider “changed” as anything that’s not current in HEAD/INDEX/WT:
        // You can tailor this to your exact definition.
        int is_untracked =
            (s & GIT_STATUS_WT_NEW) != 0; // working tree new (untracked)
        int is_workdir_changed =
            (s & (GIT_STATUS_WT_MODIFIED |
                  GIT_STATUS_WT_DELETED  |
                  GIT_STATUS_WT_RENAMED  |
                  GIT_STATUS_WT_TYPECHANGE)) != 0;
        int is_index_changed =
            (s & (GIT_STATUS_INDEX_NEW      |
                  GIT_STATUS_INDEX_MODIFIED |
                  GIT_STATUS_INDEX_DELETED  |
                  GIT_STATUS_INDEX_RENAMED  |
                  GIT_STATUS_INDEX_TYPECHANGE)) != 0;

        if (!(is_untracked || is_workdir_changed || is_index_changed))
            continue;

        // Prefer the most relevant delta for the path
        const git_diff_delta *delta = NULL;
        if (is_workdir_changed && e->index_to_workdir)
            delta = e->index_to_workdir;
        else if (is_index_changed && e->head_to_index)
            delta = e->head_to_index;
        else if (is_untracked && e->index_to_workdir)
            delta = e->index_to_workdir;

        if (!delta) continue;

        // For renames, old_file and new_file may differ; typically you want new_file.path
        const char *path = delta->new_file.path ? delta->new_file.path
                                                : delta->old_file.path;

        // Print or collect the filename
        SPDLOG_INFO("is_untracked {} is_workdir_changed {} is_index_changed {} File {} ", is_untracked, is_workdir_changed, is_index_changed, path);
    }

    git_status_list_free(statuslist);
    return 0;
}

int HfDownloader::CheckRepositoryForResume() {
    git_repository *repo = NULL;
    int error = git_repository_open_ext(&repo, this->downloadPath.c_str(), 0, NULL);
    if (error < 0) {
        const git_error *err = git_error_last();
        if (err)
            SPDLOG_ERROR("Repository open failed: {} {}", err->klass, err->message);
        else
            SPDLOG_ERROR("Repository open failed: {}", error);
        if (repo) git_repository_free(repo);

        return error;
    }

    error = print_changed_and_untracked(repo);
    if (error < 0) {
        const git_error *err = git_error_last();
        if (err)
            SPDLOG_ERROR("Print changed files failed: {} {}", err->klass, err->message);
        else
            SPDLOG_ERROR("Print changed files failed: {}", error);
    }

    if (repo) git_repository_free(repo);
    return error;
}

Status HfDownloader::downloadModel() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    // Repository exists and we do not want to overwrite
    if (std::filesystem::is_directory(this->downloadPath) && !this->overwriteModels) {
        CheckRepositoryForResume();


        std::cout << "Path already exists on local filesystem. Skipping download to path: " << this->downloadPath << std::endl;
        return StatusCode::OK;
    }

    auto status = IModelDownloader::checkIfOverwriteAndRemove();
    if (!status.ok()) {
        return status;
    }

    SPDLOG_DEBUG("Downloading to path: {}", this->downloadPath);
    git_repository* cloned_repo = NULL;
    // clone_opts for progress reporting set in libgit2 lib by patch
    git_clone_options clone_opts = GIT_CLONE_OPTIONS_INIT;
    // Credential check function
    clone_opts.fetch_opts.callbacks.credentials = cred_acquire_cb;
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

    SPDLOG_DEBUG("Checking repository status.");
    status = CheckRepositoryStatus();
    if (!status.ok()) {
        return status;
    }

    // libgit2 clone sets readonly attributes
    status = RemoveReadonlyFileAttributeFromDir(this->downloadPath);
    if (!status.ok()) {
        return status;
    }
    return StatusCode::OK;
}

}  // namespace ovms
