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
#include "libgt2.hpp"

#include <string>

#include <assert.h>
#include <fcntl.h>
#include <git2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef PRIuZ
/* Define the printf format specifier to use for size_t output */
#if defined(_MSC_VER) || defined(__MINGW32__)
#define PRIuZ "Iu"
#else
#define PRIuZ "zu"
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif

namespace ovms {

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
    (void)payload; /* unused */

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

int cred_acquire_cb(git_credential** out,
    const char* url,
    const char* username_from_url,
    unsigned int allowed_types,
    void* payload) {
    //TODO:: implement in ovms
    return 0;
}

bool HfDownloader::CheckIfProxySet() {
    const char* env_cred = std::getenv("https_proxy");
    if (!env_cred)
        return false;
    return true;
}

int HfDownloader::cloneRepository(std::string& repo_url, std::string& repo_path) {
    int res = git_libgit2_init();
    if (res < 0) {
        const git_error* err = git_error_last();
        const char* msg = err ? err->message : "unknown failure";
        fprintf(stderr, "failed to init libgit2: %s\n", msg);
        return res;
    }
    progress_data pd = {{0}};
    git_repository* cloned_repo = NULL;
    git_clone_options clone_opts = GIT_CLONE_OPTIONS_INIT;
    git_checkout_options checkout_opts = GIT_CHECKOUT_OPTIONS_INIT;
    const char* url = repo_url.c_str();
    const char* path = repo_path.c_str();
    int error;
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
        clone_opts.fetch_opts.proxy_opts.url = std::getenv("https_proxy");
    }

    /* Do the clone */
    error = git_clone(&cloned_repo, url, path, &clone_opts);
    printf("\n");
    if (error != 0) {
        const git_error* err = git_error_last();
        if (err)
            printf("ERROR %d: %s\n", err->klass, err->message);
        else
            printf("ERROR %d: no detailed info\n", error);
    } else if (cloned_repo)
        git_repository_free(cloned_repo);

    //TODO: Create guard on init and shutdown
    git_libgit2_shutdown();
    return error;
}

}  // namespace ovms
#ifdef __cplusplus
}
#endif