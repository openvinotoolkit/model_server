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
namespace fs = std::filesystem;

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

#define CHECK(call) do { \
    int _err = (call); \
    if (_err < 0) { \
        const git_error *e = git_error_last(); \
        fprintf(stderr, "Error %d: %s (%s:%d)\n", _err, e && e->message ? e->message : "no message", __FILE__, __LINE__); \
        return; \
    } \
} while (0)

// Fetch from remote and update FETCH_HEAD
static void do_fetch(git_repository *repo, const char *remote_name, const char *proxy)
{
    git_remote *remote = NULL;
    git_fetch_options fetch_opts = GIT_FETCH_OPTIONS_INIT;

    fetch_opts.prune = GIT_FETCH_PRUNE_UNSPECIFIED;
    fetch_opts.update_fetchhead = 1;
    fetch_opts.download_tags = GIT_REMOTE_DOWNLOAD_TAGS_ALL;
    fetch_opts.callbacks = (git_remote_callbacks) GIT_REMOTE_CALLBACKS_INIT;
    fetch_opts.callbacks.credentials = cred_acquire_cb;
    if (proxy) {
        fetch_opts.proxy_opts.type = GIT_PROXY_SPECIFIED;
        fetch_opts.proxy_opts.url = proxy;
    }

    CHECK(git_remote_lookup(&remote, repo, remote_name));

    printf("Fetching from %s...\n", remote_name);
    CHECK(git_remote_fetch(remote, NULL, &fetch_opts, NULL));

    // Optional: update remote-tracking branches' default refspec tips
    // (git_remote_fetch already updates tips if update_fetchhead=1; explicit
    // update_tips is not required in recent libgit2 versions.)

    git_remote_free(remote);
}

// Fast-forward the local branch to target OID
static void do_fast_forward(git_repository *repo,
                            git_reference *local_branch,
                            const git_oid *target_oid)
{
    git_object *target = NULL;
    git_checkout_options co_opts = GIT_CHECKOUT_OPTIONS_INIT;
    git_reference *updated_ref = NULL;

    CHECK(git_object_lookup(&target, repo, target_oid, GIT_OBJECT_COMMIT));

    // Update the branch reference to point to the target commit
    CHECK(git_reference_set_target(&updated_ref, local_branch, target_oid, "Fast-forward"));

    // Make sure HEAD points to that branch (it normally already does)
    CHECK(git_repository_set_head(repo, git_reference_name(updated_ref)));

    // Checkout files to match the target tree
    co_opts.checkout_strategy = GIT_CHECKOUT_SAFE;
    CHECK(git_checkout_tree(repo, target, &co_opts));

    printf("Fast-forwarded %s to %s\n",
           git_reference_shorthand(local_branch),
           git_oid_tostr_s(target_oid));

    git_object_free(target);
    git_reference_free(updated_ref);
}

// Perform a normal merge and create a merge commit if no conflicts
static void do_normal_merge(git_repository *repo,
                            git_reference *local_branch,
                            git_annotated_commit *their_head)
{
    git_merge_options merge_opts = GIT_MERGE_OPTIONS_INIT;
    git_checkout_options co_opts = GIT_CHECKOUT_OPTIONS_INIT;

    merge_opts.file_favor = GIT_MERGE_FILE_FAVOR_NORMAL;
    co_opts.checkout_strategy = GIT_CHECKOUT_SAFE | GIT_CHECKOUT_RECREATE_MISSING;

    const git_annotated_commit *their_heads[1] = { their_head };

    printf("Merging...\n");
    CHECK(git_merge(repo, their_heads, 1, &merge_opts, &co_opts));

    // Check for conflicts
    git_index *index = NULL;
    CHECK(git_repository_index(&index, repo));

    if (git_index_has_conflicts(index)) {
        printf("Merge has conflicts. Please resolve them and create the merge commit manually.\n");
        git_index_free(index);
        return; // Leave repository in merging state
    }

    // Write index to tree
    git_oid tree_oid;
    CHECK(git_index_write_tree(&tree_oid, index));
    CHECK(git_index_write(index));
    git_index_free(index);

    git_tree *tree = NULL;
    CHECK(git_tree_lookup(&tree, repo, &tree_oid));

    // Prepare signature (from config if available)
    git_signature *sig = NULL;
    int err = git_signature_default(&sig, repo);
    if (err == GIT_ENOTFOUND || sig == NULL) {
        // Fallback if user.name/email not set in config
        CHECK(git_signature_now(&sig, "Your Name", "you@example.com"));
    } else {
        CHECK(err);
    }

    // Get current HEAD (our) commit and their commit to be parents
    git_reference *head = NULL, *resolved_branch = NULL;
    CHECK(git_repository_head(&head, repo));
    CHECK(git_reference_resolve(&resolved_branch, head)); // ensure direct ref

    const git_oid *our_oid = git_reference_target(resolved_branch);
    git_commit *our_commit = NULL;
    git_commit *their_commit = NULL;
    CHECK(git_commit_lookup(&our_commit, repo, our_oid));
    CHECK(git_commit_lookup(&their_commit, repo, git_annotated_commit_id(their_head)));

    const git_commit *parents[2] = { our_commit, their_commit };
    git_oid merge_commit_oid;

    // Create merge commit on the current branch ref
    CHECK(git_commit_create(&merge_commit_oid,
                            repo,
                            git_reference_name(resolved_branch),
                            sig, sig,
                            NULL /* message_encoding */,
                            "Merge remote-tracking branch",
                            tree,
                            2, parents));

    printf("Created merge commit %s on %s\n",
           git_oid_tostr_s(&merge_commit_oid),
           git_reference_shorthand(resolved_branch));

    // Cleanup
    git_signature_free(sig);
    git_tree_free(tree);
    git_commit_free(our_commit);
    git_commit_free(their_commit);
    git_reference_free(head);
    git_reference_free(resolved_branch);
}

// Main pull routine: fetch + merge (fast-forward if possible)
static void pull(git_repository *repo, const char *remote_name, const char *proxy)
{
    // Ensure we are on a branch (not detached HEAD)
    git_reference *head = NULL;
    int head_res = git_repository_head(&head, repo);
    // HEAD state info
    bool is_detached = git_repository_head_detached(repo) == 1;
    bool is_unborn   = git_repository_head_unborn(repo) == 1;
    if (is_unborn) {
        fprintf(stderr, "Repository has no HEAD yet (unborn branch).\n");
        return;
    } else if (is_detached) {
        fprintf(stderr, "HEAD is detached; cannot pull safely. Checkout a branch first.\n");
        return;
    }
    CHECK(head_res);

    // Resolve symbolic HEAD to direct branch ref (refs/heads/…)
    git_reference *local_branch = NULL;
    CHECK(git_reference_resolve(&local_branch, head));

    // Find the upstream tracking branch (refs/remotes/<remote>/<branch>)
    git_reference *upstream = NULL;
    int up_ok = git_branch_upstream(&upstream, local_branch);
    if (up_ok != 0 || upstream == NULL) {
        fprintf(stderr, "Current branch has no upstream. Set it with:\n"
                        "  git branch --set-upstream-to=%s/<branch> <branch>\n", remote_name);
        return;
    }

    // Verify upstream belongs to the requested remote; not strictly required for fetch
    // but we fetch from the chosen remote anyway.
    do_fetch(repo, remote_name, proxy);

    // Prepare "their" commit as annotated commit from upstream
    git_annotated_commit *their_head = NULL;
    CHECK(git_annotated_commit_from_ref(&their_head, repo, upstream));

    // Merge analysis
    git_merge_analysis_t analysis;
    git_merge_preference_t preference;
    CHECK(git_merge_analysis(&analysis, &preference, repo,
                             (const git_annotated_commit **)&their_head, 1));

    if (analysis & GIT_MERGE_ANALYSIS_UP_TO_DATE) {
        printf("Already up to date.\n");
    } else if (analysis & GIT_MERGE_ANALYSIS_FASTFORWARD) {
        const git_oid *target_oid = git_annotated_commit_id(their_head);
        do_fast_forward(repo, local_branch, target_oid);
    } else if (analysis & GIT_MERGE_ANALYSIS_NORMAL) {
        do_normal_merge(repo, local_branch, their_head);
    } else {
        printf("No merge action taken (analysis=%u, preference=%u).\n",
               (unsigned)analysis, (unsigned)preference);
    }

    // Cleanup
    git_annotated_commit_free(their_head);
    git_reference_free(upstream);
    git_reference_free(local_branch);
    git_reference_free(head);
}


// Trim trailing '\r' (for CRLF files) and surrounding spaces
static inline void rtrimCrLfWhitespace(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back(); // remove trailing '\r'
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back(); // trailing ws
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; // leading ws
    if (i > 0) s.erase(0, i);
}

// Case-insensitive substring search: returns true if 'needle' is found in 'hay'
static bool containsCaseInsensitive(const std::string& hay, const std::string& needle) {
    auto toLower = [](std::string v) {
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return v;
    };
    std::string hayLower = toLower(hay);
    std::string needleLower = toLower(needle);
    return hayLower.find(needleLower) != std::string::npos;
}

// Read at most the first 3 lines of a file, with a per-line cap to avoid huge reads.
// Returns true if successful (even if <3 lines exist; vector will just be shorter).
static bool readFirstThreeLines(const fs::path& p, std::vector<std::string>& outLines) {
    outLines.clear();
    std::ifstream in(p, std::ios::in | std::ios::binary);
    if (!in) return false;

    constexpr std::streamsize kMaxPerLine = 8192;

    std::string line;
    line.reserve(static_cast<size_t>(kMaxPerLine));
    for (int i = 0; i < 3 && in.good(); ++i) {
        line.clear();
        std::streamsize count = 0;
        char ch;
        bool gotNewline = false;
        while (count < kMaxPerLine && in.get(ch)) {
            if (ch == '\n') { gotNewline = true; break; }
            line.push_back(ch);
            ++count;
        }
        // If we hit kMaxPerLine without encountering '\n', drain until newline to resync
        if (count == kMaxPerLine && !gotNewline) {
            while (in.get(ch)) {
                if (ch == '\n') break;
            }
        }

        if (!in && line.empty()) {
            // EOF with no data accumulated; if previous lines were read, that's fine.
            break;
        }
        rtrimCrLfWhitespace(line);
        outLines.push_back(line);
        if (!in) break; // Handle EOF gracefully
    }
    return true;
}

// Check if the first 3 lines contain required keywords in positional order:
// line1 -> "version", line2 -> "oid", line3 -> "size" (case-insensitive).
static bool fileHasLfsKeywordsFirst3Positional(const fs::path& p) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return false;

    std::vector<std::string> lines;
    if (!readFirstThreeLines(p, lines)) return false;

    if (lines.size() < 3) return false;

    return containsCaseInsensitive(lines[0], "version") &&
           containsCaseInsensitive(lines[1], "oid") &&
           containsCaseInsensitive(lines[2], "size");
}


// Helper: make path relative to base (best-effort, non-throwing).
static fs::path makeRelativeToBase(const fs::path& path, const fs::path& base) {
    std::error_code ec;
    // Try fs::relative first (handles canonical comparisons, may fail if on different roots)
    fs::path rel = fs::relative(path, base, ec);
    if (!ec && !rel.empty()) return rel;

    // Fallback: purely lexical relative (doesn't access filesystem)
    rel = path.lexically_relative(base);
    if (!rel.empty()) return rel;

    // Last resort: return filename only (better than absolute when nothing else works)
    if (path.has_filename()) return path.filename();
    return path;
}

// Find all files under 'directory' that satisfy the first-3-lines LFS keyword check.
std::vector<fs::path> findLfsLikeFiles(const std::string& directory, bool recursive = true) {
    std::vector<fs::path> matches;
    std::error_code ec;

    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) {
        return matches;
    }

    if (recursive) {
        for (fs::recursive_directory_iterator it(directory, ec), end; !ec && it != end; ++it) {
            const auto& p = it->path();
            if (fileHasLfsKeywordsFirst3Positional(p)) {
                matches.push_back(makeRelativeToBase(p, directory));
            }
        }
    } else {
        for (fs::directory_iterator it(directory, ec), end; !ec && it != end; ++it) {
            const auto& p = it->path();
            if (fileHasLfsKeywordsFirst3Positional(p)) {
                matches.push_back(makeRelativeToBase(p, directory));
            }
        }
    }
    return matches;
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


/*
 * checkout_one_file: Check out a single path from a treeish (commit/ref)
 * into the working directory, applying filters (smudge, EOL) just like clone.
 *
 * repo_path      : filesystem path to the existing (non-bare) repository
 * treeish        : e.g., "HEAD", "origin/main", a full commit SHA, etc.
 * path_in_repo   : repo-relative path (e.g., "src/main.c")
 *
 * Returns 0 on success; <0 (libgit2 error code) on failure.
 */
void resumeLfsDownloadForFile2(git_repository *repo, fs::path fileToResume, const std::string& repositoryPath) { 
    int error = 0;

    // Remove existing lfs pointer file from repository
    std::string fullPath = FileSystem::joinPath({repositoryPath, fileToResume.string()});
    std::filesystem::path filePath(fullPath);
    if (!std::filesystem::remove(filePath)) {
        SPDLOG_ERROR("Removing lfs file pointer error {}", fullPath);
        return;
    }

    const char *path_in_repo = fileToResume.string().c_str();
    // TODO: make sure we are on 'origin/main'.
    const char *treeish = "origin/main^{tree}";
    git_object *target = NULL;
    git_strarray paths = {0};
    git_checkout_options opts = GIT_CHECKOUT_OPTIONS_INIT;
    opts.disable_filters = 0;   // default; ensure you are NOT disabling filters

    if (git_repository_is_bare(repo)) {
        SPDLOG_ERROR("Repository is bare; cannot checkout to working directory {}", fileToResume.string());
        error = GIT_EBAREREPO;
        goto done;
    }

    if ((error = git_revparse_single(&target, repo, treeish)) < 0) {
        SPDLOG_ERROR("git_revparse_single failed {}", fileToResume.string());
        goto done;
    }

    // Restrict checkout to a single path
    paths.count = 1;
    paths.strings = (char **)&path_in_repo;

    opts.paths = paths;

    // Strategy: SAFER defaults — apply filters, write new files, update existing file
    // You can add GIT_CHECKOUT_FORCE if you want to overwrite conflicts.
    // opts.checkout_strategy = GIT_CHECKOUT_SAFE | GIT_CHECKOUT_DISABLE_PATHSPEC_MATCH;
    opts.checkout_strategy = GIT_CHECKOUT_FORCE | GIT_CHECKOUT_SAFE; // This makes sure BLOB update with filter is called
    // This actually writes the filtered content to the working directory
    error = git_checkout_tree(repo, target, &opts);
    if (error < 0) {
        SPDLOG_ERROR("git_checkout_tree failed {}", fileToResume.string());
    }

done:
    if (target) git_object_free(target);
    return;    
}

void resumeLfsDownloadForFile1(git_repository *repo, const char *file_path_in_repo) {
    git_object *obj = NULL;
    git_tree *tree = NULL;
    git_tree_entry *entry = NULL;
    git_blob *blob = NULL;
    git_buf out = GIT_BUF_INIT;
    // Configure filter behavior
    git_blob_filter_options opts = GIT_BLOB_FILTER_OPTIONS_INIT;
    // Choose direction:
    //   GIT_BLOB_FILTER_TO_WORKTREE : apply smudge (as if writing to working tree)
    //   GIT_BLOB_FILTER_TO_ODB      : apply clean  (as if writing to ODB)
    // opts.flags = GIT_FILTER_TO_WORKTREE;

    // Resolve HEAD tree
    CHECK(git_revparse_single(&obj, repo, "origin/main^{tree}") != 0);
    tree = (git_tree *)obj;

    // Find the tree entry and get the blob
    CHECK(git_tree_entry_bypath(&entry, tree, file_path_in_repo) != 0);
    CHECK(git_tree_entry_type(entry) != GIT_OBJECT_BLOB);

    CHECK(git_blob_lookup(&blob, repo, git_tree_entry_id(entry)) != 0);

    // Apply filters based on .gitattributes for this path
    CHECK(git_blob_filter(&out, blob, file_path_in_repo, &opts) != 0);

    // out.ptr now contains the filtered content
    fwrite(out.ptr, 1, out.size, stdout);

    git_buf_dispose(&out);
    if (blob) git_blob_free(blob);
    if (entry) git_tree_entry_free(entry);
    if (tree) git_tree_free(tree);
    if (obj) git_object_free(obj);
    return;
}


static int on_notify(
    git_checkout_notify_t why, const char *path,
    const git_diff_file *baseline, const git_diff_file *target, const git_diff_file *workdir,
    void *payload)
{
    (void)baseline; (void)target; (void)workdir; (void)payload;
    fprintf(stderr, "[checkout notify] why=%u path=%s\n", why, path ? path : "(null)");
    return 0; // non-zero would cancel
}

void checkout_one_from_origin_master(git_repository *repo, const char *path_rel) { 
    int err = 0;
    git_object *commitobj = NULL;
    git_commit *commit = NULL;
    git_tree *tree = NULL;
    git_tree_entry *te = NULL;
    git_diff *diff = NULL;
    git_diff_options diffopts = GIT_DIFF_OPTIONS_INIT;
    git_checkout_options opts = GIT_CHECKOUT_OPTIONS_INIT;


    printf("Path to resume '%s' \n", path_rel);
    /* 1) Resolve origin/master to a commit, then get its tree */
    if ((err = git_revparse_single(&commitobj, repo, "refs/remotes/origin/main^{commit}")) < 0) return;
    commit = (git_commit *)commitobj;
    if ((err = git_commit_tree(&tree, commit)) < 0) return;

    /* 2) Sanity-check: does the path exist in the target tree? */
    if ((err = git_tree_entry_bypath(&te, tree, path_rel)) == GIT_ENOTFOUND) {
        fprintf(stderr, "Path '%s' not found in origin/main\n", path_rel);
        err = 0; return; // nothing to do
    } else if (err < 0) {
        return;
    }
    git_tree_entry_free(te); te = NULL;

    /* 3) Diff target tree -> workdir (with index) for the one path */
    diffopts.pathspec.count = 1;
    diffopts.pathspec.strings = (char **)&path_rel;
    if ((err = git_diff_tree_to_workdir_with_index(&diff, repo, tree, &diffopts)) < 0) return;

    size_t n = git_diff_num_deltas(diff);
    fprintf(stderr, "[pre-checkout] deltas for %s: %zu\n", path_rel, n);
    git_diff_free(diff); diff = NULL;

    if (n == 0) {
        fprintf(stderr, "No changes to apply for %s (already matches target or not selected)\n", path_rel);
        /* fall through: we can still attempt checkout to let planner confirm */
    }

    /* 4) Configure checkout for a single literal path and creation allowed */
    const char *paths[] = { path_rel };
    opts.checkout_strategy = GIT_CHECKOUT_FORCE | GIT_CHECKOUT_SAFE | GIT_CHECKOUT_DISABLE_PATHSPEC_MATCH;     // or GIT_CHECKOUT_FORCE to overwrite local edits
    opts.paths.strings = (char **)paths;
    opts.paths.count   = 1;
    opts.notify_flags  = GIT_CHECKOUT_NOTIFY_ALL;
    opts.notify_cb     = on_notify;

    /* Optional: ensure baseline reflects current HEAD */
    // git_object *head = NULL; git_tree *head_tree = NULL;
    // if (git_revparse_single(&head, repo, "HEAD^{commit}") == 0) {
    //     git_commit_tree(&head_tree, (git_commit *)head);
    //     opts.baseline = head_tree;
    // }

    /* 5) Only the selected path will be considered; planner will create/update it */
    err = git_checkout_tree(repo, (git_object *)tree, &opts);

    git_tree_free(tree);
    git_commit_free(commit);
    git_object_free(commitobj);
    return;
}


Status HfDownloader::downloadModel() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    // Repository exists and we do not want to overwrite
    if (std::filesystem::is_directory(this->downloadPath) && !this->overwriteModels) {
        auto matches = findLfsLikeFiles(this->downloadPath, true);
        
        if (matches.empty()) {
            std::cout << "No files with LFS-like keywords in the first 3 lines were found.\n";
        } else {
            std::cout << "Found " << matches.size() << " matching file(s):\n";
            for (const auto& p : matches) {
                std::cout << "  " << p.string() << "\n";
            }
        }

        git_repository *repo = NULL;
        int error = git_repository_open_ext(&repo, this->downloadPath.c_str(), 0, NULL);
        if (error < 0) {
            const git_error *err = git_error_last();
            if (err)
                SPDLOG_ERROR("Repository open failed: {} {}", err->klass, err->message);
            else
                SPDLOG_ERROR("Repository open failed: {}", error);
            if (repo) git_repository_free(repo);

            std::cout << "Path already exists on local filesystem. And is not a git repository: " << this->downloadPath << std::endl;
            return StatusCode::HF_GIT_CLONE_FAILED;
        }

        // Set repository url
        std::string passRepoUrl = GetRepositoryUrlWithPassword();
        const char* url = passRepoUrl.c_str();
        error = git_repository_set_url(repo, url);
        if (error < 0) {
            const git_error *err = git_error_last();
            if (err)
                SPDLOG_ERROR("Repository set url failed: {} {}", err->klass, err->message);
            else
                SPDLOG_ERROR("Repository set url failed: {}", error);
            if (repo) git_repository_free(repo);
            std::cout << "Path already exists on local filesystem. And set git repository url failed: " << this->downloadPath << std::endl;
            return StatusCode::HF_GIT_CLONE_FAILED;
        }

        for (const auto& p : matches) {
                std::cout << " Resuming " << p.string() << "\n";
                // Remove existing lfs pointer file from repository
                std::string fullPath = FileSystem::joinPath({this->downloadPath, p.string()});
                std::filesystem::path filePath(fullPath);
                if (!std::filesystem::remove(filePath)) {
                    SPDLOG_ERROR("Removing lfs file pointer error {}", fullPath);
                    return StatusCode::HF_GIT_CLONE_FAILED;
                }
                std::string path = p.string();
                resumeLfsDownloadForFile1(repo, path.c_str());
            }
        
        // Use proxy
        if (CheckIfProxySet()) {
            SPDLOG_DEBUG("Download using https_proxy settings");
            //pull(repo, "origin", this->httpProxy.c_str());
        } else {
            SPDLOG_DEBUG("Download with https_proxy not set");
            pull(repo, "origin", nullptr);
        }

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
