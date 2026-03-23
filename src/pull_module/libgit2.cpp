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

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

namespace {
std::atomic<int> g_activeLibgit2Guards{0};
}  // namespace

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
            fprintf(stderr, "[ERROR] HF_TOKEN env variable is not set.\n");
            return -1;
        }
        error = git_credential_userpass_plaintext_new(out, username, password);
        if (error < 0) {
            fprintf(stderr, "[ERROR] Creating credentials failed.\n");
            error = -1;
        }
    } else {
        fprintf(stderr, "[ERROR] Only USERPASS_PLAINTEXT supported in OVMS.\n");
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

    this->countedAsInitialized = true;
    g_activeLibgit2Guards.fetch_add(1, std::memory_order_relaxed);
}

Libgt2InitGuard::~Libgt2InitGuard() {
    SPDLOG_DEBUG("Shutdown libgit2");
    if (this->countedAsInitialized) {
        g_activeLibgit2Guards.fetch_sub(1, std::memory_order_relaxed);
    }
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

class GitRepositoryGuard {
public:
    git_repository* repo = nullptr;
    int git_error_class = 0;

    GitRepositoryGuard(const std::string& path) {
        int error = git_repository_open_ext(&repo, path.c_str(), 0, nullptr);
        if (error < 0) {
            const git_error* err = git_error_last();
            if (err) {
                SPDLOG_ERROR("Repository open failed: {} {}", err->klass, err->message);
                git_error_class = err->klass;
            } else {
                SPDLOG_ERROR("Repository open failed: {}", error);
            }
            if (repo)
                git_repository_free(repo);
        }
    }

    ~GitRepositoryGuard() {
        if (repo) {
            git_repository_free(repo);
        }
    }

    // Allow implicit access to the raw pointer
    git_repository* get() const { return repo; }
    operator git_repository*() const { return repo; }

    // Non-copyable
    GitRepositoryGuard(const GitRepositoryGuard&) = delete;
    GitRepositoryGuard& operator=(const GitRepositoryGuard&) = delete;

    // Movable
    GitRepositoryGuard(GitRepositoryGuard&& other) noexcept {
        repo = other.repo;
        other.repo = nullptr;
    }
    GitRepositoryGuard& operator=(GitRepositoryGuard&& other) noexcept {
        if (this != &other) {
            if (repo)
                git_repository_free(repo);
            repo = other.repo;
            other.repo = nullptr;
        }
        return *this;
    }
};

Status HfDownloader::CheckRepositoryStatus(bool checkUntracked) {
    if (g_activeLibgit2Guards.load(std::memory_order_relaxed) <= 0) {
        return StatusCode::HF_GIT_LIBGIT2_NOT_INITIALIZED;
    }

    GitRepositoryGuard repoGuard(this->downloadPath);
    if (!repoGuard.get()) {
        if (repoGuard.git_error_class == GIT_ERROR_OS)
            return StatusCode::HF_GIT_STATUS_FAILED_TO_RESOLVE_PATH;
        else if (repoGuard.git_error_class == GIT_ERROR_INVALID)
            return StatusCode::HF_GIT_LIBGIT2_NOT_INITIALIZED;
        else
            return StatusCode::HF_GIT_STATUS_FAILED;
    }
    // HEAD state info
    bool is_detached = git_repository_head_detached(repoGuard.get()) == 1;
    bool is_unborn = git_repository_head_unborn(repoGuard.get()) == 1;

    // Collect status (staged/unstaged/untracked)
    git_status_options opts = GIT_STATUS_OPTIONS_INIT;

    opts.show = GIT_STATUS_SHOW_INDEX_AND_WORKDIR;
    opts.flags = GIT_STATUS_OPT_SORT_CASE_SENSITIVELY;
    if (checkUntracked) {
        // Include untracked files only when requested, as this can be expensive on large repositories.
        opts.flags |= GIT_STATUS_OPT_INCLUDE_UNTRACKED;  // | GIT_STATUS_OPT_RENAMES_HEAD_TO_INDEX    // detect renames HEAD->index - not required currently and impacts performance
    }

    git_status_list* status_list = nullptr;
    int error = git_status_list_new(&status_list, repoGuard.get(), &opts);
    if (error != 0) {
        return StatusCode::HF_GIT_STATUS_FAILED;
    }

    size_t staged = 0, unstaged = 0, untracked = 0, conflicted = 0;
    const size_t n = git_status_list_entrycount(status_list);  // iterate entries
    for (size_t i = 0; i < n; ++i) {
        const git_status_entry* e = git_status_byindex(status_list, i);
        if (!e)
            continue;
        unsigned s = e->status;

        // Staged (index) changes
        if (s & (GIT_STATUS_INDEX_NEW |
                    GIT_STATUS_INDEX_MODIFIED |
                    GIT_STATUS_INDEX_DELETED |
                    GIT_STATUS_INDEX_RENAMED |
                    GIT_STATUS_INDEX_TYPECHANGE))
            ++staged;

        // Unstaged (workdir) changes
        if (s & (GIT_STATUS_WT_MODIFIED |
                    GIT_STATUS_WT_DELETED |
                    GIT_STATUS_WT_RENAMED |
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
    ss << "Staged changes  : " << staged << "\n";
    ss << "Unstaged changes: " << unstaged << "\n";
    ss << "Untracked files : " << untracked << "\n";
    if (conflicted)
        ss << " (" << conflicted << " paths flagged)";

    SPDLOG_DEBUG(ss.str());
    git_status_list_free(status_list);

    // We do not care about untracked until after git clone
    if (is_unborn || is_detached || staged || unstaged || conflicted || (checkUntracked && untracked)) {
        return StatusCode::HF_GIT_STATUS_UNCLEAN;
    }
    return StatusCode::OK;
}

#define CHECK(call)                                                                                                             \
    do {                                                                                                                        \
        int _err = (call);                                                                                                      \
        if (_err < 0) {                                                                                                         \
            const git_error* e = git_error_last();                                                                              \
            fprintf(stderr, "[ERROR] %d: %s (%s:%d)\n", _err, e && e->message ? e->message : "no message", __FILE__, __LINE__); \
            return;                                                                                                             \
        }                                                                                                                       \
    } while (0)

namespace libgit2 {
// Trim ASCII leading/trailing whitespace in a locale-independent way.
// This keeps non-ASCII bytes (e.g. UTF-8 continuation bytes) untouched.
void rtrimCrLfWhitespace(std::string& s) {
    auto isAsciiWhitespace = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
    };

    while (!s.empty() && isAsciiWhitespace(static_cast<unsigned char>(s.back())))
        s.pop_back();  // trailing ws
    size_t i = 0;
    while (i < s.size() && isAsciiWhitespace(static_cast<unsigned char>(s[i])))
        ++i;  // leading ws
    if (i > 0)
        s.erase(0, i);
}

// Case-insensitive substring search: returns true if 'needle' is found in 'hay'
bool containsCaseInsensitive(const std::string& hay, const std::string& needle) {
    auto toLower = [](std::string v) {
        std::transform(v.begin(), v.end(), v.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return v;
    };
    std::string hayLower = toLower(hay);
    std::string needleLower = toLower(needle);
    return hayLower.find(needleLower) != std::string::npos;
}

// Read at most the first 3 lines of a file, with a per-line cap to avoid huge reads.
// Returns true if successful (even if <3 lines exist; vector will just be shorter).

bool readFirstThreeLines(const std::filesystem::path& p, std::vector<std::string>& out) {
    out.clear();

    std::ifstream in(p, std::ios::binary);
    if (!in)
        return false;

    std::string line;
    line.reserve(256);  // small optimization
    int c;

    while (out.size() < 3 && (c = in.get()) != EOF) {
        if (c == '\r') {
            // Handle CR or CRLF as one line ending
            int next = in.peek();
            if (next == '\n') {
                in.get();  // consume '\n'
            }
            // finalize current line
            rtrimCrLfWhitespace(line);
            out.push_back(std::move(line));
            line.clear();
        } else if (c == '\n') {
            // LF line ending
            rtrimCrLfWhitespace(line);
            out.push_back(std::move(line));
            line.clear();
        } else {
            line.push_back(static_cast<char>(c));
        }
    }

    // Handle the last line if file did not end with EOL
    if (!line.empty() && out.size() < 3) {
        rtrimCrLfWhitespace(line);
        out.push_back(std::move(line));
    }

    return true;
}

// Check if the first 3 lines contain required keywords in positional order:
// line1 -> "version", line2 -> "oid", line3 -> "size" (case-insensitive).
bool fileHasLfsKeywordsFirst3Positional(const fs::path& p) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec))
        return false;

    std::vector<std::string> lines;
    if (!readFirstThreeLines(p, lines))
        return false;

    if (lines.size() < 3)
        return false;

    return containsCaseInsensitive(lines[0], "version") &&
           containsCaseInsensitive(lines[1], "oid") &&
           containsCaseInsensitive(lines[2], "size");
}

// Helper: make path relative to base (best-effort, non-throwing).
fs::path makeRelativeToBase(const fs::path& path, const fs::path& base) {
    // Root-like paths (e.g. "/" or "C:\\") have no filename component.
    // Keep them unchanged instead of converting to a cwd/base-dependent ".." chain.
    if (!path.has_filename())
        return path;

    std::error_code ec;
    // Try fs::relative first (handles canonical comparisons, may fail if on different roots)
    fs::path rel = fs::relative(path, base, ec);
    if (!ec && !rel.empty())
        return rel;

    // Fallback: purely lexical relative (doesn't access filesystem)
    rel = path.lexically_relative(base);
    if (!rel.empty())
        return rel;

    // Last resort: return filename only (better than absolute when nothing else works)
    if (path.has_filename())
        return path.filename();
    return path;
}

// Find all files under 'directory' that satisfy the first-3-lines LFS keyword check. Default:  bool recursive = true
std::vector<fs::path> findLfsLikeFiles(const std::string& directory, bool recursive) {
    std::vector<fs::path> matches;
    std::error_code ec;

    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) {
        return matches;
    }

    if (recursive) {
        for (fs::recursive_directory_iterator it(directory, ec), end; !ec && it != end; ++it) {
            const auto& p = it->path();
            std::error_code dirEc;
            if (it->is_directory(dirEc) && !dirEc && p.filename() == ".git") {
                it.disable_recursion_pending();
                continue;
            }
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
}  // namespace libgit2

// pick the right entry pointer type for your libgit2
#if defined(GIT_LIBGIT2_VER_MAJOR)
// libgit2 ≥ 1.0 generally has const-correct free() (accepts const*)
using git_tree_entry_ptr = const git_tree_entry*;
#else
using git_tree_entry_ptr = git_tree_entry*;
#endif

// Single guard that owns all temporaries used in resumeLfsDownloadForFile
struct GitScope {
    git_object* tree_obj = nullptr;      // owns the tree as a generic git_object
    git_tree_entry_ptr entry = nullptr;  // owns the entry
    git_blob* blob = nullptr;            // owns the blob
    git_buf out = GIT_BUF_INIT;          // owns the buffer

    GitScope() = default;
    ~GitScope() { cleanup(); }

    GitScope(const GitScope&) = delete;
    GitScope& operator=(const GitScope&) = delete;

    GitScope(GitScope&& other) noexcept :
        tree_obj(other.tree_obj),
        entry(other.entry),
        blob(other.blob),
        out(other.out) {
        other.tree_obj = nullptr;
        other.entry = nullptr;
        other.blob = nullptr;
        other.out = GIT_BUF_INIT;
    }
    GitScope& operator=(GitScope&& other) noexcept {
        if (this != &other) {
            cleanup();
            tree_obj = other.tree_obj;
            entry = other.entry;
            blob = other.blob;
            out = other.out;
            other.tree_obj = nullptr;
            other.entry = nullptr;
            other.blob = nullptr;
            other.out = GIT_BUF_INIT;
        }
        return *this;
    }

    git_tree* tree() const { return reinterpret_cast<git_tree*>(tree_obj); }

private:
    void cleanup() noexcept {
        git_buf_dispose(&out);
        if (blob) {
            git_blob_free(blob);
            blob = nullptr;
        }
        if (entry) {
            git_tree_entry_free(entry);
            entry = nullptr;
        }
        if (tree_obj) {
            git_object_free(tree_obj);
            tree_obj = nullptr;
        }
    }
};

void resumeLfsDownloadForFile(git_repository* repo, const char* filePathInRepo) {
    GitScope g;

    // Resolve HEAD tree (HEAD^{tree})
    CHECK(git_revparse_single(&g.tree_obj, repo, "HEAD^{tree}"));

    // Find the tree entry by path
    CHECK(git_tree_entry_bypath(&g.entry, g.tree(), filePathInRepo));

    // Ensure it's a blob
    if (git_tree_entry_type(g.entry) != GIT_OBJECT_BLOB) {
        fprintf(stderr, "[ERROR] Path is not a blob: %s\n", filePathInRepo);
        return;  // Guard cleans up
    }

    // Lookup the blob
    CHECK(git_blob_lookup(&g.blob, repo, git_tree_entry_id(g.entry)));

    // Configure filter behavior
    git_blob_filter_options opts = GIT_BLOB_FILTER_OPTIONS_INIT;
    // Choose direction:
    //   GIT_FILTER_TO_WORKTREE : apply smudge (as if writing to working tree)
    //   GIT_FILTER_TO_ODB      : apply clean  (as if writing to ODB)
    opts.flags |= GIT_FILTER_TO_WORKTREE;

    // Apply filters based on .gitattributes for this path (triggers LFS smudge/clean)
    CHECK(git_blob_filter(&g.out, g.blob, filePathInRepo, &opts));

    // We don't need the buffer contents; the filter side-effects are enough.
    // All resources (out, blob, entry, tree_obj) will be freed automatically here.
}

Status HfDownloader::downloadModel() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    // Repository exists and we do not want to overwrite
    if (std::filesystem::is_directory(this->downloadPath) && !this->overwriteModels) {
        // Checking if the download was partially finished for any files in repository
        auto matches = libgit2::findLfsLikeFiles(this->downloadPath, true);

        if (matches.empty()) {
            SPDLOG_DEBUG("No files to resume download found.");
            std::cout << "Path already exists on local filesystem. Skipping download to path: " << this->downloadPath << std::endl;
            return StatusCode::OK;
        } else {
            std::cout << "Found " << matches.size() << " file(s) to resume partial download:\n";
            for (const auto& p : matches) {
                std::cout << "  " << p.string() << "\n";
            }
        }

        GitRepositoryGuard repoGuard(this->downloadPath);
        if (!repoGuard.get()) {
            std::cout << "Path already exists on local filesystem. Cannot download model to: " << this->downloadPath << std::endl;
            std::cout << "Use --override to start download from scratch." << std::endl;
            if (repoGuard.git_error_class == GIT_ERROR_OS)
                return StatusCode::HF_GIT_STATUS_FAILED_TO_RESOLVE_PATH;
            else if (repoGuard.git_error_class == GIT_ERROR_INVALID)
                return StatusCode::HF_GIT_LIBGIT2_NOT_INITIALIZED;
            else
                return StatusCode::HF_GIT_STATUS_FAILED;
        }

        // Set repository url
        std::string passRepoUrl = GetRepositoryUrlWithPassword();
        const char* url = passRepoUrl.c_str();
        int error = git_repository_set_url(repoGuard.get(), url);
        if (error < 0) {
            const git_error* err = git_error_last();
            if (err)
                SPDLOG_ERROR("Repository set url failed: {} {}", err->klass, err->message);
            else
                SPDLOG_ERROR("Repository set url failed: {}", error);
            std::cout << "Path already exists on local filesystem. And set git repository url failed: " << this->downloadPath << std::endl;
            std::cout << "Consider --override to start download from scratch." << std::endl;
            return StatusCode::HF_GIT_CLONE_FAILED;
        }

        for (const auto& p : matches) {
            std::cout << " Resuming " << p.string() << "...\n";
            std::string path = p.string();
            resumeLfsDownloadForFile(repoGuard.get(), path.c_str());
        }

        // Non blocking check
        SPDLOG_DEBUG("Checking repository status.");
        auto status =  CheckRepositoryStatus(false);
        if (!status.ok()) {
            SPDLOG_DEBUG("[WARNING] Model repository status check failed after resuming download. Status: {}", status.string());
            SPDLOG_DEBUG("Consider --override to start download from scratch.");
        } else {
            SPDLOG_DEBUG("Model repository status check passed after resuming download.");
        }
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
    status = CheckRepositoryStatus(true);
    if (!status.ok()) {
        SPDLOG_ERROR("Model repository status check failed after model download. Status: {}", status.string());
        SPDLOG_ERROR("Consider rerunning the command to resume the download after network issues.");
        SPDLOG_ERROR("Consider --override flag to start download from scratch.");
        return status;
    } else {
        SPDLOG_DEBUG("Model repository status check passed after model download.");
    }

    // libgit2 clone sets readonly attributes
    status = RemoveReadonlyFileAttributeFromDir(this->downloadPath);
    if (!status.ok()) {
        return status;
    }
    return StatusCode::OK;
}

}  // namespace ovms
