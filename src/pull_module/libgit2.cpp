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
#include <array>
#include <atomic>
#include <cctype>
#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <sstream>
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
#include "src/filesystem/filesystem.hpp"
#include "src/filesystem/localfilesystem.hpp"
#include "../logging.hpp"
#include "../shutdown_state.hpp"
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

/* Exported cancellation accessors in libgit2 patch – used to abort
 * ongoing LFS downloads from OVMS. */
extern "C" {
void git_lfs_cancel_set(int value);
int git_lfs_cancel_get(void);
}

namespace ovms {
namespace fs = std::filesystem;

namespace {
std::atomic<int> g_activeLibgit2Guards{0};

/**
 * Builds the side-marker path used to track interrupted LFS operations.
 * The marker lives next to the repository directory as <repo>.lfswip.
 */
fs::path getLfsWipMarkerPath(const std::string& repositoryPath) {
    const fs::path repository = fs::path(repositoryPath);
    return repository.parent_path() / (repository.filename().string() + ".lfswip");
}

/**
 * Creates or truncates the .lfswip marker file for the target repository.
 * Marker presence indicates clone/download interruption handling is in progress.
 */
bool createLfsWipMarker(const std::string& repositoryPath) {
    const fs::path markerPath = getLfsWipMarkerPath(repositoryPath);
    std::error_code ec;
    if (!markerPath.parent_path().empty()) {
        fs::create_directories(markerPath.parent_path(), ec);
    }
    std::ofstream marker(markerPath, std::ios::trunc);
    return static_cast<bool>(marker);
}

bool hasLfsWipMarker(const std::string& repositoryPath) {
    const fs::path markerPath = getLfsWipMarkerPath(repositoryPath);
    std::error_code ec;
    return fs::exists(markerPath, ec) && fs::is_regular_file(markerPath, ec);
}

void removeLfsWipMarker(const std::string& repositoryPath) {
    const fs::path markerPath = getLfsWipMarkerPath(repositoryPath);
    std::error_code ec;
    if (!fs::remove(markerPath, ec) && ec) {
        SPDLOG_WARN("Failed to remove .lfswip marker {}: {}", markerPath.string(), ec.message());
    }
}

/**
 * Sets the LFS (Large File Storage) cancellation flag.
 * This signal is checked by the patched libgit2 library to abort ongoing LFS downloads.
 */
static void setLfsCancelRequested(int value) {
    git_lfs_cancel_set(value != 0 ? 1 : 0);
}

#define RETURN_IF_OVMS_CLONE_CANCELLED()                         \
    do {                                                         \
        if (libgit2::isCloneCancellationRequestedFromServer()) { \
            setLfsCancelRequested(1);                            \
            return -1;                                           \
        }                                                        \
    } while (0)

int checkCloneCancellationTransferProgressCallback(const git_indexer_progress* stats, void* payload) {
    (void)stats;
    (void)payload;
    RETURN_IF_OVMS_CLONE_CANCELLED();
    return 0;
}

int checkCloneCancellationSidebandProgressCallback(const char* str, int len, void* payload) {
    (void)str;
    (void)len;
    (void)payload;
    RETURN_IF_OVMS_CLONE_CANCELLED();
    return 0;
}

int checkCloneCancellationUpdateTipsCallback(const char* refname, const git_oid* a, const git_oid* b, void* payload) {
    (void)refname;
    (void)a;
    (void)b;
    (void)payload;
    RETURN_IF_OVMS_CLONE_CANCELLED();
    return 0;
}

/**
 * Callback fired during git checkout to notify about files being modified in the working tree.
 * Raised when files are created, updated, or deleted during clone or checkout operations.
 * 
 * @param why Notification type (GIT_CHECKOUT_NOTIFY_*) indicating what changed.
 * @param path Path of the file being checked out in the working directory.
 * @param baseline Diff file info at HEAD (unused).
 * @param target Diff file info being checked out (unused).
 * @param workdir Current working directory file state (unused).
 * @param payload User-supplied payload pointer (unused).
 * @return -1 if cancellation requested (aborts checkout), 0 otherwise.
 * @note Works on the git repository working directory; modifies local filesystem.
 */
int checkCloneCancellationCheckoutNotifyCallback(git_checkout_notify_t why,
    const char* path,
    const git_diff_file* baseline,
    const git_diff_file* target,
    const git_diff_file* workdir,
    void* payload) {
    (void)why;
    (void)path;
    (void)baseline;
    (void)target;
    (void)workdir;
    (void)payload;
    RETURN_IF_OVMS_CLONE_CANCELLED();
    return 0;
}
}  // namespace

/**
 * Callback for acquiring authentication credentials during git clone operations.
 * Attempts to use HF_TOKEN environment variable for Hugging Face authentication.
 * 
 * @param out [out] Pointer to git_credential structure to fill with credentials.
 * @param url The repository URL requiring authentication.
 * @param username_from_url Username parsed from URL (if any).
 * @param allowed_types Bitmask of credential types supported by the remote.
 * @param payload User-supplied payload pointer (unused).
 * @return 0 on success, -1 on error, 1 if credential type not supported.
 * @note Connects to internet for authentication; reads HF_TOKEN environment variable.
 * @note Limited LFS support: LFS downloads may fail if HF_TOKEN is not set.
 */
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

/**
 * Initializes the libgit2 library with server connection settings.
 * Sets timeouts for server connections and LFS operations, and configures SSL certificate locations.
 * Maintains a reference count of active guards to ensure single initialization.
 * 
 * @param opts Configuration options including server timeouts and SSL certificate path.
 * @note Thread-safe initialization; uses atomic counter to track active guards.
 * @note Connects to internet: configures timeouts for remote git/LFS operations.
 */
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

    /**
     * Opens a git repository at the specified filesystem path.
     * 
     * @param path Absolute or relative path to the git repository directory.
     * @note Works on specific git repository location (searches for .git directory).
     */
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
            if (repo) {
                git_repository_free(repo);
                repo = nullptr;
            }
        }
    }

    /**
     * Destructor: safely releases the git repository resource.
     */
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

/**
 * Verifies the cleanliness of a git repository after clone or download operations.
 * Checks for staged changes, unstaged modifications, untracked files, and merge conflicts.
 * 
 * @param checkUntracked If true, also check for untracked files (more expensive on large repos).
 * @return StatusCode::OK if repository is clean, StatusCode::HF_GIT_STATUS_UNCLEAN if changes exist,
 *         or error status if repository check fails.
 * @note Works on specific git repository (at downloadPath);scans the entire working directory.
 * @note Part of the git domain: analyzes HEAD state, index, and working directory.
 */
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

namespace libgit2 {

bool isCloneCancellationRequestedFromServer() {
    if (isSignalShutdownRequested()) {
        processSignalShutdownRequest();
        setLfsCancelRequested(getShutdownRequestValue());
    }
    return getShutdownRequestValue() != 0;
}

bool hasLfsErrorFile(const std::string& repositoryRootPath) {
    const fs::path errorFilePath = fs::path(repositoryRootPath) / "lfs_error.txt";
    std::error_code ec;
    return fs::exists(errorFilePath, ec) && fs::is_regular_file(errorFilePath, ec);
}

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

bool readFirstThreeLines(const std::filesystem::path& p, std::vector<std::string>& out, size_t maxLineBytes) {
    out.clear();
    if (maxLineBytes == 0)
        return false;

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
            if (line.size() > maxLineBytes) {
                out.clear();
                return false;
            }
        }
    }

    // Handle the last line if file did not end with EOL
    if (!line.empty() && out.size() < 3) {
        rtrimCrLfWhitespace(line);
        out.push_back(std::move(line));
    }

    return true;
}

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

bool hasLfsKeywordsFirst3Positional(std::string_view content) {
    std::array<std::string, 3> lines;
    size_t lineIndex = 0;
    size_t position = 0;

    while (lineIndex < lines.size() && position < content.size()) {
        size_t next = content.find_first_of("\r\n", position);
        auto line = content.substr(position, next == std::string_view::npos ? content.size() - position : next - position);
        lines[lineIndex] = std::string(line);
        rtrimCrLfWhitespace(lines[lineIndex]);
        ++lineIndex;

        if (next == std::string_view::npos)
            break;
        position = next + 1;
        if (next + 1 < content.size() && content[next] == '\r' && content[next + 1] == '\n')
            ++position;
    }

    if (lineIndex < lines.size())
        return false;

    return containsCaseInsensitive(lines[0], "version") &&
           containsCaseInsensitive(lines[1], "oid") &&
           containsCaseInsensitive(lines[2], "size");
}

bool blobHasLfsKeywordsFirst3Positional(git_blob* blob) {
    if (!blob)
        return false;
    const void* raw = git_blob_rawcontent(blob);
    if (!raw)
        return false;
    size_t rawSize = git_blob_rawsize(blob);
    return hasLfsKeywordsFirst3Positional(std::string_view(static_cast<const char*>(raw), rawSize));
}

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

/**
 * Recursively/non-recursively searches a directory for files matching the LFS pointer format.
 * Skips .git directories to avoid searching the repository metadata.
 * 
 * @param directory Root directory to search for LFS-like files.
 * @param recursive If true, search subdirectories; if false, only search directory root.
 * @return Vector of relative paths (relative to directory) of files matching LFS format.
 * @note Works on local filesystem; scans directory tree (may be expensive on large repos).
 * @note Part of the git LFS domain: identifies LFS pointer files in the working directory.
 */
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

struct MissingLfsPathsWalkPayload {
    git_repository* repo;
    fs::path worktreeRoot;
    std::set<fs::path>* matches;
};

struct MissingNonLfsPathsWalkPayload {
    git_repository* repo;
    fs::path worktreeRoot;
    std::set<fs::path>* matches;
};

/**
 * Tree-walk callback that collects paths from HEAD tree that have LFS pointers but missing worktree files.
 * Used to identify LFS files that need to be downloaded/resumed during clone recovery.
 * 
 * @param root Current directory path in the tree traversal (accumulates parent directories).
 * @param entry Current git_tree_entry being visited.
 * @param payloadPtr Pointer to MissingLfsPathsWalkPayload containing repo, worktree path, and output set.
 * @return 0 to continue tree walk; non-zero aborts the walk.
 * @note Works on git repository object database (ODB) and working directory.
 * @note Part of the git domain: walks the HEAD commit tree to find tracked LFS files.
 */
int collectMissingLfsPathsFromHeadTreeCb(const char* root, const git_tree_entry* entry, void* payloadPtr) {
    auto* payload = reinterpret_cast<MissingLfsPathsWalkPayload*>(payloadPtr);
    if (!payload || git_tree_entry_type(entry) != GIT_OBJECT_BLOB)
        return 0;

    fs::path relativePath = fs::path(root ? root : "") / git_tree_entry_name(entry);
    std::error_code ec;
    if (fs::exists(payload->worktreeRoot / relativePath, ec))
        return 0;

    git_blob* blob = nullptr;
    if (git_blob_lookup(&blob, payload->repo, git_tree_entry_id(entry)) != 0)
        return 0;
    std::unique_ptr<git_blob, decltype(&git_blob_free)> blobGuard(blob, git_blob_free);

    if (blobHasLfsKeywordsFirst3Positional(blob))
        payload->matches->insert(relativePath);

    return 0;
}

/**
 * Tree-walk callback that collects missing tracked non-LFS files from HEAD tree.
 * Paths are included only when missing in worktree and blob is not an LFS pointer.
 * 
 * @param root Current directory path in the tree traversal.
 * @param entry Current git_tree_entry being visited.
 * @param payloadPtr Pointer to MissingNonLfsPathsWalkPayload with output set.
 * @return 0 to continue walk; non-zero aborts traversal.
 * @note Works on git object database and local worktree metadata.
 */
int collectMissingNonLfsPathsFromHeadTreeCb(const char* root, const git_tree_entry* entry, void* payloadPtr) {
    auto* payload = reinterpret_cast<MissingNonLfsPathsWalkPayload*>(payloadPtr);
    if (!payload || git_tree_entry_type(entry) != GIT_OBJECT_BLOB)
        return 0;

    fs::path relativePath = fs::path(root ? root : "") / git_tree_entry_name(entry);
    std::error_code ec;
    if (fs::exists(payload->worktreeRoot / relativePath, ec))
        return 0;

    git_blob* blob = nullptr;
    if (git_blob_lookup(&blob, payload->repo, git_tree_entry_id(entry)) != 0)
        return 0;
    std::unique_ptr<git_blob, decltype(&git_blob_free)> blobGuard(blob, git_blob_free);

    if (!blobHasLfsKeywordsFirst3Positional(blob))
        payload->matches->insert(relativePath);

    return 0;
}

/**
 * Identifies LFS files that are safe to resume downloading after interruption.
 * Combines working directory LFS pointers with missing LFS blobs from the HEAD tree.
 * Useful for recovery after abrupt clone termination.
 * 
 * @param repo Pointer to the git repository object.
 * @param directory Root directory of the git working tree to search.
 * @param parseHeadTreeIfInterrupted If true, also scan HEAD tree for missing tracked LFS blobs.
 * @return Vector of relative paths to all resumable LFS files needing download completion.
 * @note Works on specific git repository; scans both working directory and object database.
 * @note Part of the git LFS domain: identifies candidates for partial-clone recovery.
 */
std::vector<fs::path> findResumableLfsFiles(git_repository* repo, const std::string& directory, bool parseHeadTreeIfInterrupted) {
    std::set<fs::path> uniqueMatches;
    auto existingMatches = findLfsLikeFiles(directory, true);
    uniqueMatches.insert(existingMatches.begin(), existingMatches.end());

    if (parseHeadTreeIfInterrupted) {
        git_object* headTreeObject = nullptr;
        if (git_revparse_single(&headTreeObject, repo, "HEAD^{tree}") == 0) {
            std::unique_ptr<git_object, decltype(&git_object_free)> headTreeGuard(headTreeObject, git_object_free);
            MissingLfsPathsWalkPayload payload{repo, fs::path(directory), &uniqueMatches};
            (void)git_tree_walk(reinterpret_cast<git_tree*>(headTreeObject), GIT_TREEWALK_PRE, collectMissingLfsPathsFromHeadTreeCb, &payload);
        }
    }

    return std::vector<fs::path>(uniqueMatches.begin(), uniqueMatches.end());
}

/**
 * Finds tracked non-LFS files present in HEAD but missing in worktree.
 * Used during resume flow to restore required metadata/config artifacts.
 * 
 * @param repo Pointer to the git repository object.
 * @param directory Repository worktree root directory.
 * @return Vector of relative paths for missing tracked non-LFS files.
 * @note Works on specific git repository; scans HEAD tree and filesystem.
 */
std::vector<fs::path> findMissingTrackedNonLfsFiles(git_repository* repo, const std::string& directory) {
    std::set<fs::path> uniqueMatches;

    git_object* headTreeObject = nullptr;
    if (git_revparse_single(&headTreeObject, repo, "HEAD^{tree}") == 0) {
        std::unique_ptr<git_object, decltype(&git_object_free)> headTreeGuard(headTreeObject, git_object_free);
        MissingNonLfsPathsWalkPayload payload{repo, fs::path(directory), &uniqueMatches};
        (void)git_tree_walk(reinterpret_cast<git_tree*>(headTreeObject), GIT_TREEWALK_PRE, collectMissingNonLfsPathsFromHeadTreeCb, &payload);
    }

    return std::vector<fs::path>(uniqueMatches.begin(), uniqueMatches.end());
}

/**
 * Checks for LFS download errors recorded by libgit2 patch in the repository root.
 * Reads and logs error messages from lfs_error.txt, then removes the file for cleanup.
 * 
 * @param repositoryRootPath Absolute path to the git repository root directory.
 * @return true if error file exists (even if unreadable); false if no error file found.
 * @note Works on specific git repository location; reads and deletes lfs_error.txt.
 * @note Part of the git LFS domain: detects LFS download failures recorded by patched libgit2.
 */
bool ifHasLfsErrorFileLogContentAndRemove(const std::string& repositoryRootPath) {
    const fs::path errorFilePath = fs::path(repositoryRootPath) / "lfs_error.txt";
    std::error_code ec;
    if (!fs::exists(errorFilePath, ec) || !fs::is_regular_file(errorFilePath, ec)) {
        return false;
    }

    std::ifstream errorFile(errorFilePath, std::ios::binary);
    if (!errorFile) {
        SPDLOG_ERROR("Detected lfs_error.txt but failed to open file: {}", errorFilePath.string());
        // Still attempt to remove the stale error file for clean state
        fs::remove(errorFilePath, ec);
        return true;
    }

    std::ostringstream content;
    content << errorFile.rdbuf();
    errorFile.close();
    SPDLOG_ERROR("{}", content.str());

    // Remove the error file to ensure clean state for subsequent download/resume attempts
    std::error_code removeEc;
    if (fs::remove(errorFilePath, removeEc)) {
        SPDLOG_DEBUG("Removed lfs_error.txt from repository root");
    } else if (removeEc) {
        SPDLOG_WARN("Failed to remove lfs_error.txt: {}", removeEc.message());
    }

    return true;
}
}  // namespace libgit2

/**
 * Restores missing tracked non-LFS files after interrupted clone/resume.
 * Rebuilds index entries from HEAD and checks out selected paths into worktree.
 * 
 * @param repo Pointer to the git repository object.
 * @param missingPaths Relative paths of tracked non-LFS files missing in worktree.
 * @return StatusCode::OK on success, cancellation/error status otherwise.
 * @note Works on repository index and working directory; does not download data from remote.
 */
Status restoreMissingTrackedNonLfsFiles(git_repository* repo, const std::vector<fs::path>& missingPaths) {
    if (missingPaths.empty())
        return StatusCode::OK;

    auto runGitCall = [&](int rc, const char* callName) -> Status {
        if (rc >= 0) {
            return StatusCode::OK;
        }
        if (libgit2::isCloneCancellationRequestedFromServer() || rc == GIT_EUSER) {
            setLfsCancelRequested(1);
            SPDLOG_ERROR("Non-LFS restore cancelled in {}.", callName);
            return StatusCode::HF_GIT_CLONE_CANCELLED;
        }
        const git_error* e = git_error_last();
        SPDLOG_ERROR("Non-LFS restore failed in {} rc:{} msg:{}", callName, rc, e && e->message ? e->message : "no message");
        return StatusCode::HF_GIT_STATUS_FAILED;
    };

    // Repair index entries from HEAD for missing tracked non-LFS files before checkout.
    // Interrupted clone may leave these paths absent in index, which later surfaces as staged changes.
    {
        git_object* headObj = nullptr;
        if (auto s = runGitCall(git_revparse_single(&headObj, repo, "HEAD^{tree}"), "git_revparse_single"); !s.ok())
            return s;
        auto headGuard = std::unique_ptr<git_object, decltype(&git_object_free)>{headObj, git_object_free};
        git_tree* headTree = reinterpret_cast<git_tree*>(headObj);

        git_index* index = nullptr;
        if (auto s = runGitCall(git_repository_index(&index, repo), "git_repository_index"); !s.ok())
            return s;
        auto indexGuard = std::unique_ptr<git_index, decltype(&git_index_free)>{index, git_index_free};

        std::vector<std::string> repairedPaths;
        repairedPaths.reserve(missingPaths.size());
        for (const auto& p : missingPaths) {
            std::string path = p.generic_string();
            git_tree_entry* treeEntry = nullptr;
            if (auto s = runGitCall(git_tree_entry_bypath(&treeEntry, headTree, path.c_str()), "git_tree_entry_bypath"); !s.ok())
                return s;
            auto treeEntryGuard = std::unique_ptr<git_tree_entry, decltype(&git_tree_entry_free)>{treeEntry, git_tree_entry_free};

            git_index_entry idxEntry = {};
            idxEntry.id = *git_tree_entry_id(treeEntry);
            idxEntry.mode = git_tree_entry_filemode_raw(treeEntry);
            idxEntry.path = path.c_str();
            if (auto s = runGitCall(git_index_add(index, &idxEntry), "git_index_add"); !s.ok())
                return s;
            repairedPaths.emplace_back(std::move(path));
        }

        if (auto s = runGitCall(git_index_write(index), "git_index_write"); !s.ok())
            return s;
    }

    std::vector<std::string> ownedPaths;
    ownedPaths.reserve(missingPaths.size());
    for (const auto& p : missingPaths) {
        ownedPaths.emplace_back(p.generic_string());
    }

    std::vector<char*> checkoutPaths;
    checkoutPaths.reserve(ownedPaths.size());
    for (auto& p : ownedPaths) {
        checkoutPaths.push_back(const_cast<char*>(p.c_str()));
    }

    git_strarray pathspec = {checkoutPaths.data(), checkoutPaths.size()};
    git_checkout_options checkoutOptions = GIT_CHECKOUT_OPTIONS_INIT;
    checkoutOptions.checkout_strategy = GIT_CHECKOUT_FORCE | GIT_CHECKOUT_RECREATE_MISSING;
    checkoutOptions.notify_flags = GIT_CHECKOUT_NOTIFY_ALL;
    checkoutOptions.notify_cb = checkCloneCancellationCheckoutNotifyCallback;
    checkoutOptions.paths = pathspec;

    int rc = git_checkout_head(repo, &checkoutOptions);
    if (rc >= 0)
        return StatusCode::OK;

    if (libgit2::isCloneCancellationRequestedFromServer() || rc == GIT_EUSER) {
        setLfsCancelRequested(1);
        SPDLOG_ERROR("Non-LFS restore cancelled by shutdown request.");
        return StatusCode::HF_GIT_CLONE_CANCELLED;
    }

    const git_error* e = git_error_last();
    SPDLOG_ERROR("Non-LFS restore failed rc:{} msg:{}", rc, e && e->message ? e->message : "no message");
    return StatusCode::HF_GIT_STATUS_FAILED;
}

/**
 * Resumes the download of a single LFS file after an interrupted clone.
 * Repairs the index entry and forces git checkout to re-trigger the LFS smudge filter.
 * Removes the stale worktree file to ensure libgit2 re-downloads it.
 * 
 * @param repo Pointer to the git repository object.
 * @param filePathInRepo Path to the LFS file relative to repository root (e.g., "model.bin").
 * @return StatusCode::OK on successful resume, StatusCode::HF_GIT_CLONE_CANCELLED if shutdown
 *         requested, StatusCode::HF_GIT_LIBGIT2_LFS_DOWNLOAD_FAILED on download errors.
 * @note Connects to internet: downloads actual LFS file content from remote.
 * @note Works on specific git repository; modifies index, working directory, and accesses ODB.
 * @note Part of the git LFS domain: repairs and completes interrupted LFS transfers.
 */
Status resumeLfsDownloadForFile(git_repository* repo, const char* filePathInRepo) {
    setLfsCancelRequested(0);
    if (libgit2::isCloneCancellationRequestedFromServer()) {
        setLfsCancelRequested(1);
        return StatusCode::HF_GIT_CLONE_CANCELLED;
    }

    auto runGitCall = [&](int rc, const char* callName) -> Status {
        if (rc >= 0) {
            return StatusCode::OK;
        }
        if (libgit2::isCloneCancellationRequestedFromServer() || rc == GIT_EUSER) {
            setLfsCancelRequested(1);
            SPDLOG_ERROR("LFS resume cancelled in {} for path: {}", callName, filePathInRepo);
            return StatusCode::HF_GIT_CLONE_CANCELLED;
        }
        const git_error* e = git_error_last();
        SPDLOG_ERROR("LFS resume failed in {} for path: {} rc:{} msg:{}", callName, filePathInRepo, rc, e && e->message ? e->message : "no message");
        return StatusCode::HF_GIT_LIBGIT2_LFS_DOWNLOAD_FAILED;
    };

    // Repair the index entry for this path.
    // After a Ctrl+C mid-clone the index can be partially written and the
    // entry for the aborted file may be missing entirely.
    // git_checkout_head silently skips any path that has no index entry, so
    // the LFS smudge filter is never reached.
    // We re-add the correct entry (pointer blob SHA from the HEAD tree) so
    // that the subsequent git_checkout_head always finds it and triggers the
    // LFS download.
    {
        git_object* headObj = nullptr;
        if (auto s = runGitCall(git_revparse_single(&headObj, repo, "HEAD^{tree}"), "git_revparse_single"); !s.ok())
            return s;
        auto headGuard = std::unique_ptr<git_object, decltype(&git_object_free)>{headObj, git_object_free};
        git_tree* headTree = reinterpret_cast<git_tree*>(headObj);

        git_tree_entry* treeEntry = nullptr;
        if (auto s = runGitCall(git_tree_entry_bypath(&treeEntry, headTree, filePathInRepo), "git_tree_entry_bypath"); !s.ok())
            return s;
        auto treeEntryGuard = std::unique_ptr<git_tree_entry, decltype(&git_tree_entry_free)>{treeEntry, git_tree_entry_free};

        git_index* index = nullptr;
        if (auto s = runGitCall(git_repository_index(&index, repo), "git_repository_index"); !s.ok())
            return s;
        auto indexGuard = std::unique_ptr<git_index, decltype(&git_index_free)>{index, git_index_free};

        git_index_entry idxEntry = {};
        idxEntry.id = *git_tree_entry_id(treeEntry);
        idxEntry.mode = git_tree_entry_filemode_raw(treeEntry);
        idxEntry.path = filePathInRepo;
        if (auto s = runGitCall(git_index_add(index, &idxEntry), "git_index_add"); !s.ok())
            return s;
        if (auto s = runGitCall(git_index_write(index), "git_index_write"); !s.ok())
            return s;
    }

    // Remove the worktree file before calling git_checkout_head.
    // After an aborted clone the worktree contains the LFS pointer text and
    // the index blob is also that pointer text, so git_checkout_head sees
    // worktree == ODB blob and skips the entry entirely (no smudge filter
    // is invoked).  Deleting the file first forces libgit2 to recreate it
    // through the normal checkout/smudge pipeline, which triggers the LFS
    // filter and downloads the actual binary.
    const char* workdir = git_repository_workdir(repo);
    if (workdir) {
        std::error_code ec;
        fs::remove(fs::path(workdir) / filePathInRepo, ec);
        // Deliberately ignore the error: if the file is already absent the
        // checkout will still recreate it via GIT_CHECKOUT_RECREATE_MISSING.
    }

    std::array<char*, 1> checkoutPaths = {const_cast<char*>(filePathInRepo)};
    git_strarray pathspec = {checkoutPaths.data(), checkoutPaths.size()};
    git_checkout_options checkoutOptions = GIT_CHECKOUT_OPTIONS_INIT;
    checkoutOptions.checkout_strategy = GIT_CHECKOUT_FORCE | GIT_CHECKOUT_RECREATE_MISSING;
    checkoutOptions.notify_flags = GIT_CHECKOUT_NOTIFY_ALL;
    checkoutOptions.notify_cb = checkCloneCancellationCheckoutNotifyCallback;
    checkoutOptions.paths = pathspec;

    auto status = runGitCall(git_checkout_head(repo, &checkoutOptions), "git_checkout_head");
    if (!status.ok())
        return status;

    return StatusCode::OK;
}

namespace {

struct ResumeCandidates {
    bool hasWipMarker = false;
    bool hasLfsErrorFile = false;
    bool interruptionLikely = false;
    std::vector<fs::path> lfsMatches;
    std::vector<fs::path> missingNonLfsMatches;
};

Status mapRepositoryOpenFailureToStatus(const GitRepositoryGuard& repoGuard) {
    if (repoGuard.git_error_class == GIT_ERROR_OS)
        return StatusCode::HF_GIT_STATUS_FAILED_TO_RESOLVE_PATH;
    if (repoGuard.git_error_class == GIT_ERROR_INVALID)
        return StatusCode::HF_GIT_LIBGIT2_NOT_INITIALIZED;
    return StatusCode::HF_GIT_STATUS_FAILED;
}

/**
 * Builds resume candidate lists based on interruption markers and repository scan.
 * 
 * @param repo Pointer to the git repository object.
 * @param downloadPath Repository worktree root path.
 * @return ResumeCandidates containing LFS and non-LFS recovery targets.
 * @note Works on local repository metadata and filesystem; no network operations.
 */
ResumeCandidates buildResumeCandidates(git_repository* repo, const std::string& downloadPath) {
    ResumeCandidates candidates;
    candidates.hasWipMarker = hasLfsWipMarker(downloadPath);
    candidates.hasLfsErrorFile = libgit2::hasLfsErrorFile(downloadPath);

    // Checking if the download was partially finished for any files in repository,
    // including tracked LFS pointer blobs missing from the worktree after abrupt termination.
    candidates.lfsMatches = libgit2::findResumableLfsFiles(repo, downloadPath, candidates.hasWipMarker || candidates.hasLfsErrorFile);
    if (candidates.hasWipMarker) {
        candidates.missingNonLfsMatches = libgit2::findMissingTrackedNonLfsFiles(repo, downloadPath);
    }

    candidates.interruptionLikely = candidates.hasWipMarker || candidates.hasLfsErrorFile || !candidates.lfsMatches.empty();
    return candidates;
}

void printResumeCandidates(const ResumeCandidates& candidates) {
    if (!candidates.lfsMatches.empty()) {
        std::cout << "Found " << candidates.lfsMatches.size() << " LFS file(s) to resume partial download:\n";
        for (const auto& p : candidates.lfsMatches) {
            std::cout << "  " << p.string() << "\n";
        }
    }

    if (!candidates.missingNonLfsMatches.empty()) {
        std::cout << "Found " << candidates.missingNonLfsMatches.size() << " missing tracked non-LFS file(s) to restore:\n";
        for (const auto& p : candidates.missingNonLfsMatches) {
            std::cout << "  " << p.string() << "\n";
        }
    }
}

/**
 * Reconciles index with HEAD tree after crash recovery operations.
 * This removes staged index artifacts before strict cleanliness validation.
 * 
 * @param repo Pointer to the git repository object.
 * @return StatusCode::OK on success, cancellation/error status otherwise.
 * @note Works on local git index; does not modify tracked file contents in worktree.
 */
Status restoreGitIndexToHead(git_repository* repo) {
    auto runGitCall = [&](int rc, const char* callName) -> Status {
        if (rc >= 0) {
            return StatusCode::OK;
        }
        if (libgit2::isCloneCancellationRequestedFromServer() || rc == GIT_EUSER) {
            setLfsCancelRequested(1);
            SPDLOG_ERROR("Index normalization cancelled in {}.", callName);
            return StatusCode::HF_GIT_CLONE_CANCELLED;
        }
        const git_error* e = git_error_last();
        SPDLOG_ERROR("Index normalization failed in {} rc:{} msg:{}", callName, rc, e && e->message ? e->message : "no message");
        return StatusCode::HF_GIT_STATUS_FAILED;
    };

    git_object* headObj = nullptr;
    if (auto s = runGitCall(git_revparse_single(&headObj, repo, "HEAD^{tree}"), "git_revparse_single"); !s.ok())
        return s;
    auto headGuard = std::unique_ptr<git_object, decltype(&git_object_free)>{headObj, git_object_free};
    git_tree* headTree = reinterpret_cast<git_tree*>(headObj);

    git_index* index = nullptr;
    if (auto s = runGitCall(git_repository_index(&index, repo), "git_repository_index"); !s.ok())
        return s;
    auto indexGuard = std::unique_ptr<git_index, decltype(&git_index_free)>{index, git_index_free};

    if (auto s = runGitCall(git_index_read_tree(index, headTree), "git_index_read_tree"); !s.ok())
        return s;
    if (auto s = runGitCall(git_index_write(index), "git_index_write"); !s.ok())
        return s;

    return StatusCode::OK;
}

Status resumeExistingRepository(git_repository* repo,
    const std::string& downloadPath,
    const ResumeCandidates& candidates,
    const std::function<Status(bool)>& checkRepositoryStatusFn) {
    if (!createLfsWipMarker(downloadPath)) {
        SPDLOG_WARN("Failed to create .lfswip marker before resume at {}", getLfsWipMarkerPath(downloadPath).string());
    }

    printResumeCandidates(candidates);

    for (const auto& p : candidates.lfsMatches) {
        if (libgit2::isCloneCancellationRequestedFromServer()) {
            setLfsCancelRequested(1);
            return StatusCode::HF_GIT_CLONE_CANCELLED;
        }
        if (!createLfsWipMarker(downloadPath)) {
            SPDLOG_WARN("Failed to refresh .lfswip marker before LFS resume at {}", getLfsWipMarkerPath(downloadPath).string());
        }
        std::cout << " Resuming " << p.string() << "...\n";
        std::string path = p.string();
        auto resumeStatus = resumeLfsDownloadForFile(repo, path.c_str());
        if (!resumeStatus.ok()) {
            return resumeStatus;
        }
    }

    if (candidates.hasWipMarker) {
        auto restoreMissingStatus = restoreMissingTrackedNonLfsFiles(repo, candidates.missingNonLfsMatches);
        if (!restoreMissingStatus.ok()) {
            return restoreMissingStatus;
        }
    } else if (!candidates.missingNonLfsMatches.empty()) {
        SPDLOG_DEBUG("Skipping non-LFS missing tracked file restoration because .lfswip marker was not found.");
    }

    auto restoredGitIndexStatus = restoreGitIndexToHead(repo);
    if (!restoredGitIndexStatus.ok()) {
        return restoredGitIndexStatus;
    }

    // Checking if git status is ok but we are left with LFS errors recorded by libgit2 patch in repository root.
    if (libgit2::ifHasLfsErrorFileLogContentAndRemove(downloadPath)) {
        SPDLOG_ERROR("Model failed after resuming download.");
        return StatusCode::HF_GIT_LIBGIT2_LFS_DOWNLOAD_FAILED;
    }

    // Blocking check
    SPDLOG_DEBUG("Checking repository status.");
    auto status = checkRepositoryStatusFn(false);
    if (!status.ok()) {
        SPDLOG_ERROR("Model repository status check failed after resuming download. Status: {}", status.string());
        SPDLOG_ERROR("Consider --override to start download from scratch.");
        return status;
    }
    SPDLOG_DEBUG("Model repository status check passed after resuming download.");

    removeLfsWipMarker(downloadPath);
    return StatusCode::OK;
}

Status handleExistingRepositoryWithoutOverwrite(const std::string& downloadPath,
    const std::function<Status(bool)>& checkRepositoryStatusFn) {
    GitRepositoryGuard repoGuard(downloadPath);
    if (!repoGuard.get()) {
        std::cout << "Path already exists on local filesystem. Cannot download model to: " << downloadPath << std::endl;
        std::cout << "Use --override to start download from scratch." << std::endl;
        return mapRepositoryOpenFailureToStatus(repoGuard);
    }

    auto candidates = buildResumeCandidates(repoGuard.get(), downloadPath);
    if (!candidates.interruptionLikely) {
        SPDLOG_DEBUG("No interruption signals found (.lfswip/lfs_error/LFS pointers/missing tracked files).");
        std::cout << "Path already exists on local filesystem. Skipping download to path: " << downloadPath << std::endl;
        return StatusCode::OK;
    }

    return resumeExistingRepository(repoGuard.get(), downloadPath, candidates, checkRepositoryStatusFn);
}

/**
 * Configures libgit2 clone options with callbacks, credentials, and optional proxy.
 * 
 * @param cloneOptions [in/out] libgit2 clone options structure to populate.
 * @param useProxy Whether to route git operations through proxyUrl.
 * @param proxyUrl Proxy URL to pass into libgit2 when useProxy is true.
 * @note Configures clone behavior only; no network call is performed by this function itself.
 */
void configureCloneOptions(git_clone_options& cloneOptions, bool useProxy, const std::string& proxyUrl) {
    // clone_opts for progress reporting set in libgit2 lib by patch
    cloneOptions.fetch_opts.callbacks.transfer_progress = checkCloneCancellationTransferProgressCallback;
    cloneOptions.fetch_opts.callbacks.sideband_progress = checkCloneCancellationSidebandProgressCallback;
    cloneOptions.fetch_opts.callbacks.update_tips = checkCloneCancellationUpdateTipsCallback;

    cloneOptions.checkout_opts.notify_flags = GIT_CHECKOUT_NOTIFY_ALL;
    cloneOptions.checkout_opts.notify_cb = checkCloneCancellationCheckoutNotifyCallback;

    // Credential check function
    cloneOptions.fetch_opts.callbacks.credentials = cred_acquire_cb;
    // Use proxy
    if (useProxy) {
        cloneOptions.fetch_opts.proxy_opts.type = GIT_PROXY_SPECIFIED;
        cloneOptions.fetch_opts.proxy_opts.url = proxyUrl.c_str();
        SPDLOG_DEBUG("Download using https_proxy settings");
    } else {
        SPDLOG_DEBUG("Download with https_proxy not set");
    }
}

/**
 * Executes git clone for a model repository and handles cancellation/error mapping.
 * 
 * @param downloadPath Destination repository path on local filesystem.
 * @param passRepoUrl Source repository URL (possibly with embedded credentials).
 * @param cloneOptions Prepared libgit2 clone options.
 * @return StatusCode::OK on success, cancellation or clone failure status otherwise.
 * @note Connects to remote git endpoint and writes repository data to local filesystem.
 */
Status executeClone(const std::string& downloadPath, const std::string& passRepoUrl, git_clone_options& cloneOptions) {
    git_repository* clonedRepo = nullptr;
    const char* url = passRepoUrl.c_str();
    const char* path = downloadPath.c_str();
    SPDLOG_TRACE("Starting git clone to: {}", path);
    if (!createLfsWipMarker(downloadPath)) {
        SPDLOG_WARN("Failed to create .lfswip marker before clone at {}", getLfsWipMarkerPath(downloadPath).string());
    }
    setLfsCancelRequested(0); /* reset for this operation */
    int error = git_clone(&clonedRepo, url, path, &cloneOptions);
    SPDLOG_TRACE("Ended git clone");
    if (error != 0) {
        if (libgit2::isCloneCancellationRequestedFromServer() || error == GIT_EUSER) {
            SPDLOG_ERROR("Libgit2 clone cancelled due to shutdown request.");
            return StatusCode::HF_GIT_CLONE_CANCELLED;
        }
        const git_error* err = git_error_last();
        if (err)
            SPDLOG_ERROR("Libgit2 clone error: {} message: {}", err->klass, err->message);
        else
            SPDLOG_ERROR("Libgit2 clone error: {}", error);
        return StatusCode::HF_GIT_CLONE_FAILED;
    }
    if (clonedRepo) {
        git_repository_free(clonedRepo);
    }
    return StatusCode::OK;
}

Status finalizeAfterClone(const std::string& downloadPath,
    const std::function<Status(bool)>& checkRepositoryStatusFn,
    const std::function<Status(const std::string&)>& removeReadonlyFn) {
    // Checking if git status is ok but we are left with LFS errors recorded by libgit2 patch in repository root.
    if (libgit2::ifHasLfsErrorFileLogContentAndRemove(downloadPath)) {
        SPDLOG_ERROR("Model download failed.");
        return StatusCode::HF_GIT_LIBGIT2_LFS_DOWNLOAD_FAILED;
    }

    SPDLOG_DEBUG("Checking repository status.");
    auto status = checkRepositoryStatusFn(true);
    if (!status.ok()) {
        SPDLOG_ERROR("Model repository status check failed after model download. Status: {}", status.string());
        SPDLOG_ERROR("Consider rerunning the command to resume the download after network issues.");
        SPDLOG_ERROR("Consider --override flag to start download from scratch.");
        return status;
    }
    SPDLOG_DEBUG("Model repository status check passed after model download.");

    // libgit2 clone sets readonly attributes
    status = removeReadonlyFn(downloadPath);
    if (!status.ok()) {
        return status;
    }

    removeLfsWipMarker(downloadPath);
    return StatusCode::OK;
}

Status handleFreshClone(const std::string& downloadPath,
    const std::string& repoUrl,
    const std::string& passRepoUrl,
    bool useProxy,
    const std::string& proxyUrl,
    const std::function<Status(bool)>& checkRepositoryStatusFn,
    const std::function<Status(const std::string&)>& removeReadonlyFn) {
    SPDLOG_DEBUG("Downloading to path: {}", downloadPath);

    git_clone_options cloneOptions = GIT_CLONE_OPTIONS_INIT;
    configureCloneOptions(cloneOptions, useProxy, proxyUrl);

    SPDLOG_DEBUG("Downloading from url: {}", repoUrl.c_str());

    auto status = executeClone(downloadPath, passRepoUrl, cloneOptions);
    if (!status.ok()) {
        return status;
    }

    return finalizeAfterClone(downloadPath, checkRepositoryStatusFn, removeReadonlyFn);
}

}  // namespace

/**
 * Main method to download a model from Hugging Face via git clone.
 * Handles initial full clone, resumption of partial downloads, and validation of repository state.
 * Applies appropriate authentication, proxy settings, and LFS configuration during clone.
 * 
 * @return StatusCode::OK on successful download or if model already exists locally,
 *         StatusCode::HF_GIT_CLONE_CANCELLED if shutdown requested during operation,
 *         StatusCode::HF_GIT_CLONE_FAILED or StatusCode::HF_GIT_LIBGIT2_LFS_DOWNLOAD_FAILED on errors.
 * @note Connects to internet: performs git clone from Hugging Face endpoint (requires network).
 * @note Works on local filesystem and downloads git repository to downloadPath.
 * @note Implements the full git domain: clone, LFS, checkout, authentication, and repository validation.
 */
Status HfDownloader::downloadModel() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    auto checkRepositoryStatusFn = [this](bool checkUntracked) {
        return this->CheckRepositoryStatus(checkUntracked);
    };
    auto removeReadonlyFn = [this](const std::string& path) {
        return this->RemoveReadonlyFileAttributeFromDir(path);
    };

    // Repository exists and we do not want to overwrite
    if (std::filesystem::is_directory(this->downloadPath) && !this->overwriteModels) {
        return handleExistingRepositoryWithoutOverwrite(this->downloadPath, checkRepositoryStatusFn);
    }

    auto status = IModelDownloader::checkIfOverwriteAndRemove();
    if (!status.ok()) {
        return status;
    }

    const bool useProxy = CheckIfProxySet();
    std::string repoUrl = GetRepoUrl();
    std::string passRepoUrl = GetRepositoryUrlWithPassword();

    return handleFreshClone(this->downloadPath, repoUrl, passRepoUrl, useProxy, this->httpProxy, checkRepositoryStatusFn, removeReadonlyFn);
}

}  // namespace ovms

/**
 * C-style function exported for libgit2 LFS extension to check if LFS downloads should be aborted.
 * Called by patched libgit2 to determine if server shutdown was requested.
 * 
 * @return 1 if LFS shutdown/cancellation is requested, 0 otherwise.
 * @note Reads in-process shutdown state only; does not perform filesystem or network I/O.
 * @note Extern C interface: for compatibility with libgit2 C library patches.
 */
extern "C" int git_lfs_shutdown_requested(void) {
    return ovms::libgit2::isCloneCancellationRequestedFromServer() ? 1 : 0;
}
