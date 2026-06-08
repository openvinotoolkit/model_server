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
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <openssl/sha.h>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <httplib.h>

#include "absl/strings/escaping.h"

#include "src/utils/env_guard.hpp"
#include "src/test/light_test_utils.hpp"
#include "src/test/test_utils.hpp"
#include "src/test/test_file_utils.hpp"
#include "src/test/test_with_temp_dir.hpp"
#include "src/filesystem/filesystem.hpp"
#include "src/pull_module/hf_pull_model_module.hpp"
#include "src/pull_module/libgit2.hpp"
#include "src/pull_module/optimum_export.hpp"
#include "src/servables_config_manager_module/listmodels.hpp"
#include "src/modelextensions.hpp"
#include "src/capi_frontend/server_settings.hpp"

#include "../module.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "src/stringutils.hpp"
#include "../timer.hpp"

#include "environment.hpp"

namespace {

constexpr std::uintmax_t OPENVINO_MODEL_BIN_FULL_SIZE_BYTES = 52417240;
constexpr std::uintmax_t OPENVINO_MODEL_BIN_HALF_SIZE_BYTES = 26208620;
constexpr std::uintmax_t OPENVINO_DETOKENIZER_BIN_FULL_SIZE_BYTES = 339125;
constexpr std::uintmax_t OPENVINO_TOKENIZER_BIN_FULL_SIZE_BYTES = 500292;
constexpr std::uintmax_t TOKENIZER_MODEL_FULL_SIZE_BYTES = 499723;
constexpr std::uintmax_t DRAFT_OPENVINO_TOKENIZER_BIN_SIZE_BYTES = 2022483;

struct ProbeErrorTracker {
    std::size_t consecutiveErrors = 0;
    std::error_code lastError;
    std::string lastOperation;
    std::string lastPath;
    std::string persistentFailureReason;
};

bool trackProbeError(ProbeErrorTracker& tracker, const std::error_code& probeError, std::string_view operation, const std::string& path,
    std::string_view context, std::size_t maxConsecutiveErrors) {
    if (!probeError || probeError == std::errc::no_such_file_or_directory) {
        tracker.consecutiveErrors = 0;
        tracker.lastError.clear();
        tracker.lastOperation.clear();
        tracker.lastPath.clear();
        return false;
    }

    if ((probeError == tracker.lastError) && (tracker.lastOperation == operation) && (tracker.lastPath == path)) {
        ++tracker.consecutiveErrors;
    } else {
        tracker.consecutiveErrors = 1;
        tracker.lastError = probeError;
        tracker.lastOperation = operation;
        tracker.lastPath = path;
    }

    if ((tracker.consecutiveErrors == 1) || (tracker.consecutiveErrors % 5 == 0)) {
        SPDLOG_WARN("{}: non-benign filesystem probe error repeated {} time(s): op={} path={} ec={} ({})",
            context, tracker.consecutiveErrors, operation, path, probeError.value(), probeError.message());
    }
    if (tracker.consecutiveErrors >= maxConsecutiveErrors) {
        tracker.persistentFailureReason = "Persistent non-benign filesystem probe error while waiting for in-progress pull: op=" + std::string(operation) +
                                          " path=" + path +
                                          " ec=" + std::to_string(probeError.value()) +
                                          " (" + probeError.message() + ") repeated " +
                                          std::to_string(tracker.consecutiveErrors) + " time(s)";
        return true;
    }
    return false;
}

::testing::AssertionResult waitForResumableInProgressPull(
    const std::string& modelBasePath,
    const std::string& modelPath,
    const std::string& downloadPath,
    std::uintmax_t expectedFullModelSize,
    std::string_view context,
    int timeoutSeconds,
    int pollIntervalMs,
    int postObservationDelayMs,
    std::size_t maxConsecutiveProbeErrors) {
    ProbeErrorTracker probeErrorTracker;
    const std::string mainRefPath = ovms::FileSystem::appendSlash(modelBasePath) + ".git/refs/heads/main";
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSeconds);
    while (std::chrono::steady_clock::now() < deadline) {
        std::error_code ec;
        const bool hasMainRef = std::filesystem::exists(mainRefPath, ec);
        if (trackProbeError(probeErrorTracker, ec, "exists", mainRefPath, context, maxConsecutiveProbeErrors)) {
            return ::testing::AssertionFailure() << probeErrorTracker.persistentFailureReason;
        }

        ec.clear();
        const bool modelExists = std::filesystem::exists(modelPath, ec);
        if (trackProbeError(probeErrorTracker, ec, "exists", modelPath, context, maxConsecutiveProbeErrors)) {
            return ::testing::AssertionFailure() << probeErrorTracker.persistentFailureReason;
        }

        std::uintmax_t modelSize = 0;
        if (modelExists) {
            ec.clear();
            modelSize = std::filesystem::file_size(modelPath, ec);
            if (ec) {
                if (trackProbeError(probeErrorTracker, ec, "file_size", modelPath, context, maxConsecutiveProbeErrors)) {
                    return ::testing::AssertionFailure() << probeErrorTracker.persistentFailureReason;
                }
                modelSize = 0;
            }
        }

        auto lfsCandidates = ovms::libgit2::findLfsLikeFiles(downloadPath, true);
        const bool hasLfsArtifacts = !lfsCandidates.empty();
        const bool modelInFlight = modelExists && (modelSize > 0) && (modelSize < expectedFullModelSize);
        if (hasMainRef && (hasLfsArtifacts || modelInFlight)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(postObservationDelayMs));
            return ::testing::AssertionSuccess();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
    }

    return ::testing::AssertionFailure() << context << ": did not observe in-progress pull within timeout.";
}

#ifdef _WIN32
struct WindowsResumeWorkerConfig {
    const char* workerFlagEnv;
    const char* modelEnv;
    const char* downloadPathEnv;
    const char* taskEnv;
    const char* gtestFilter;
    DWORD processCreationFlags;
};

const WindowsResumeWorkerConfig RESUME_TERMINATE_WORKER_CONFIG = {
    "OVMS_RESUME_TERMINATE_WORKER",
    "OVMS_RESUME_TERMINATE_MODEL",
    "OVMS_RESUME_TERMINATE_DOWNLOAD_PATH",
    "OVMS_RESUME_TERMINATE_TASK",
    "HfPullWindowsWorker.ResumeTerminateChildProcess",
    0};

const WindowsResumeWorkerConfig RESUME_CTRLC_WORKER_CONFIG = {
    "OVMS_RESUME_CTRLC_WORKER",
    "OVMS_RESUME_CTRLC_MODEL",
    "OVMS_RESUME_CTRLC_DOWNLOAD_PATH",
    "OVMS_RESUME_CTRLC_TASK",
    "HfPullWindowsWorker.ResumeCtrlCChildProcess",
    CREATE_NEW_PROCESS_GROUP};

::testing::AssertionResult setWindowsResumeWorkerEnvironment(
    const WindowsResumeWorkerConfig& config,
    const std::string& modelName,
    const std::string& downloadPath,
    const std::string& task) {
    if (!SetEnvironmentVariableA(config.workerFlagEnv, "1")) {
        return ::testing::AssertionFailure() << "Failed to set env var: " << config.workerFlagEnv;
    }
    if (!SetEnvironmentVariableA(config.modelEnv, modelName.c_str())) {
        return ::testing::AssertionFailure() << "Failed to set env var: " << config.modelEnv;
    }
    if (!SetEnvironmentVariableA(config.downloadPathEnv, downloadPath.c_str())) {
        return ::testing::AssertionFailure() << "Failed to set env var: " << config.downloadPathEnv;
    }
    if (!SetEnvironmentVariableA(config.taskEnv, task.c_str())) {
        return ::testing::AssertionFailure() << "Failed to set env var: " << config.taskEnv;
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult clearWindowsResumeWorkerEnvironment(const WindowsResumeWorkerConfig& config) {
    if (!SetEnvironmentVariableA(config.workerFlagEnv, nullptr)) {
        return ::testing::AssertionFailure() << "Failed to clear env var: " << config.workerFlagEnv;
    }
    if (!SetEnvironmentVariableA(config.modelEnv, nullptr)) {
        return ::testing::AssertionFailure() << "Failed to clear env var: " << config.modelEnv;
    }
    if (!SetEnvironmentVariableA(config.downloadPathEnv, nullptr)) {
        return ::testing::AssertionFailure() << "Failed to clear env var: " << config.downloadPathEnv;
    }
    if (!SetEnvironmentVariableA(config.taskEnv, nullptr)) {
        return ::testing::AssertionFailure() << "Failed to clear env var: " << config.taskEnv;
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult launchWindowsResumeWorkerProcess(const WindowsResumeWorkerConfig& config, PROCESS_INFORMATION& pi) {
    char testExePath[MAX_PATH] = {0};
    const DWORD exePathLen = GetModuleFileNameA(nullptr, testExePath, MAX_PATH);
    if ((exePathLen == 0u) || (exePathLen >= static_cast<DWORD>(MAX_PATH))) {
        return ::testing::AssertionFailure() << "Failed to resolve current test executable path";
    }

    std::string commandLine = std::string("\"") + testExePath + "\" --gtest_filter=" + config.gtestFilter;
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessA(
            nullptr,
            commandLine.data(),
            nullptr,
            nullptr,
            TRUE,
            config.processCreationFlags,
            nullptr,
            nullptr,
            &si,
            &pi)) {
        return ::testing::AssertionFailure() << "CreateProcessA failed for filter: " << config.gtestFilter;
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult startWindowsResumeWorker(
    const WindowsResumeWorkerConfig& config,
    const std::string& modelName,
    const std::string& downloadPath,
    const std::string& task,
    PROCESS_INFORMATION& pi) {
    auto envStatus = setWindowsResumeWorkerEnvironment(config, modelName, downloadPath, task);
    if (!envStatus) {
        return envStatus;
    }

    auto launchStatus = launchWindowsResumeWorkerProcess(config, pi);
    if (!launchStatus) {
        (void)clearWindowsResumeWorkerEnvironment(config);
        return launchStatus;
    }
    return ::testing::AssertionSuccess();
}

void closeWindowsWorkerHandles(PROCESS_INFORMATION& pi) {
    if (pi.hThread != nullptr) {
        CloseHandle(pi.hThread);
        pi.hThread = nullptr;
    }
    if (pi.hProcess != nullptr) {
        CloseHandle(pi.hProcess);
        pi.hProcess = nullptr;
    }
}

::testing::AssertionResult terminateWindowsWorker(PROCESS_INFORMATION& pi, int timeoutSeconds) {
    if (!TerminateProcess(pi.hProcess, 1)) {
        return ::testing::AssertionFailure() << "TerminateProcess failed";
    }
    if (WaitForSingleObject(pi.hProcess, timeoutSeconds * 1000) != WAIT_OBJECT_0) {
        return ::testing::AssertionFailure() << "Timed out waiting for terminated child process";
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult sendCtrlBreakToWindowsWorker(PROCESS_INFORMATION& pi, int timeoutSeconds) {
    if (!GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId)) {
        return ::testing::AssertionFailure() << "GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT) failed";
    }
    if (WaitForSingleObject(pi.hProcess, timeoutSeconds * 1000) != WAIT_OBJECT_0) {
        return ::testing::AssertionFailure() << "Timed out waiting for CTRL_BREAK child process shutdown";
    }
    DWORD childExitCode = 0;
    if (!GetExitCodeProcess(pi.hProcess, &childExitCode)) {
        return ::testing::AssertionFailure() << "GetExitCodeProcess failed";
    }
    if (childExitCode == static_cast<DWORD>(EXIT_SUCCESS)) {
        return ::testing::AssertionFailure()
               << "Child process exited with EXIT_SUCCESS after CTRL_BREAK, expected interrupted failure";
    }
    return ::testing::AssertionSuccess();
}
#else
::testing::AssertionResult launchPosixResumeWorker(
    const std::string& modelName,
    const std::string& downloadPath,
    const std::string& task,
    pid_t& childPid) {
    childPid = fork();
    if (childPid == -1) {
        return ::testing::AssertionFailure() << "fork() failed";
    }
    if (childPid == 0) {
        ovms::Server& childServer = ovms::Server::instance();
        childServer.setShutdownRequest(0);
        char* argv[] = {(char*)"ovms",
            (char*)"--pull",
            (char*)"--source_model",
            (char*)modelName.c_str(),
            (char*)"--model_repository_path",
            (char*)downloadPath.c_str(),
            (char*)"--task",
            (char*)task.c_str()};
        int argc = 8;
        int rc = childServer.start(argc, argv);
        _exit(rc == EXIT_SUCCESS ? 0 : 1);
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult killPosixWorkerAndExpectSigKill(pid_t childPid) {
    if (kill(childPid, SIGKILL) != 0) {
        return ::testing::AssertionFailure() << "Failed to send SIGKILL to child process";
    }

    int childStatus = 0;
    if (waitpid(childPid, &childStatus, 0) != childPid) {
        return ::testing::AssertionFailure() << "waitpid failed for SIGKILL child process";
    }
    if (!WIFSIGNALED(childStatus) || (WTERMSIG(childStatus) != SIGKILL)) {
        return ::testing::AssertionFailure() << "Child did not terminate due to SIGKILL";
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult interruptPosixWorkerAndExpectGracefulExit(pid_t childPid) {
    if (kill(childPid, SIGINT) != 0) {
        return ::testing::AssertionFailure() << "Failed to send SIGINT to child process";
    }

    int childStatus = 0;
    if (waitpid(childPid, &childStatus, 0) != childPid) {
        return ::testing::AssertionFailure() << "waitpid failed for SIGINT child process";
    }
    if (!WIFEXITED(childStatus)) {
        return ::testing::AssertionFailure()
               << "Child was terminated by signal " << (WIFSIGNALED(childStatus) ? WTERMSIG(childStatus) : 0)
               << " instead of graceful exit after SIGINT";
    }
    if (WEXITSTATUS(childStatus) == EXIT_SUCCESS) {
        return ::testing::AssertionFailure() << "Child exited with EXIT_SUCCESS after SIGINT, expected interrupted failure";
    }
    return ::testing::AssertionSuccess();
}
#endif

}  // namespace

// RAII helper class for managing log file lifecycle.
// Creates a log file path and automatically removes it on destruction.
class LogFileGuard {
private:
    std::string logFilePath;

public:
    explicit LogFileGuard(const std::string& path) :
        logFilePath(path) {}

    ~LogFileGuard() {
        if (std::filesystem::exists(logFilePath)) {
            std::error_code ec;
            std::filesystem::remove(logFilePath, ec);
        }
    }

    bool create() {
        std::error_code ec;
        std::filesystem::remove(logFilePath, ec);
        std::ofstream file(logFilePath);
        return file.is_open();
    }

    std::string getContent() const {
        std::ifstream file(logFilePath);
        if (!file.is_open()) {
            return "";
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    bool contains(const std::string& text) const {
        return getContent().find(text) != std::string::npos;
    }

    const std::string& getPath() const {
        return logFilePath;
    }

    bool exists() const {
        return std::filesystem::exists(logFilePath);
    }
};

class HfPull : public TestWithTempDir {
protected:
    static constexpr const char* MODEL_NAMESPACE = "OpenVINO";
    static constexpr const char* MODEL_ID = "Phi-3-mini-FastDraft-50M-int8-ov";
    static constexpr const char* TASK_NAME = "text_generation";

    // Timeout (seconds) for detecting in-progress pull in interrupt tests
    static constexpr int HF_PULL_DETECT_TIMEOUT_SECONDS = 180;
    // Timeout (seconds) for waiting on shutdown request acknowledgement
    static constexpr int HF_PULL_SHUTDOWN_TIMEOUT_SECONDS = 120;
    // Timeout (seconds) for regular server operations
    static constexpr int HF_PULL_SERVER_TIMEOUT_SECONDS = 60;
    // Delay (milliseconds) before sending interrupt signal
    static constexpr int HF_PULL_INTERRUPT_DELAY_MS = 200;
    // Poll interval (milliseconds) when checking for in-progress download
    static constexpr int HF_PULL_POLL_INTERVAL_MS = 100;
    // Max consecutive non-benign filesystem probe errors before failing diagnostics.
    static constexpr int HF_PULL_MAX_CONSECUTIVE_FS_PROBE_ERRORS = 15;

    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;
    std::string modelName;
    std::string downloadPath;
    std::string task;
    std::string modelBasePath;
    std::string modelPath;
    std::string modelPartPath;
    std::string openvinoDetokenizerBinPath;
    std::string openvinoTokenizerBinPath;
    std::string tokenizerModelPath;
    std::string tokenizerJsonPath;
    std::string graphPath;
    std::string gitDirPath;

    void SetUp() override {
        TestWithTempDir::SetUp();
        modelName = std::string(MODEL_NAMESPACE) + "/" + MODEL_ID;
        downloadPath = ovms::FileSystem::joinPath({this->directoryPath, "repository"});
        task = TASK_NAME;
        modelBasePath = ovms::FileSystem::joinPath({downloadPath, MODEL_NAMESPACE, MODEL_ID});
        modelPath = ovms::FileSystem::appendSlash(modelBasePath) + "openvino_model.bin";
        modelPartPath = ovms::FileSystem::appendSlash(modelBasePath) + "openvino_model.binlfs_part";
        openvinoDetokenizerBinPath = ovms::FileSystem::appendSlash(modelBasePath) + "openvino_detokenizer.bin";
        openvinoTokenizerBinPath = ovms::FileSystem::appendSlash(modelBasePath) + "openvino_tokenizer.bin";
        tokenizerModelPath = ovms::FileSystem::appendSlash(modelBasePath) + "tokenizer.model";
        tokenizerJsonPath = ovms::FileSystem::appendSlash(modelBasePath) + "tokenizer.json";
        graphPath = ovms::FileSystem::appendSlash(modelBasePath) + "graph.pbtxt";
        gitDirPath = ovms::FileSystem::appendSlash(modelBasePath) + ".git";
    }

    void ServerPullHfModel(std::string& sourceModel, std::string& downloadPath, std::string& task, int expected_code = 0, int timeoutSeconds = 60) {
        ::SetUpServerForDownload(this->t, this->server, sourceModel, downloadPath, task, expected_code, timeoutSeconds);
    }

    // Variant that captures output to a log file for assertions
    void ServerPullHfModel(std::string& sourceModel, std::string& downloadPath, std::string& task, LogFileGuard& logFile, int expected_code = 0, int timeoutSeconds = 60) {
        ASSERT_TRUE(logFile.create()) << "Failed to create log file at: " << logFile.getPath();
        ::SetUpServerForDownload(this->t, this->server, sourceModel, downloadPath, task, logFile.getPath(), expected_code, timeoutSeconds);
    }

    void ServerPullHfModelWithDraft(std::string& draftModel, std::string& sourceModel, std::string& downloadPath, std::string& task, int expected_code = 0, int timeoutSeconds = 60) {
        ::SetUpServerForDownloadWithDraft(this->t, this->server, draftModel, sourceModel, downloadPath, task, expected_code, timeoutSeconds);
    }

    // Variant with draft model that captures output to a log file for assertions
    void ServerPullHfModelWithDraft(std::string& draftModel, std::string& sourceModel, std::string& downloadPath, std::string& task, LogFileGuard& logFile, int expected_code = 0, int timeoutSeconds = 60) {
        ASSERT_TRUE(logFile.create()) << "Failed to create log file at: " << logFile.getPath();
        ::SetUpServerForDownloadWithDraft(this->t, this->server, draftModel, sourceModel, downloadPath, task, logFile.getPath(), expected_code, timeoutSeconds);
    }

    void SetUpServerForDownloadAndStart(std::string& sourceModel, std::string& downloadPath, std::string& task, int timeoutSeconds = 60) {
        ::SetUpServerForDownloadAndStart(this->t, this->server, sourceModel, downloadPath, task, timeoutSeconds);
    }

    void TearDown() {
        server.setShutdownRequest(1);
        if (t)
            t->join();
        server.setShutdownRequest(0);
        // Clone sets readonly - need to remove it before we can delete on windows
        RemoveReadonlyFileAttributeFromDir(this->directoryPath);
        TestWithTempDir::TearDown();
    }
};

class HfPullCache : public HfPull {
protected:
    static std::once_flag cacheInitFlag;
    static std::unique_ptr<TempDir> cacheDir;
    static std::string cachedRepositoryPath;
    std::string testRepositoryPath;

    void SetUp() override {
        HfPull::SetUp();
        testRepositoryPath = downloadPath;
        initializeSharedCache();
        seedCurrentTestRepository();
    }

    void initializeSharedCache() {
        std::call_once(cacheInitFlag, [this]() {
            cacheDir = std::make_unique<TempDir>();
            std::string sourceModelName = this->modelName;
            std::string cacheDownloadPath = ovms::FileSystem::joinPath({cacheDir->dir.string(), "repository"});
            std::string pullTask = this->task;

            this->ServerPullHfModel(sourceModelName, cacheDownloadPath, pullTask);
            server.setShutdownRequest(1);
            if (t)
                t->join();
            server.setShutdownRequest(0);

            cachedRepositoryPath = cacheDownloadPath;
            ASSERT_TRUE(std::filesystem::exists(cachedRepositoryPath));
        });
    }

    void seedCurrentTestRepository() {
        std::error_code ec;
        std::filesystem::copy(cachedRepositoryPath,
            testRepositoryPath,
            std::filesystem::copy_options::recursive,
            ec);
        ASSERT_EQ(ec, std::errc()) << "Failed to copy cached model repository to test directory";
#ifdef _WIN32
        std::string mutableRepositoryPath = testRepositoryPath;
        RemoveReadonlyFileAttributeFromDir(mutableRepositoryPath);
#endif
    }
};

std::once_flag HfPullCache::cacheInitFlag;
std::unique_ptr<TempDir> HfPullCache::cacheDir = nullptr;
std::string HfPullCache::cachedRepositoryPath;

const std::string expectedGraphContents = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 0,
        }
    }
    input_stream_handler {
        input_stream_handler: "SyncSetInputStreamHandler",
        options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
            sync_set {
            tag_index: "LOOPBACK:0"
            }
        }
        }
    }
    }
)";

const std::string expectedGraphContentsDraft = R"(
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 0,
            # Speculative decoding configuration
            draft_models_path: "OpenVINO-distil-small.en-int4-ov",
        }
    }
    input_stream_handler {
        input_stream_handler: "SyncSetInputStreamHandler",
        options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
            sync_set {
            tag_index: "LOOPBACK:0"
            }
        }
        }
    }
    }
)";

TEST_F(HfPull, Download) {
    GTEST_SKIP() << "Skipping test in CI - PositiveDownloadAndStart has full scope testing.";
    this->ServerPullHfModel(modelName, downloadPath, task);

    std::string basePath = ovms::FileSystem::joinPath({this->directoryPath, "repository", "OpenVINO", "Phi-3-mini-FastDraft-50M-int8-ov"});
    std::string modelPath = ovms::FileSystem::appendSlash(basePath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

// Truncate the file to half its size (OPENVINO_MODEL_BIN_FULL_SIZE_BYTES / 2), keeping the first half.
bool removeSecondHalf(const std::string& fileStr) {
    const std::filesystem::path& file(fileStr);
    std::error_code ec;
    ec.clear();

    if (!std::filesystem::exists(file, ec) || !std::filesystem::is_regular_file(file, ec)) {
        if (!ec)
            ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return false;
    }

    const std::uintmax_t size = std::filesystem::file_size(file, ec);
    if (ec)
        return false;

    const std::uintmax_t newSize = size / 2;  // floor(size/2)
    std::filesystem::resize_file(file, newSize, ec);
    return !ec;
}

bool createGitLfsPointerFile(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file << "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:cecf0224201415144c00cf3a6cf3350306f9c78888d631eb590939a63722fefa\n"
            "size "
         << OPENVINO_MODEL_BIN_FULL_SIZE_BYTES << "\n";

    return true;
}

// Returns lowercase hex SHA-256 string on success, empty string on failure.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
std::string sha256File(std::string_view path, std::error_code& ec) {
    ec.clear();

    std::ifstream ifs(std::string(path), std::ios::binary);
    if (!ifs) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return {};
    }

    SHA256_CTX ctx;
    if (SHA256_Init(&ctx) != 1) {
        ec = std::make_error_code(std::errc::io_error);
        return {};
    }

    // Read in chunks to support large files without high memory usage.
    std::vector<unsigned char> buffer(1 << 20);  // 1 MiB
    while (ifs) {
        ifs.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        std::streamsize got = ifs.gcount();
        if (got > 0) {
            if (SHA256_Update(&ctx, buffer.data(), static_cast<size_t>(got)) != 1) {
                ec = std::make_error_code(std::errc::io_error);
                return {};
            }
        }
    }
    if (!ifs.eof()) {  // read failed not due to EOF
        ec = std::make_error_code(std::errc::io_error);
        return {};
    }

    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    if (SHA256_Final(digest.data(), &ctx) != 1) {
        ec = std::make_error_code(std::errc::io_error);
        return {};
    }

    // Convert to lowercase hex
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::nouppercase;
    for (unsigned char b : digest) {
        oss << std::setw(2) << static_cast<unsigned int>(b);
    }
    return oss.str();
}
#pragma GCC diagnostic pop

class TestHfDownloader : public ovms::HfDownloader {
public:
    TestHfDownloader(const std::string& sourceModel, const std::string& downloadPath, const std::string& hfEndpoint, const std::string& hfToken, const std::string& httpProxy, bool overwrite) :
        HfDownloader(sourceModel, downloadPath, hfEndpoint, hfToken, httpProxy, overwrite) {}
    std::string GetRepoUrl() { return HfDownloader::GetRepoUrl(); }
    std::string GetRepositoryUrlWithPassword() { return HfDownloader::GetRepositoryUrlWithPassword(); }
    bool CheckIfProxySet() { return HfDownloader::CheckIfProxySet(); }
    const std::string& getEndpoint() { return this->hfEndpoint; }
    const std::string& getProxy() { return this->httpProxy; }
    std::string getGraphDirectory(const std::string& downloadPath, const std::string& sourceModel) { return IModelDownloader::getGraphDirectory(downloadPath, sourceModel); }
    std::string getGraphDirectory() { return HfDownloader::getGraphDirectory(); }
    ovms::Status CheckRepositoryStatus(bool checkUntracked) { return HfDownloader::CheckRepositoryStatus(checkUntracked); }
};

TEST_F(HfPullCache, RePull) {
    testing::internal::CaptureStdout();
    this->ServerPullHfModel(modelName, downloadPath, task);
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("Path already exists on local filesystem. Skipping download to path: "), std::string::npos);
    EXPECT_EQ(out.find("LFS file(s) to resume"), std::string::npos);
    EXPECT_EQ(out.find(" Resuming "), std::string::npos);
    // The LFS work-in-progress marker is a SIBLING of the repository directory
    // (e.g. for "<dir>/repository" the marker is "<dir>/repository.lfswip"), not a child of it.
    std::string lfsWipPath = ovms::libgit2::getLfsWipMarkerPath(downloadPath).string();
    EXPECT_EQ(std::filesystem::exists(lfsWipPath), false);
}

TEST_F(HfPullCache, Resume) {
    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeGeneratedGraphHeaders(graphContents)) << graphContents;

    EXPECT_EXIT({
        auto guardOrError = ovms::createLibGitGuard();
        // Check status function
        std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), "", "", "", false);

        // Fails because we want clean and it has the graph.pbtxt after download
        ASSERT_EQ(hfDownloader->CheckRepositoryStatus(true).getCode(), ovms::StatusCode::HF_GIT_STATUS_UNCLEAN);

        exit(0); }, ::testing::ExitedWithCode(0), "");

    std::error_code ec;
    ec.clear();
    std::string expectedDigest = sha256File(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    // Prepare a git repository with a lfs_part file and lfs pointer file to simulate partial download error of a big model
    ASSERT_EQ(removeSecondHalf(modelPath), true);
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_HALF_SIZE_BYTES);

    std::filesystem::rename(modelPath, modelPartPath, ec);
    ASSERT_EQ(ec, std::errc());
    ASSERT_EQ(std::filesystem::file_size(modelPartPath), OPENVINO_MODEL_BIN_HALF_SIZE_BYTES);
    ASSERT_EQ(createGitLfsPointerFile(modelPath), true);

    // Call ovms pull to resume the file
    this->ServerPullHfModel(modelName, downloadPath, task);

    ASSERT_EQ(std::filesystem::exists(modelPartPath), false) << modelPath;
    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeGeneratedGraphHeaders(graphContents)) << graphContents;

    std::string resumedDigest = sha256File(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    ASSERT_EQ(expectedDigest, resumedDigest);
}

// ResumeAfterShutdownRequestAndRerun
TEST_F(HfPull, ResumeShutdown) {
    server.setShutdownRequest(0);
    int firstRunCode = EXIT_SUCCESS;
    char* argv[] = {(char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--task",
        (char*)task.c_str()};
    int argc = 8;
    t.reset(new std::thread([&argc, &argv, &firstRunCode, this]() {
        firstRunCode = this->server.start(argc, argv);
    }));

    // Wait until pull is clearly in-progress before sending shutdown request.
    // A fixed sleep is unreliable: on fast CPU/network setups download may finish
    // before sleep expires, leaving no resumable artifacts.
    bool observedPartialDownload = false;
    ProbeErrorTracker probeErrorTracker;
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(HF_PULL_SERVER_TIMEOUT_SECONDS);
        while (std::chrono::steady_clock::now() < deadline) {
            std::error_code ec;
            const bool hasPartFile = std::filesystem::exists(modelPartPath, ec);
            if (trackProbeError(probeErrorTracker, ec, "exists", modelPartPath, "ResumeShutdown", HF_PULL_MAX_CONSECUTIVE_FS_PROBE_ERRORS)) {
                break;
            }
            ec.clear();
            const bool modelExists = std::filesystem::exists(modelPath, ec);
            if (trackProbeError(probeErrorTracker, ec, "exists", modelPath, "ResumeShutdown", HF_PULL_MAX_CONSECUTIVE_FS_PROBE_ERRORS)) {
                break;
            }
            std::uintmax_t modelSize = 0;
            if (modelExists) {
                ec.clear();
                modelSize = std::filesystem::file_size(modelPath, ec);
                if (ec) {
                    if (trackProbeError(probeErrorTracker, ec, "file_size", modelPath, "ResumeShutdown", HF_PULL_MAX_CONSECUTIVE_FS_PROBE_ERRORS)) {
                        break;
                    }
                    modelSize = 0;
                }
            }
            const bool modelInFlight = modelExists && (modelSize > 0) && (modelSize < OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
            if (hasPartFile || modelInFlight) {
                observedPartialDownload = true;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    server.setShutdownRequest(1);
    EnsureServerModelDownloadFinishedWithTimeout(server, 120, 120);
    if (t)
        t->join();
    server.setShutdownRequest(0);

    if (!probeErrorTracker.persistentFailureReason.empty()) {
        FAIL() << probeErrorTracker.persistentFailureReason;
    }

    if (!observedPartialDownload) {
        FAIL() << "Did not observe in-progress pull before timeout; cannot validate resume-after-shutdown path.";
    }

    EXPECT_NE(firstRunCode, EXIT_SUCCESS);
    auto remainingPointers = ovms::libgit2::findLfsLikeFiles(downloadPath, true);
    EXPECT_FALSE(remainingPointers.empty());

    this->ServerPullHfModel(modelName, downloadPath, task);

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(openvinoDetokenizerBinPath), OPENVINO_DETOKENIZER_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(openvinoTokenizerBinPath), OPENVINO_TOKENIZER_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(tokenizerModelPath), TOKENIZER_MODEL_FULL_SIZE_BYTES);
}

// PullAfterUserRemovedTrackedFileDoesNotRestoreIt
TEST_F(HfPullCache, UserRemoved) {
    std::string preservedFilePath = modelPath;
    std::string removedFilePath = openvinoTokenizerBinPath;
    std::string removedFilePath2 = tokenizerJsonPath;

    ASSERT_TRUE(std::filesystem::exists(preservedFilePath));
    ASSERT_TRUE(std::filesystem::exists(removedFilePath));
    ASSERT_TRUE(std::filesystem::exists(removedFilePath2));

    std::error_code ec;
    std::string preservedDigestBefore = sha256File(preservedFilePath, ec);
    ASSERT_EQ(ec, std::errc());

    ec.clear();
    ASSERT_TRUE(std::filesystem::remove(removedFilePath, ec));
    ASSERT_EQ(ec, std::errc());
    ASSERT_FALSE(std::filesystem::exists(removedFilePath));
    ec.clear();
    ASSERT_TRUE(std::filesystem::remove(removedFilePath2, ec));
    ASSERT_EQ(ec, std::errc());
    ASSERT_FALSE(std::filesystem::exists(removedFilePath2));

    int secondRunCode = EXIT_SUCCESS;
    server.setShutdownRequest(0);
    char* argv[] = {(char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--task",
        (char*)task.c_str()};
    int argc = 8;
    t.reset(new std::thread([&argc, &argv, &secondRunCode, this]() {
        secondRunCode = this->server.start(argc, argv);
    }));

    EnsureServerModelDownloadFinishedWithTimeout(server, 120, 120);

    EXPECT_EQ(secondRunCode, EXIT_SUCCESS);
    EXPECT_FALSE(std::filesystem::exists(removedFilePath));
    EXPECT_FALSE(std::filesystem::exists(removedFilePath2));

    std::string preservedDigestAfter = sha256File(preservedFilePath, ec);
    ASSERT_EQ(ec, std::errc());
    EXPECT_EQ(preservedDigestBefore, preservedDigestAfter);
}

// PullAfterUserEditedTrackedFileDoesNotOverwriteIt
TEST_F(HfPullCache, UserEdited) {
    std::string editedFilePath = openvinoTokenizerBinPath;
    std::string editedFilePath2 = tokenizerJsonPath;

    ASSERT_TRUE(std::filesystem::exists(editedFilePath));
    ASSERT_TRUE(std::filesystem::exists(editedFilePath2));
    const std::uintmax_t originalSize = std::filesystem::file_size(editedFilePath);
    const std::uintmax_t originalSize2 = std::filesystem::file_size(editedFilePath2);

    std::error_code ec;
    std::string originalDigest = sha256File(editedFilePath, ec);
    ASSERT_EQ(ec, std::errc());
    std::string originalDigest2 = sha256File(editedFilePath2, ec);
    ASSERT_EQ(ec, std::errc());

    ASSERT_TRUE(removeSecondHalf(editedFilePath));
    ASSERT_TRUE(removeSecondHalf(editedFilePath2));
    const std::uintmax_t editedSize = std::filesystem::file_size(editedFilePath);
    const std::uintmax_t editedSize2 = std::filesystem::file_size(editedFilePath2);
    ASSERT_LT(editedSize, originalSize);
    ASSERT_LT(editedSize2, originalSize2);

    std::string editedDigestBeforeRerun = sha256File(editedFilePath, ec);
    ASSERT_EQ(ec, std::errc());
    ASSERT_NE(originalDigest, editedDigestBeforeRerun);
    std::string editedDigestBeforeRerun2 = sha256File(editedFilePath2, ec);
    ASSERT_EQ(ec, std::errc());
    ASSERT_NE(originalDigest2, editedDigestBeforeRerun2);

    int secondRunCode = EXIT_SUCCESS;
    server.setShutdownRequest(0);
    char* argv[] = {(char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--task",
        (char*)task.c_str()};
    int argc = 8;
    t.reset(new std::thread([&argc, &argv, &secondRunCode, this]() {
        secondRunCode = this->server.start(argc, argv);
    }));

    EnsureServerModelDownloadFinishedWithTimeout(server, HF_PULL_SHUTDOWN_TIMEOUT_SECONDS, HF_PULL_SHUTDOWN_TIMEOUT_SECONDS);

    EXPECT_EQ(secondRunCode, EXIT_SUCCESS);
    EXPECT_EQ(std::filesystem::file_size(editedFilePath), editedSize);
    EXPECT_EQ(std::filesystem::file_size(editedFilePath2), editedSize2);

    std::string editedDigestAfterRerun = sha256File(editedFilePath, ec);
    ASSERT_EQ(ec, std::errc());
    EXPECT_EQ(editedDigestBeforeRerun, editedDigestAfterRerun);
    EXPECT_NE(originalDigest, editedDigestAfterRerun);

    std::string editedDigestAfterRerun2 = sha256File(editedFilePath2, ec);
    ASSERT_EQ(ec, std::errc());
    EXPECT_EQ(editedDigestBeforeRerun2, editedDigestAfterRerun2);
    EXPECT_NE(originalDigest2, editedDigestAfterRerun2);
}

// PullAgainstNonGitDirectoryWarnsAndSucceedsWithoutChangingFiles
//
// Verifies the production behavior of handleExistingRepositoryWithoutOverwrite()
// when --model_repository_path/<source_model> already contains a user-prepared
// model directory that is NOT a git repository. The expected behavior is:
//   * pull returns success (so subsequent model loading proceeds),
//   * existing files are left untouched (no clone, no resume, no overwrite),
//   * no .lfswip work-in-progress marker is created next to the directory,
//   * the user-facing warning is emitted to stdout.
//
// We simulate the non-git directory by removing the .git folder from the cached
// HfPullCache repository. The model artifacts (openvino_model.bin, tokenizer*,
// graph.pbtxt, etc.) remain in place exactly as a user-supplied directory would.
TEST_F(HfPullCache, PullNonGit) {
    std::string basePath = modelBasePath;
    std::string tokenizerPath = openvinoTokenizerBinPath;
    std::string gitDir = gitDirPath;

    ASSERT_TRUE(std::filesystem::exists(modelPath));
    ASSERT_TRUE(std::filesystem::exists(tokenizerPath));
    ASSERT_TRUE(std::filesystem::is_directory(gitDir));

    // Capture pre-pull file fingerprints so we can confirm pull did not modify them.
    std::error_code ec;
    const std::uintmax_t modelSizeBefore = std::filesystem::file_size(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    const std::uintmax_t tokenizerSizeBefore = std::filesystem::file_size(tokenizerPath, ec);
    ASSERT_EQ(ec, std::errc());
    std::string modelDigestBefore = sha256File(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    std::string tokenizerDigestBefore = sha256File(tokenizerPath, ec);
    ASSERT_EQ(ec, std::errc());

    // Turn the cached repository into an opaque user-provided model directory by
    // removing .git. Drop readonly attributes first so std::filesystem::remove_all
    // succeeds on Windows where libgit2 marks pack files read-only.
    RemoveReadonlyFileAttributeFromDir(gitDir);
    ec.clear();
    std::filesystem::remove_all(gitDir, ec);
    ASSERT_EQ(ec, std::errc()) << "Failed to remove .git from cached repository: " << ec.message();
    ASSERT_FALSE(std::filesystem::exists(gitDir));

#ifdef _WIN32
    // Logger configuration is process-global and initialized earlier in the test
    // binary, so a per-test --log_path does not reliably capture this warning.
    this->ServerPullHfModel(modelName, downloadPath, task);
#else
    testing::internal::CaptureStdout();
    this->ServerPullHfModel(modelName, downloadPath, task);
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("not a git repository"), std::string::npos)
        << "Expected non-git-repository warning in stdout, got:\n"
        << out;
    EXPECT_EQ(out.find("LFS file(s) to resume"), std::string::npos);
    EXPECT_EQ(out.find(" Resuming "), std::string::npos);
#endif

    // No work-in-progress marker should be created next to the model directory.
    const std::string lfsWipPath = ovms::libgit2::getLfsWipMarkerPath(basePath).string();
    EXPECT_FALSE(std::filesystem::exists(lfsWipPath));

    // Files must be left exactly as the user provided them.
    ASSERT_TRUE(std::filesystem::exists(modelPath));
    ASSERT_TRUE(std::filesystem::exists(tokenizerPath));
    EXPECT_EQ(std::filesystem::file_size(modelPath), modelSizeBefore);
    EXPECT_EQ(std::filesystem::file_size(tokenizerPath), tokenizerSizeBefore);

    std::string modelDigestAfter = sha256File(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    std::string tokenizerDigestAfter = sha256File(tokenizerPath, ec);
    ASSERT_EQ(ec, std::errc());
    EXPECT_EQ(modelDigestBefore, modelDigestAfter);
    EXPECT_EQ(tokenizerDigestBefore, tokenizerDigestAfter);

    // .git must NOT have been recreated (no fresh clone happened).
    EXPECT_FALSE(std::filesystem::exists(gitDir));
}

// PullAgainstDirectoryWithEmptyDotGitFailsWithRepositoryError
//
// Companion to HfPullCache.PullNonGit. Verifies that when .git IS present but is
// empty (a corrupt / partially-initialized repository) handleExistingRepositoryWithoutOverwrite()
// does NOT silently succeed: the .git probe passes, GitRepositoryGuard then fails to open
// the repository and the real error is propagated via mapRepositoryOpenFailureToStatus()
// so the operator can act (re-clone, fix permissions, --overwrite_models, ...).
TEST_F(HfPullCache, PullEmptyGitDir) {
    std::string basePath = modelBasePath;
    std::string tokenizerPath = openvinoTokenizerBinPath;
    std::string gitDir = gitDirPath;

    ASSERT_TRUE(std::filesystem::exists(modelPath));
    ASSERT_TRUE(std::filesystem::exists(tokenizerPath));
    ASSERT_TRUE(std::filesystem::is_directory(gitDir));

    // Capture pre-pull file fingerprints so we can confirm pull did not modify them.
    std::error_code ec;
    const std::uintmax_t modelSizeBefore = std::filesystem::file_size(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    const std::uintmax_t tokenizerSizeBefore = std::filesystem::file_size(tokenizerPath, ec);
    ASSERT_EQ(ec, std::errc());
    std::string modelDigestBefore = sha256File(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    std::string tokenizerDigestBefore = sha256File(tokenizerPath, ec);
    ASSERT_EQ(ec, std::errc());

    // Replace the cached .git with an empty directory to simulate corruption / partial init.
    // Drop readonly attributes first so std::filesystem::remove_all succeeds on Windows.
    RemoveReadonlyFileAttributeFromDir(gitDir);
    ec.clear();
    std::filesystem::remove_all(gitDir, ec);
    ASSERT_EQ(ec, std::errc()) << "Failed to remove .git from cached repository: " << ec.message();
    ec.clear();
    std::filesystem::create_directory(gitDir, ec);
    ASSERT_EQ(ec, std::errc()) << "Failed to recreate empty .git directory: " << ec.message();
    ASSERT_TRUE(std::filesystem::is_directory(gitDir));
    ASSERT_TRUE(std::filesystem::is_empty(gitDir));

    // Pull must NOT silently succeed: handleExistingRepositoryWithoutOverwrite should
    // surface the libgit2 open failure (mapRepositoryOpenFailureToStatus -> non-OK).
    this->ServerPullHfModel(modelName, downloadPath, task, EXIT_FAILURE);

    // No work-in-progress marker should be created next to the model directory.
    const std::string lfsWipPath = ovms::libgit2::getLfsWipMarkerPath(basePath).string();
    EXPECT_FALSE(std::filesystem::exists(lfsWipPath));

    // Files must be left exactly as they were on disk.
    ASSERT_TRUE(std::filesystem::exists(modelPath));
    ASSERT_TRUE(std::filesystem::exists(tokenizerPath));
    EXPECT_EQ(std::filesystem::file_size(modelPath), modelSizeBefore);
    EXPECT_EQ(std::filesystem::file_size(tokenizerPath), tokenizerSizeBefore);

    std::string modelDigestAfter = sha256File(modelPath, ec);
    ASSERT_EQ(ec, std::errc());
    std::string tokenizerDigestAfter = sha256File(tokenizerPath, ec);
    ASSERT_EQ(ec, std::errc());
    EXPECT_EQ(modelDigestBefore, modelDigestAfter);
    EXPECT_EQ(tokenizerDigestBefore, tokenizerDigestAfter);

    // .git is still present (we left an empty directory there); no fresh clone happened.
    EXPECT_TRUE(std::filesystem::is_directory(gitDir));
    EXPECT_TRUE(std::filesystem::is_empty(gitDir));
}

#ifdef _WIN32
// Helper test used only as a child process launched by HfPull.ResumeTerminate.
TEST(HfPullWindowsWorker, ResumeTerminateChildProcess) {
    // Enables this helper body only for the parent-launched worker process.
    const char* runWorker = std::getenv("OVMS_RESUME_TERMINATE_WORKER");
    if ((runWorker == nullptr) || (std::string(runWorker) != "1")) {
        GTEST_SKIP() << "Helper test - runs only when launched by HfPull.ResumeTerminate.";
    }

    // Model identifier passed from parent to child worker.
    const char* modelNameEnv = std::getenv("OVMS_RESUME_TERMINATE_MODEL");
    // Target repository path passed from parent to child worker.
    const char* downloadPathEnv = std::getenv("OVMS_RESUME_TERMINATE_DOWNLOAD_PATH");
    // Pull task passed from parent to child worker.
    const char* taskEnv = std::getenv("OVMS_RESUME_TERMINATE_TASK");
    ASSERT_NE(modelNameEnv, nullptr);
    ASSERT_NE(downloadPathEnv, nullptr);
    ASSERT_NE(taskEnv, nullptr);

    ovms::Server& childServer = ovms::Server::instance();
    childServer.setShutdownRequest(0);

    std::string modelName = modelNameEnv;
    std::string downloadPath = downloadPathEnv;
    std::string task = taskEnv;
    char* argv[] = {(char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--task",
        (char*)task.c_str()};
    int argc = 8;

    (void)childServer.start(argc, argv);
}
#endif

// ResumeAfterForcedTerminationAndRerun
TEST_F(HfPull, ResumeTerminate) {
#ifdef _WIN32
    PROCESS_INFORMATION pi;
    ASSERT_TRUE(startWindowsResumeWorker(RESUME_TERMINATE_WORKER_CONFIG, modelName, downloadPath, task, pi));
#else
    pid_t childPid = -1;
    ASSERT_TRUE(launchPosixResumeWorker(modelName, downloadPath, task, childPid));

#endif

    ASSERT_TRUE(waitForResumableInProgressPull(
        modelBasePath,
        modelPath,
        downloadPath,
        OPENVINO_MODEL_BIN_FULL_SIZE_BYTES,
        "ResumeTerminate",
        HF_PULL_DETECT_TIMEOUT_SECONDS,
        HF_PULL_POLL_INTERVAL_MS,
        HF_PULL_INTERRUPT_DELAY_MS,
        HF_PULL_MAX_CONSECUTIVE_FS_PROBE_ERRORS));

    const bool observedPartialDownload = true;
    bool interruptionSent = false;

#ifdef _WIN32
    ASSERT_TRUE(terminateWindowsWorker(pi, HF_PULL_SHUTDOWN_TIMEOUT_SECONDS));
    interruptionSent = true;
    closeWindowsWorkerHandles(pi);
    ASSERT_TRUE(clearWindowsResumeWorkerEnvironment(RESUME_TERMINATE_WORKER_CONFIG));
#else
    ASSERT_TRUE(killPosixWorkerAndExpectSigKill(childPid));
    interruptionSent = true;
#endif

    auto remainingPointers = ovms::libgit2::findLfsLikeFiles(downloadPath, true);
    SPDLOG_INFO("ResumeTerminate test state: observedPartialDownload={}, interruptionSent={}, remainingPointers count={}",
        observedPartialDownload, interruptionSent, remainingPointers.size());
    for (const auto& p : remainingPointers) {
        SPDLOG_INFO("  - {}", p.string());
    }

    EXPECT_FALSE(remainingPointers.empty());

    this->ServerPullHfModel(modelName, downloadPath, task);

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(openvinoDetokenizerBinPath), OPENVINO_DETOKENIZER_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(openvinoTokenizerBinPath), OPENVINO_TOKENIZER_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(tokenizerModelPath), TOKENIZER_MODEL_FULL_SIZE_BYTES);
}

#ifdef _WIN32
// Helper test used only as a child process launched by HfPull.ResumeCtrlC.
// Mirrors HfPullWindowsWorker.ResumeTerminateChildProcess but is invoked separately
// so the parent test can target this child specifically with GenerateConsoleCtrlEvent.
TEST(HfPullWindowsWorker, ResumeCtrlCChildProcess) {
    // Enables this helper body only for the parent-launched worker process.
    const char* runWorker = std::getenv("OVMS_RESUME_CTRLC_WORKER");
    if ((runWorker == nullptr) || (std::string(runWorker) != "1")) {
        GTEST_SKIP() << "Helper test - runs only when launched by HfPull.ResumeCtrlC.";
    }

    // Model identifier passed from parent to child worker.
    const char* modelNameEnv = std::getenv("OVMS_RESUME_CTRLC_MODEL");
    // Target repository path passed from parent to child worker.
    const char* downloadPathEnv = std::getenv("OVMS_RESUME_CTRLC_DOWNLOAD_PATH");
    // Pull task passed from parent to child worker.
    const char* taskEnv = std::getenv("OVMS_RESUME_CTRLC_TASK");
    ASSERT_NE(modelNameEnv, nullptr);
    ASSERT_NE(downloadPathEnv, nullptr);
    ASSERT_NE(taskEnv, nullptr);

    ovms::Server& childServer = ovms::Server::instance();
    childServer.setShutdownRequest(0);

    std::string modelName = modelNameEnv;
    std::string downloadPath = downloadPathEnv;
    std::string task = taskEnv;
    char* argv[] = {(char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--task",
        (char*)task.c_str()};
    int argc = 8;

    (void)childServer.start(argc, argv);
}
#endif

// ResumeAfterCtrlCAndRerun
// Like HfPull.ResumeTerminate, but interrupts the in-flight pull with a graceful
// signal (SIGINT on Linux, CTRL_BREAK_EVENT on Windows) instead of an unconditional
// kill. This exercises the production ctrl+c code path: SIGINT -> onInterrupt() ->
// requestShutdownFromSignal(1) -> libgit2::isCloneCancellationRequestedFromServer()
// returns true and the clone aborts cleanly. After the child exits we verify that
// partial-download artifacts remain on disk and that re-running --pull resumes the
// download to completion.
TEST_F(HfPull, ResumeCtrlC) {
#ifdef _WIN32
    PROCESS_INFORMATION pi;
    ASSERT_TRUE(startWindowsResumeWorker(RESUME_CTRLC_WORKER_CONFIG, modelName, downloadPath, task, pi));
#else
    pid_t childPid = -1;
    ASSERT_TRUE(launchPosixResumeWorker(modelName, downloadPath, task, childPid));
#endif

    ASSERT_TRUE(waitForResumableInProgressPull(
        modelBasePath,
        modelPath,
        downloadPath,
        OPENVINO_MODEL_BIN_FULL_SIZE_BYTES,
        "ResumeCtrlC",
        HF_PULL_DETECT_TIMEOUT_SECONDS,
        HF_PULL_POLL_INTERVAL_MS,
        HF_PULL_INTERRUPT_DELAY_MS,
        HF_PULL_MAX_CONSECUTIVE_FS_PROBE_ERRORS));

    const bool observedPartialDownload = true;
    bool interruptionSent = false;

#ifdef _WIN32
    ASSERT_TRUE(sendCtrlBreakToWindowsWorker(pi, HF_PULL_SHUTDOWN_TIMEOUT_SECONDS));
    interruptionSent = true;
    closeWindowsWorkerHandles(pi);
    ASSERT_TRUE(clearWindowsResumeWorkerEnvironment(RESUME_CTRLC_WORKER_CONFIG));
#else
    ASSERT_TRUE(interruptPosixWorkerAndExpectGracefulExit(childPid));
    interruptionSent = true;
#endif

    SPDLOG_DEBUG("ResumeCtrlC test state: observedPartialDownload={}, interruptionSent={}",
        observedPartialDownload, interruptionSent);
    auto remainingPointers = ovms::libgit2::findLfsLikeFiles(downloadPath, true);
    EXPECT_FALSE(remainingPointers.empty());

    this->ServerPullHfModel(modelName, downloadPath, task);

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(openvinoDetokenizerBinPath), OPENVINO_DETOKENIZER_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(openvinoTokenizerBinPath), OPENVINO_TOKENIZER_BIN_FULL_SIZE_BYTES);
    ASSERT_EQ(std::filesystem::file_size(tokenizerModelPath), TOKENIZER_MODEL_FULL_SIZE_BYTES);
}

TEST_F(HfPull, Start) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    // EnvGuard guard;
    // guard.set("HF_ENDPOINT", "https://modelscope.cn");
    // guard.set("HF_ENDPOINT", "https://hf-mirror.com");
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    this->filesToPrintInCaseOfFailure.emplace_back("config.json");
    this->SetUpServerForDownloadAndStart(modelName, downloadPath, task);

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

TEST_F(HfPull, OutOfOvOrg) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    // EnvGuard guard;
    // guard.set("HF_ENDPOINT", "https://modelscope.cn");
    // guard.set("HF_ENDPOINT", "https://hf-mirror.com");

    std::string downloadPathRoot = this->directoryPath;
    this->ServerPullHfModel(modelName, downloadPathRoot, task);

    // Shutdown
    server.setShutdownRequest(1);
    if (t)
        t->join();
    server.setShutdownRequest(0);
    std::string basePath = ovms::FileSystem::joinPath({this->directoryPath, "OpenVINO", "Phi-3-mini-FastDraft-50M-int8-ov"});
    std::string modelPath = ovms::FileSystem::appendSlash(basePath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeGeneratedGraphHeaders(graphContents)) << graphContents;

    std::string changePath = ovms::FileSystem::joinPath({this->directoryPath, "OpenVINO"});
    std::string newPath = ovms::FileSystem::joinPath({this->directoryPath, "META"});
    try {
        std::filesystem::rename(changePath, newPath);
        std::cout << "Directory renamed successfully.\n";
    } catch (const std::filesystem::filesystem_error& e) {
        std::cout << "Error: " << e.what() << '\n';
        ASSERT_EQ(1, 0);
    }

    std::string modelName2 = "META/Phi-3-mini-FastDraft-50M-int8-ov";
    std::filesystem::file_time_type ftime1 = std::filesystem::last_write_time(newPath);
    this->SetUpServerForDownloadAndStart(modelName2, downloadPathRoot, task);
    std::filesystem::file_time_type ftime2 = std::filesystem::last_write_time(newPath);
    ASSERT_EQ(ftime1, ftime2);
}

TEST_F(HfPull, StartOutsideOvOrg) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    this->filesToPrintInCaseOfFailure.emplace_back("config.json");
    std::string modelName = "AIFunOver/SmolLM2-360M-Instruct-openvino-4bit";
    std::string downloadPath = ovms::FileSystem::joinPath({this->directoryPath, "repository"});
    std::string task = "text_generation";
    this->SetUpServerForDownloadAndStart(modelName, downloadPath, task);

    std::string basePath = ovms::FileSystem::joinPath({this->directoryPath, "repository", "AIFunOver", "SmolLM2-360M-Instruct-openvino-4bit"});
    std::string modelPath = ovms::FileSystem::appendSlash(basePath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeGeneratedGraphHeaders(graphContents)) << graphContents;
}

TEST_F(HfPull, DraftModel) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();  // CVS-180127
    // EnvGuard guard;
    // guard.set("HF_ENDPOINT", "https://modelscope.cn");
    // guard.set("HF_ENDPOINT", "https://hf-mirror.com");
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    std::string draftModel = "OpenVINO/distil-small.en-int4-ov";
    this->ServerPullHfModelWithDraft(draftModel, modelName, this->directoryPath, task);

    std::string basePath = ovms::FileSystem::joinPath({this->directoryPath, "OpenVINO", "Phi-3-mini-FastDraft-50M-int8-ov"});
    std::string modelPath = ovms::FileSystem::appendSlash(basePath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), OPENVINO_MODEL_BIN_FULL_SIZE_BYTES);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContentsDraft, removeGeneratedGraphHeaders(graphContents)) << graphContents;

    std::string basePath2 = ovms::FileSystem::joinPath({basePath, "OpenVINO-distil-small.en-int4-ov"});
    std::string modelPath2 = ovms::FileSystem::appendSlash(basePath2) + "openvino_tokenizer.bin";

    ASSERT_EQ(std::filesystem::exists(modelPath2), true) << modelPath2;
    ASSERT_EQ(std::filesystem::file_size(modelPath2), DRAFT_OPENVINO_TOKENIZER_BIN_SIZE_BYTES);
}

class TestOptimumDownloader : public ovms::OptimumDownloader {
public:
    TestOptimumDownloader(const ovms::HFSettingsImpl& inHfSettings) :
        ovms::OptimumDownloader(inHfSettings.exportSettings, inHfSettings.task, inHfSettings.sourceModel, ovms::HfDownloader::getGraphDirectory(inHfSettings.downloadPath, inHfSettings.sourceModel), inHfSettings.overwriteModels) {}
    std::string getExportCmd() { return ovms::OptimumDownloader::getExportCmd(); }
    std::string getConvertCmd() { return ovms::OptimumDownloader::getConvertCmd(); }
    std::string getGraphDirectory() { return ovms::OptimumDownloader::getGraphDirectory(); }
    void setExportCliCheckCommand(const std::string& input) { this->OPTIMUM_CLI_CHECK_COMMAND = input; }
    void setConvertCliCheckCommand(const std::string& input) { this->CONVERT_TOKENIZER_CHECK_COMMAND = input; }
    void setExportCliExportCommand(const std::string& input) { this->OPTIMUM_CLI_EXPORT_COMMAND = input; }
    void setConvertCliExportCommand(const std::string& input) { this->CONVERT_TOKENIZER_EXPORT_COMMAND = input; }
    ovms::Status checkRequiredToolsArePresent() { return ovms::OptimumDownloader::checkRequiredToolsArePresent(); }
    bool checkIfDetokenizerFileIsExported() { return ovms::OptimumDownloader::checkIfDetokenizerFileIsExported(); }
    bool checkIfTokenizerFileIsExported() { return ovms::OptimumDownloader::checkIfTokenizerFileIsExported(); }
};

TEST(HfDownloaderClassTest, Methods) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::string hfEndpoint = "www.new_hf.com/";
    std::string hfToken = "123$$o_O123!AAbb";
    std::string httpProxy = "https://proxy_test1:123";
    std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, httpProxy, false);
    ASSERT_EQ(hfDownloader->getProxy(), httpProxy);
    ASSERT_EQ(hfDownloader->CheckIfProxySet(), true);

    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).CheckIfProxySet(), false);
    ASSERT_EQ(hfDownloader->getEndpoint(), "www.new_hf.com/");
    ASSERT_EQ(hfDownloader->GetRepoUrl(), "www.new_hf.com/model/name");
    ASSERT_EQ(hfDownloader->GetRepositoryUrlWithPassword(), "123$$o_O123!AAbb:123$$o_O123!AAbb@www.new_hf.com/model/name");

    std::string expectedPath = downloadPath + "/" + modelName;
#ifdef _WIN32
    std::replace(expectedPath.begin(), expectedPath.end(), '/', '\\');
#endif
    ASSERT_EQ(hfDownloader->getGraphDirectory(downloadPath, modelName), expectedPath);
    ASSERT_EQ(hfDownloader->getGraphDirectory(), expectedPath);
}

TEST(HfDownloaderClassTest, RepositoryStatusCheckErrors) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::string hfEndpoint = "www.new_hf.com/";
    std::string hfToken = "123$$o_O123!AAbb";
    std::string httpProxy = "https://proxy_test1:123";
    EXPECT_EXIT({
        std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, httpProxy, false);
        // Fails without libgit init
        ASSERT_EQ(hfDownloader->CheckRepositoryStatus(true).getCode(), ovms::StatusCode::HF_GIT_LIBGIT2_NOT_INITIALIZED);
        ASSERT_EQ(hfDownloader->CheckRepositoryStatus(false).getCode(), ovms::StatusCode::HF_GIT_LIBGIT2_NOT_INITIALIZED);
        exit(0); }, ::testing::ExitedWithCode(0), "");

    EXPECT_EXIT({
        std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, httpProxy, false);
        auto guardOrError = ovms::createLibGitGuard();
        ASSERT_EQ(std::holds_alternative<ovms::Status>(guardOrError), false);

        // Path does not exist
        ASSERT_EQ(hfDownloader->CheckRepositoryStatus(true).getCode(), ovms::StatusCode::HF_GIT_STATUS_FAILED_TO_RESOLVE_PATH);
        ASSERT_EQ(hfDownloader->CheckRepositoryStatus(false).getCode(), ovms::StatusCode::HF_GIT_STATUS_FAILED_TO_RESOLVE_PATH);

        // Path not a git repository
        TempDir td;
        downloadPath = td.dir.string();

        std::unique_ptr<TestHfDownloader> existingHfDownloader = std::make_unique<TestHfDownloader>(modelName, downloadPath, hfEndpoint, hfToken, httpProxy, false);
        ASSERT_EQ(existingHfDownloader->CheckRepositoryStatus(true).getCode(), ovms::StatusCode::HF_GIT_STATUS_FAILED);
        ASSERT_EQ(existingHfDownloader->CheckRepositoryStatus(false).getCode(), ovms::StatusCode::HF_GIT_STATUS_FAILED);
        exit(0); }, ::testing::ExitedWithCode(0), "");
}

TEST(HfDownloaderClassTest, CloneCancellationFollowsServerShutdownRequest) {
    ovms::Server& server = ovms::Server::instance();
    server.setShutdownRequest(0);
    EXPECT_FALSE(ovms::libgit2::isCloneCancellationRequestedFromServer());

    server.setShutdownRequest(1);
    EXPECT_TRUE(ovms::libgit2::isCloneCancellationRequestedFromServer());

    server.setShutdownRequest(0);
}

class TestOptimumDownloaderSetup : public ::testing::Test {
public:
    ovms::HFSettingsImpl inHfSettings;
    std::string cliMockPath;
    void SetUp() override {
        inHfSettings.sourceModel = "model/name";
        inHfSettings.downloadPath = "/path/to/Download";
        inHfSettings.exportSettings.precision = "fp64";
        inHfSettings.exportSettings.extraQuantizationParams = "--someOptimumParam --anotherOptParam value";
        inHfSettings.task = ovms::TEXT_GENERATION_GRAPH;
        inHfSettings.downloadType = ovms::OPTIMUM_CLI_DOWNLOAD;
#ifdef _WIN32
        cliMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/optimum-cli.exe");
#else
        cliMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/optimum-cli");
#endif
    }
};

class TestOptimumDownloaderSetupWithFile : public TestOptimumDownloaderSetup {
public:
    ovms::HFSettingsImpl inHfSettings;
    std::string cliMockPath;
    std::filesystem::path file_path;
    std::filesystem::path dir_path;
    void TearDown() override {
        std::filesystem::remove(file_path);
        std::filesystem::remove_all(dir_path);
    }
};

TEST_F(TestOptimumDownloaderSetup, Methods) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedPath = inHfSettings.downloadPath + "/" + inHfSettings.sourceModel;
    std::string expectedCmd = "optimum-cli export openvino --model model/name --trust-remote-code  --weight-format fp64 --someOptimumParam --anotherOptParam value \\path\\to\\Download\\model\\name";
    std::string expectedCmd2 = "convert_tokenizer model/name --with-detokenizer -o \\path\\to\\Download\\model\\name";
#ifdef _WIN32
    std::replace(expectedPath.begin(), expectedPath.end(), '/', '\\');
#endif
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
    std::replace(expectedCmd2.begin(), expectedCmd2.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getGraphDirectory(), expectedPath);
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
    ASSERT_EQ(optimumDownloader->getConvertCmd(), expectedCmd2);
}

TEST_F(TestOptimumDownloaderSetup, RerankExportCmd) {
    inHfSettings.task = ovms::RERANK_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --disable-convert-tokenizer --model model/name --trust-remote-code  --weight-format fp64 --task text-classification  --someOptimumParam --anotherOptParam value \\path\\to\\Download\\model\\name";
    std::string expectedCmd2 = "convert_tokenizer model/name -o \\path\\to\\Download\\model\\name";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
    std::replace(expectedCmd2.begin(), expectedCmd2.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
    ASSERT_EQ(optimumDownloader->getConvertCmd(), expectedCmd2);
}

TEST_F(TestOptimumDownloaderSetup, ImageGenExportCmd) {
    inHfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --model model/name --weight-format fp64 --someOptimumParam --anotherOptParam value \\path\\to\\Download\\model\\name";
    std::string expectedCmd2 = "";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
    ASSERT_EQ(optimumDownloader->getConvertCmd(), expectedCmd2);
}

TEST_F(TestOptimumDownloaderSetup, ImageGenExportCmdNoExtraParams) {
    inHfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    inHfSettings.exportSettings.extraQuantizationParams = std::nullopt;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --model model/name --weight-format fp64 \\path\\to\\Download\\model\\name";
    std::string expectedCmd2 = "";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
    ASSERT_EQ(optimumDownloader->getConvertCmd(), expectedCmd2);
}

TEST_F(TestOptimumDownloaderSetup, EmbeddingsExportCmd) {
    inHfSettings.task = ovms::EMBEDDINGS_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --disable-convert-tokenizer --task feature-extraction --library sentence_transformers --model model/name --trust-remote-code  --weight-format fp64 --someOptimumParam --anotherOptParam value \\path\\to\\Download\\model\\name";
    std::string expectedCmd2 = "convert_tokenizer model/name -o \\path\\to\\Download\\model\\name";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
    std::replace(expectedCmd2.begin(), expectedCmd2.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
    ASSERT_EQ(optimumDownloader->getConvertCmd(), expectedCmd2);
}

TEST_F(TestOptimumDownloaderSetup, TextToSpeechExportCmd) {
    inHfSettings.task = ovms::TEXT_TO_SPEECH_GRAPH;
    inHfSettings.exportSettings.vocoder = "microsoft/speecht5_hifigan";
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --model-kwargs \"{\"vocoder\": \"microsoft/speecht5_hifigan\"}\" --model model/name --trust-remote-code  --weight-format fp64 --someOptimumParam --anotherOptParam value \\path\\to\\Download\\model\\name";
    std::string expectedCmd2 = "convert_tokenizer model/name -o \\path\\to\\Download\\model\\name";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
    std::replace(expectedCmd2.begin(), expectedCmd2.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
    ASSERT_EQ(optimumDownloader->getConvertCmd(), expectedCmd2);
}

TEST_F(TestOptimumDownloaderSetup, SpeechToTextExportCmd) {
    inHfSettings.task = ovms::SPEECH_TO_TEXT_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --model model/name --trust-remote-code  --weight-format fp64 --someOptimumParam --anotherOptParam value \\path\\to\\Download\\model\\name";
    std::string expectedCmd2 = "convert_tokenizer model/name -o \\path\\to\\Download\\model\\name";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
    std::replace(expectedCmd2.begin(), expectedCmd2.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
    ASSERT_EQ(optimumDownloader->getConvertCmd(), expectedCmd2);
}

TEST_F(TestOptimumDownloaderSetup, DetokenizerCheckNegative) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->checkIfDetokenizerFileIsExported(), false);
    ASSERT_EQ(optimumDownloader->checkIfTokenizerFileIsExported(), false);
}

TEST_F(TestOptimumDownloaderSetupWithFile, DetokenizerCheckPositive) {
    file_path = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/model/name/openvino_detokenizer.xml");
    inHfSettings.sourceModel = "model/name";
    inHfSettings.downloadPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/");
    dir_path = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/model/");
    std::filesystem::create_directories(getGenericFullPathForBazelOut("/ovms/bazel-bin/src/model/name"));
    std::ofstream ofs(file_path);  // Creates an empty file
    ofs.close();
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->checkIfDetokenizerFileIsExported(), true);
}

TEST_F(TestOptimumDownloaderSetupWithFile, TokenizerCheckPositive) {
    file_path = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/model/name/openvino_tokenizer.xml");
    inHfSettings.sourceModel = "model/name";
    inHfSettings.downloadPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/");
    dir_path = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/model/");
    std::filesystem::create_directories(getGenericFullPathForBazelOut("/ovms/bazel-bin/src/model/name"));
    std::ofstream ofs(file_path);  // Creates an empty file
    ofs.close();
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->checkIfTokenizerFileIsExported(), true);
}

TEST_F(TestOptimumDownloaderSetup, UnknownExportCmd) {
    inHfSettings.task = ovms::UNKNOWN_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->getExportCmd(), "");
}

TEST_F(TestOptimumDownloaderSetup, NegativeWrongPath) {
    inHfSettings.downloadPath = "../path/to/Download";
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->downloadModel(), ovms::StatusCode::PATH_INVALID);
}

TEST_F(TestOptimumDownloaderSetup, NegativeExportCommandFailed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    optimumDownloader->setExportCliCheckCommand("echo ");
    optimumDownloader->setConvertCliCheckCommand("echo ");
    optimumDownloader->setExportCliExportCommand("NonExistingCommand22");
    ASSERT_EQ(optimumDownloader->downloadModel(), ovms::StatusCode::HF_RUN_OPTIMUM_CLI_EXPORT_FAILED);
}

TEST_F(TestOptimumDownloaderSetup, NegativeConvertCommandFailed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    optimumDownloader->setExportCliCheckCommand("echo ");
    optimumDownloader->setConvertCliCheckCommand("echo ");
    optimumDownloader->setExportCliExportCommand("echo ");
    optimumDownloader->setConvertCliExportCommand("nonExistingCommand222");
    ASSERT_EQ(optimumDownloader->downloadModel(), ovms::StatusCode::HF_RUN_CONVERT_TOKENIZER_EXPORT_FAILED);
}

TEST_F(TestOptimumDownloaderSetup, NegativeCheckOptimumExistsCommandFailed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    optimumDownloader->setExportCliCheckCommand("NonExistingCommand33");
    optimumDownloader->setConvertCliCheckCommand("echo ");
    ASSERT_EQ(optimumDownloader->checkRequiredToolsArePresent(), ovms::StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI);
}

TEST_F(TestOptimumDownloaderSetup, NegativeCheckConverterExistsCommandFailed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    optimumDownloader->setExportCliCheckCommand("echo ");
    optimumDownloader->setConvertCliCheckCommand("NonExistingCommand33");
    ASSERT_EQ(optimumDownloader->checkRequiredToolsArePresent(), ovms::StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI);
}

TEST_F(TestOptimumDownloaderSetup, PositiveOptimumExistsCommandPassed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    cliMockPath += " -h";
    optimumDownloader->setExportCliCheckCommand(cliMockPath);
    optimumDownloader->setConvertCliCheckCommand("echo ");
    ASSERT_EQ(optimumDownloader->checkRequiredToolsArePresent(), ovms::StatusCode::OK);
}

TEST_F(TestOptimumDownloaderSetup, PositiveOptimumExportCommandPassed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string cliCheckCommand = cliMockPath += " -h";
    optimumDownloader->setExportCliCheckCommand(cliCheckCommand);
    optimumDownloader->setConvertCliCheckCommand("echo ");
    cliMockPath += " export";
    optimumDownloader->setExportCliExportCommand(cliMockPath);
    optimumDownloader->setConvertCliExportCommand("echo ");
    ASSERT_EQ(optimumDownloader->downloadModel(), ovms::StatusCode::OK);
}

TEST(HfDownloaderClassTest, ProtocollsWithPassword) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::string hfEndpoint = "www.new_hf.com/";
    std::string hfToken = "";
    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "www.new_hf.com/model/name");
    hfEndpoint = "https://www.new_hf.com/";
    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "https://www.new_hf.com/model/name");
    hfEndpoint = "www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "http://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "http://123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "git://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "git://123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "ssh://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "ssh://123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "what_ever_is_here://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName), hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "what_ever_is_here://123!$token:123!$token@www.new_hf.com/model/name");
}

TEST_F(HfPull, MethodsNegative) {
    EXPECT_EQ(TestHfDownloader("name/test", "../some/path", "", "", "", false).downloadModel(), ovms::StatusCode::PATH_INVALID);
    // Library not initialized
    EXPECT_EQ(TestHfDownloader("name/test", ovms::IModelDownloader::getGraphDirectory(this->directoryPath, "name2/test"), "", "", "", false).downloadModel(), ovms::StatusCode::HF_GIT_CLONE_FAILED);
}

TEST_F(HfPull, CloneCancelledByShutdownRequest) {
    std::string downloadPath = ovms::FileSystem::joinPath({this->directoryPath, "repository_cancel"});
    std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(
        modelName,
        ovms::IModelDownloader::getGraphDirectory(downloadPath, modelName),
        "https://huggingface.co/",
        "",
        "",
        false);

    server.setShutdownRequest(1);
    EXPECT_EQ(hfDownloader->downloadModel(), ovms::StatusCode::HF_GIT_CLONE_CANCELLED);
    server.setShutdownRequest(0);
}

class TestHfPullModelModule : public ovms::HfPullModelModule {
public:
    const std::string GetHfToken() const { return HfPullModelModule::GetHfToken(); }
    const std::string GetHfEndpoint() const { return HfPullModelModule::GetHfEndpoint(); }
    const std::string GetProxy() const { return HfPullModelModule::GetProxy(); }
};

class HfDownloaderHfEnvTest : public ::testing::Test {
public:
    std::string proxy_env = "https_proxy";
    std::string token_env = "HF_TOKEN";
    std::string endpoint_env = "HF_ENDPOINT";
    EnvGuard guard;
};

TEST(Libgt2InitGuardTest, LfsFilterCaptureDefaultResumeOptions) {
    // Need new process beacase we use INIT_ONCE in libgit2 lfs filter for env variables and once they are set they are set for the whole process lifetime
    EXPECT_EXIT({
        // Act: capture stdout during object construction
        testing::internal::CaptureStdout();
        {
            auto guardOrError = ovms::createLibGitGuard();
            ASSERT_EQ(std::holds_alternative<ovms::Status>(guardOrError), false);
        }
        std::string output = testing::internal::GetCapturedStdout();

        // Optional: trim trailing newline
        if (!output.empty() && output.back() == '\n') {
            output.pop_back();
        }

        EXPECT_THAT(output, ::testing::HasSubstr("[INFO] LFS resume: attempts=5 interval=10 s"));
        exit(0); }, ::testing::ExitedWithCode(0), "");
}

TEST(Libgt2InitGuardTest, LfsFilterCaptureNonDefaultResumeOptions) {
    // Need new process beacase we use INIT_ONCE in libgit2 lfs filter for env variables and once they are set they are set for the whole process lifetime
    EXPECT_EXIT({
        EnvGuard guard;
        guard.set("GIT_LFS_RESUME_ATTEMPTS", "3");
        guard.set("GIT_LFS_RESUME_INTERVAL_SECONDS", "20");
        // Act: capture stdout during object construction
        testing::internal::CaptureStdout();
        {
            auto guardOrError = ovms::createLibGitGuard();
            ASSERT_EQ(std::holds_alternative<ovms::Status>(guardOrError), false);
        }
        std::string output = testing::internal::GetCapturedStdout();

        // Optional: trim trailing newline
        if (!output.empty() && output.back() == '\n') {
            output.pop_back();
        }

        EXPECT_THAT(output, ::testing::HasSubstr("[INFO] LFS resume: attempts=3 interval=20 s"));
        exit(0); }, ::testing::ExitedWithCode(0), "");
}

TEST_F(HfDownloaderHfEnvTest, Methods) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::unique_ptr<TestHfPullModelModule> testHfPullModelModule = std::make_unique<TestHfPullModelModule>();

    std::string proxy = "https://proxy_test1:123";
    this->guard.unset(proxy_env);
    ASSERT_EQ(testHfPullModelModule->GetProxy(), "");
    this->guard.set(proxy_env, proxy);
    ASSERT_EQ(testHfPullModelModule->GetProxy(), proxy);

    std::string token = "123$$o_O123!AAbb";
    this->guard.unset(token_env);
    ASSERT_EQ(testHfPullModelModule->GetHfToken(), "");
    this->guard.set(token_env, token);
    ASSERT_EQ(testHfPullModelModule->GetHfToken(), token);

    std::string endpoint = "www.new_hf.com";
    this->guard.unset(endpoint_env);
    ASSERT_EQ(testHfPullModelModule->GetHfEndpoint(), "https://huggingface.co/");
    this->guard.set(endpoint_env, endpoint);

    std::string hfEndpoint = testHfPullModelModule->GetHfEndpoint();
    ASSERT_EQ(hfEndpoint, "www.new_hf.com/");
}

class HfDownloadModelModule : public TestWithTempDir {};

TEST_F(HfDownloadModelModule, TestInvalidProxyTimeout) {
#ifdef _WIN32
    GTEST_SKIP() << "Setting timeout does not work on windows - there is some default used ~80s which is too long";
    // https://github.com/libgit2/libgit2/issues/7072
#endif
    ovms::HfPullModelModule hfModule;
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath) + "repository";  // Cleanup

    char* n_argv[] = {
        (char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--task",
        (char*)"text_generation",
        nullptr};

    int arg_count = 8;
    ConstructorEnabledConfig config;
    {
        EnvGuard eGuard;
        // prepareLibgit2Opts() in hf_pull_model_module.cpp only applies the
        // GIT_OPT_SET_SERVER_CONNECT_TIMEOUT option when https_proxy is empty,
        // so we always clear https_proxy here.
        //
        // To make the timeout actually fire we need the destination to be
        // unreachable. The behavior depends on the host's network setup:
        //   * Host originally used a proxy (https_proxy was set in the
        //     environment): the host is on a proxy-only network where a
        //     direct connection to huggingface.co will hang and hit the
        //     timeout. Keep the default HF_ENDPOINT.
        //   * Host has no proxy configured (direct internet access): a direct
        //     connection to huggingface.co would succeed within the 1 s
        //     timeout and the assertion below would fail. Redirect the clone
        //     to an unroutable RFC 5737 TEST-NET-1 address so the connect
        //     must time out.
        const char* hostHttpsProxy = std::getenv("https_proxy");
        const bool hostHadProxy = (hostHttpsProxy != nullptr) && (std::string(hostHttpsProxy) != "");
        eGuard.set("https_proxy", "");
        if (!hostHadProxy) {
            eGuard.set("HF_ENDPOINT", "https://192.0.2.1/");
        }
        const std::string timeoutConnectVal = "1000";
        eGuard.set(ovms::HfPullModelModule::GIT_SERVER_CONNECT_TIMEOUT_ENV, timeoutConnectVal);
        config.parse(arg_count, const_cast<char**>(n_argv));
        auto status = hfModule.start(config);
        ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
        ovms::Timer<1> timer;
        timer.start(0);
        status = hfModule.clone();
        EXPECT_NE(status, ovms::StatusCode::OK) << status.string();
        timer.stop(0);
        double timeSpentMs = timer.elapsed<std::chrono::microseconds>(0) / 1000;
        SPDLOG_DEBUG("Time spent:{} ms", timeSpentMs);
        EXPECT_LE(timeSpentMs, 3 * ovms::stoi32(timeoutConnectVal).value()) << "We should timeout before 1ms has passed but clone worked for: " << timeSpentMs << "ms > " << timeoutConnectVal << "ms. Status: " << status.string();
    }
    SPDLOG_TRACE("After guard closure");
}

TEST(Libgit2Framework, TimeoutTestProxy) {
    GTEST_SKIP() << "Does not work with proxy set";
    // https://github.com/libgit2/libgit2/issues/7072
    git_libgit2_init();

    git_repository* cloned_repo = NULL;
    git_clone_options clone_opts = GIT_CLONE_OPTIONS_INIT;
    git_checkout_options checkout_opts = GIT_CHECKOUT_OPTIONS_INIT;

    checkout_opts.checkout_strategy = GIT_CHECKOUT_SAFE;
    clone_opts.checkout_opts = checkout_opts;
    // Use proxy
    if (true) {
        clone_opts.fetch_opts.proxy_opts.type = GIT_PROXY_SPECIFIED;
        clone_opts.fetch_opts.proxy_opts.url = "http://proxy-dmz.intel.com:912";
    }
    int e = git_libgit2_opts(GIT_OPT_SET_SERVER_CONNECT_TIMEOUT, 1000);
    EXPECT_EQ(e, 0);

    std::string passRepoUrl = "https://huggingface.co/OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    const char* path = "/tmp/model";
    int error = git_clone(&cloned_repo, passRepoUrl.c_str(), path, &clone_opts);
    if (error != 0) {
        const git_error* err = git_error_last();
        if (err) {
            std::cout << "Libgit2 clone error:" << err->klass << "; " << err->message << std::endl;
        }
        EXPECT_EQ(error, 0);
    } else if (cloned_repo) {
        git_repository_free(cloned_repo);
    }

    git_libgit2_shutdown();
}

class DefaultEmptyValuesConfig : public ovms::Config {
public:
    DefaultEmptyValuesConfig() :
        Config() {
        std::string port{"9000"};
        randomizeAndEnsureFree(port);
        this->serverSettings.grpcPort = std::stoul(port);
    }

    ovms::ServerSettingsImpl& getServerSettings() {
        return this->serverSettings;
    }

    ovms::ModelsSettingsImpl& getModelSettings() {
        return this->modelsSettings;
    }
};

class ServerShutdownGuard {
    ovms::Server& ovmsServer;

public:
    ServerShutdownGuard(ovms::Server& ovmsServer) :
        ovmsServer(ovmsServer) {}
    ~ServerShutdownGuard() {
        ovmsServer.shutdownModules();
    }
};

TEST(ServerModulesBehaviorTests, ListModelErrorAndExpectSuccessAndNoOtherModulesStarted) {
    std::unique_ptr<ServerShutdownGuard> serverGuard;
    ovms::Server& server = ovms::Server::instance();
    DefaultEmptyValuesConfig config;
    config.getServerSettings().serverMode = ovms::LIST_MODELS_MODE;
    auto retCode = server.startModules(config);
    // Empty config.getServerSettings().hfSettings.downloadPath
    // [error][listmodels.cpp:121] Path is not a directory:
    EXPECT_TRUE(retCode.ok()) << retCode.string();
    serverGuard = std::make_unique<ServerShutdownGuard>(server);
    EXPECT_TRUE(server.getModule(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME) != nullptr);
    ASSERT_EQ(server.getModule(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME)->getState(), ovms::ModuleState::INITIALIZED);
    ASSERT_EQ(server.getModule(ovms::GRPC_SERVER_MODULE_NAME), nullptr);
    ASSERT_EQ(server.getModule(ovms::HF_MODEL_PULL_MODULE_NAME), nullptr);
}

TEST(ServerModulesBehaviorTests, ModifyConfigErrorAndExpectFailAndNoOtherModulesStarted) {
    std::unique_ptr<ServerShutdownGuard> serverGuard;
    ovms::Server& server = ovms::Server::instance();
    DefaultEmptyValuesConfig config;
    config.getServerSettings().serverMode = ovms::MODIFY_CONFIG_MODE;
    auto retCode = server.startModules(config);
    // Empty modelSettings.configPath
    // [error][config_export.cpp:197] Directory path empty:
    EXPECT_TRUE(!retCode.ok()) << retCode.string();
    serverGuard = std::make_unique<ServerShutdownGuard>(server);
    EXPECT_TRUE(server.getModule(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME) != nullptr);
    ASSERT_EQ(server.getModule(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME)->getState(), ovms::ModuleState::INITIALIZED);
    ASSERT_EQ(server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME), nullptr);
    ASSERT_EQ(server.getModule(ovms::HF_MODEL_PULL_MODULE_NAME), nullptr);
}

TEST(ServerModulesBehaviorTests, PullModeErrorAndExpectFailAndNoOtherModulesStarted) {
    std::unique_ptr<ServerShutdownGuard> serverGuard;
    ovms::Server& server = ovms::Server::instance();
    DefaultEmptyValuesConfig config;
    config.getServerSettings().serverMode = ovms::HF_PULL_MODE;
    auto retCode = server.startModules(config);
    // Empty config.getServerSettings().hfSettings.downloadPath
    // [error][libit2.cpp:336] Libgit2 clone error: 6 message: cannot pick working directory for non-bare repository that isn't a '.git' directory
    EXPECT_TRUE(!retCode.ok()) << retCode.string();
    serverGuard = std::make_unique<ServerShutdownGuard>(server);
    EXPECT_TRUE(server.getModule(ovms::HF_MODEL_PULL_MODULE_NAME) != nullptr);
    ASSERT_EQ(server.getModule(ovms::HF_MODEL_PULL_MODULE_NAME)->getState(), ovms::ModuleState::INITIALIZED);
    ASSERT_EQ(server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME), nullptr);
    ASSERT_EQ(server.getModule(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME), nullptr);
}

TEST(ServerModulesBehaviorTests, PullAndStartModeErrorAndExpectFailAndCheckOtherModules) {
    std::unique_ptr<ServerShutdownGuard> serverGuard;
    ovms::Server& server = ovms::Server::instance();
    DefaultEmptyValuesConfig config;
    config.getServerSettings().serverMode = ovms::HF_PULL_AND_START_MODE;
    auto retCode = server.startModules(config);
    // Empty sourceModel: takes task+model_path path, but model_path is empty
    // -> GraphExport::createServableConfig fails with PATH_INVALID
    EXPECT_TRUE(!retCode.ok()) << retCode.string();
    serverGuard = std::make_unique<ServerShutdownGuard>(server);
    EXPECT_TRUE(server.getModule(ovms::HF_MODEL_PULL_MODULE_NAME) != nullptr);
    ASSERT_EQ(server.getModule(ovms::HF_MODEL_PULL_MODULE_NAME)->getState(), ovms::ModuleState::INITIALIZED);
    ASSERT_NE(server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME), nullptr);  // expected to be started
    ASSERT_EQ(server.getModule(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME), nullptr);
}

// ===================== LoRA Pull Module Tests =====================

class TestHfPullModelModuleForLora : public ovms::HfPullModelModule {
public:
    ovms::HFSettingsImpl& getHfSettings() { return this->hfSettings; }
    ovms::Status testResolveHfLoraFilenames() { return this->resolveHfLoraFilenames(); }
    ovms::Status testPullLoraAdapters(const std::string& graphDirectory) { return this->pullLoraAdapters(graphDirectory); }
};

class HfPullModelModuleLoraTest : public TestWithTempDir {};

TEST_F(HfPullModelModuleLoraTest, ResolveHfLoraFilenames) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();
    const char* hfToken = std::getenv("HF_TOKEN");
    if (!hfToken || std::string(hfToken).empty()) {
        GTEST_SKIP() << "Skipping: HF_TOKEN not set (required for HF API resolution)";
    }
    TestHfPullModelModuleForLora module;
    auto& settings = module.getHfSettings();
    settings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl graphSettings;
    ovms::LoraAdapterSettings adapter;
    adapter.alias = "pokemon";
    adapter.sourceLora = "juliensimon/sd-pokemon-lora";
    adapter.sourceType = ovms::LoraSourceType::HF_REPO;
    graphSettings.loraAdapters.push_back(adapter);
    settings.graphSettings = graphSettings;

    auto status = module.testResolveHfLoraFilenames();
    ASSERT_TRUE(status.ok()) << status.string();

    const auto& resolved = std::get<ovms::ImageGenerationGraphSettingsImpl>(settings.graphSettings);
    ASSERT_EQ(resolved.loraAdapters.size(), 1);
    EXPECT_FALSE(resolved.loraAdapters[0].safetensorsFile.has_value());
    EXPECT_EQ(resolved.loraAdapters[0].resolvedSafetensorsFile.value(), "pytorch_lora_weights.safetensors");
    EXPECT_EQ(resolved.loraAdapters[0].effectiveSafetensorsFile().value(), "pytorch_lora_weights.safetensors");
}

TEST_F(HfPullModelModuleLoraTest, PullLoraAdaptersFromHfRepo) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();
    const char* hfToken = std::getenv("HF_TOKEN");
    if (!hfToken || std::string(hfToken).empty()) {
        GTEST_SKIP() << "Skipping: HF_TOKEN not set (required for HF download)";
    }
    TestHfPullModelModuleForLora module;
    auto& settings = module.getHfSettings();
    settings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl graphSettings;
    ovms::LoraAdapterSettings adapter;
    adapter.alias = "pokemon";
    adapter.sourceLora = "juliensimon/sd-pokemon-lora";
    adapter.safetensorsFile = "pytorch_lora_weights.safetensors";  // explicit filename — skips HF API resolve
    adapter.sourceType = ovms::LoraSourceType::HF_REPO;
    graphSettings.loraAdapters.push_back(adapter);
    settings.graphSettings = graphSettings;

    auto status = module.testPullLoraAdapters(this->directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();

    auto loraFilePath = ovms::FileSystem::joinPath({this->directoryPath, "loras", "juliensimon/sd-pokemon-lora", "pytorch_lora_weights.safetensors"});
    ASSERT_TRUE(std::filesystem::exists(loraFilePath)) << loraFilePath;
    EXPECT_GT(std::filesystem::file_size(loraFilePath), 0);
}

TEST_F(HfPullModelModuleLoraTest, PullLoraAdaptersSkipsLocalFile) {
    TestHfPullModelModuleForLora module;
    auto& settings = module.getHfSettings();
    settings.task = ovms::IMAGE_GENERATION_GRAPH;
    ovms::ImageGenerationGraphSettingsImpl graphSettings;
    ovms::LoraAdapterSettings adapter;
    adapter.alias = "local_lora";
    adapter.sourceLora = "/some/local/path/model.safetensors";
    adapter.safetensorsFile = "model.safetensors";
    adapter.sourceType = ovms::LoraSourceType::LOCAL_FILE;
    graphSettings.loraAdapters.push_back(adapter);
    settings.graphSettings = graphSettings;

    auto status = module.testPullLoraAdapters(this->directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
    // No files should have been downloaded to the temp directory
    EXPECT_TRUE(std::filesystem::is_empty(this->directoryPath));
}

TEST_F(HfPullModelModuleLoraTest, PullLoraAdaptersNonImageGenGraphIsNoOp) {
    TestHfPullModelModuleForLora module;
    auto& settings = module.getHfSettings();
    settings.task = ovms::TEXT_GENERATION_GRAPH;
    settings.graphSettings = ovms::TextGenGraphSettingsImpl{};

    auto status = module.testPullLoraAdapters(this->directoryPath);
    ASSERT_TRUE(status.ok()) << status.string();
}

class HfDownloaderPullHfModel : public HfPull {};

// Full-flow test: download SD model + LoRA via --pull mode, verify files and graph.pbtxt.
// This exercises: CLI parsing -> source_loras -> HF resolution -> LoRA download -> graph.pbtxt generation.
// Runtime clone()+LoRA behavior is guaranteed by the GenAI API: clone() "reuses underlying models"
// which share the AdapterController. Adapters are selected per-request via generate() properties.
TEST_F(HfDownloaderPullHfModel, DownloadImageGenModelWithLoRA) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();
    const char* hfToken = std::getenv("HF_TOKEN");
    if (!hfToken || std::string(hfToken).empty()) {
        GTEST_SKIP() << "Skipping: HF_TOKEN not set (required for HF LoRA download)";
    }
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    std::string modelName = "OpenVINO/stable-diffusion-v1-5-int8-ov";
    std::string downloadPath = ovms::FileSystem::joinPath({this->directoryPath, "repository"});
    std::string task = "image_generation";
    std::string sourceLoras = "pokemon=juliensimon/sd-pokemon-lora@pytorch_lora_weights.safetensors";
    ::SetUpServerForDownloadWithLoras(this->t, this->server, modelName, downloadPath, task, sourceLoras);

    std::string basePath = ovms::FileSystem::joinPath({downloadPath, "OpenVINO", "stable-diffusion-v1-5-int8-ov"});
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    // Verify model was downloaded
    ASSERT_TRUE(std::filesystem::exists(basePath)) << basePath;
    ASSERT_TRUE(std::filesystem::exists(graphPath)) << graphPath;

    // Verify LoRA adapter was downloaded
    std::string loraDir = ovms::FileSystem::joinPath({basePath, "loras", "juliensimon", "sd-pokemon-lora"});
    auto loraFiles = searchFilesRecursively(loraDir, {"pytorch_lora_weights.safetensors"});
    ASSERT_FALSE(loraFiles.empty()) << "LoRA .safetensors not found in: " << loraDir;

    // Verify graph.pbtxt contains the LoRA adapter entry
    std::string graphContents = GetFileContents(graphPath);
    EXPECT_NE(graphContents.find("lora_adapters"), std::string::npos) << "graph.pbtxt should contain lora_adapters";
    EXPECT_NE(graphContents.find("pokemon"), std::string::npos) << "graph.pbtxt should reference pokemon alias";
}

// ===================== Full Image Generation with Pull + LoRA Integration Test =====================
// Single test that:
//   1. Pulls SDXL-int8 model from HuggingFace + 2 LoRA adapters from direct HF URLs
//   2. Verifies downloaded files and graph.pbtxt
//   3. Starts serving from the pulled directory (second server launch — no re-download)
//   4. Makes REST requests: base model, individual LoRA, composite LoRA
//   5. Saves generated images to disk for manual inspection
//
// Model directory persists at: /tmp/ovms_test_sdxl_lora/
// Output images saved to:     /tmp/ovms_test_sdxl_lora_output/
//
// LoRA adapters (all SDXL-compatible, from openvino_notebooks/multilora-image-generation):
//   - xray: DoctorDiffusion/doctor-diffusion-s-xray-xl-lora / DD-xray-v1.safetensors (weight 0.8)
//   - chalkboard: Norod78/sdxl-chalkboarddrawing-lora / SDXL_ChalkBoardDrawing_LoRA_r8.safetensors (weight 0.45)
//   - combo: composite of @xray:0.8+@chalkboard:0.45
//
// Additional LoRAs available (commented out, can be swapped in):
//   - point: alvdansen/the-point / araminta_k_the_point.safetensors (weight 0.6)
//   - ukiyoe: KappaNeuro/ukiyo-e-art / Ukiyo-e Art.safetensors (weight 0.8)
//   - vector: DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora / DD-vector-v2.safetensors (weight 0.8)
//
// Manual reproduction (run inside docker container):
//   # Pull:
//   ./bazel-bin/src/ovms --pull --source_model OpenVINO/stable-diffusion-xl-base-1.0-int8-ov --model_repository_path /tmp/ovms_test_sdxl_lora --task image_generation --source_loras "xray=https://huggingface.co/DoctorDiffusion/doctor-diffusion-s-xray-xl-lora/resolve/main/DD-xray-v1.safetensors,chalkboard=https://huggingface.co/Norod78/sdxl-chalkboarddrawing-lora/resolve/main/SDXL_ChalkBoardDrawing_LoRA_r8.safetensors,combo=@xray:0.8+@chalkboard:0.45"
//
//   # Serve:
//   ./bazel-bin/src/ovms --source_model OpenVINO/stable-diffusion-xl-base-1.0-int8-ov --model_repository_path /tmp/ovms_test_sdxl_lora --task image_generation --source_loras "xray=/tmp/ovms_test_sdxl_lora/OpenVINO/stable-diffusion-xl-base-1.0-int8-ov/loras/xray/DD-xray-v1.safetensors,chalkboard=/tmp/ovms_test_sdxl_lora/OpenVINO/stable-diffusion-xl-base-1.0-int8-ov/loras/chalkboard/SDXL_ChalkBoardDrawing_LoRA_r8.safetensors,combo=@xray:0.8+@chalkboard:0.45" --rest_port 8080
//
//   # Generate (curl):
//   curl -s http://localhost:8080/v3/images/generations -H "Content-Type: application/json" -d '{"model":"xray","prompt":"xray a castle on a hill","size":"256x256","num_inference_steps":4}' | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('/tmp/xray.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
//   curl -s http://localhost:8080/v3/images/generations -H "Content-Type: application/json" -d '{"model":"chalkboard","prompt":"A colorful chalkboard drawing of a castle","size":"256x256","num_inference_steps":4}' | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('/tmp/chalkboard.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
//   curl -s http://localhost:8080/v3/images/generations -H "Content-Type: application/json" -d '{"model":"combo","prompt":"xray chalkboard castle","size":"256x256","num_inference_steps":4}' | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('/tmp/combo.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
#ifndef _WIN32

// LoRA direct download URLs
static const std::string LORA_XRAY_URL = "https://huggingface.co/DoctorDiffusion/doctor-diffusion-s-xray-xl-lora/resolve/main/DD-xray-v1.safetensors";
static const std::string LORA_CHALKBOARD_URL = "https://huggingface.co/Norod78/sdxl-chalkboarddrawing-lora/resolve/main/SDXL_ChalkBoardDrawing_LoRA_r8.safetensors";
// static const std::string LORA_POINT_URL = "https://huggingface.co/alvdansen/the-point/resolve/main/araminta_k_the_point.safetensors";
// static const std::string LORA_UKIYOE_URL = "https://huggingface.co/KappaNeuro/ukiyo-e-art/resolve/main/Ukiyo-e%20Art.safetensors";
// static const std::string LORA_VECTOR_URL = "https://huggingface.co/DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora/resolve/main/DD-vector-v2.safetensors";

static const std::string SDXL_MODEL_NAME = "OpenVINO/stable-diffusion-xl-base-1.0-int8-ov";
static const std::string SDXL_DOWNLOAD_PATH = "/tmp/ovms_test_sdxl_lora";
static const std::string SDXL_OUTPUT_PATH = "/tmp/ovms_test_sdxl_lora_output";

// Helper: extract b64_json from response body and save as PNG file
static void saveGeneratedImage(const std::string& responseBody, const std::string& outputPath) {
    // Find b64_json value in JSON response
    std::string marker = "\"b64_json\":\"";
    auto pos = responseBody.find(marker);
    if (pos == std::string::npos)
        return;
    pos += marker.size();
    auto endPos = responseBody.find("\"", pos);
    if (endPos == std::string::npos)
        return;
    std::string b64 = responseBody.substr(pos, endPos - pos);

    // Decode base64
    std::string decoded;
    if (!absl::Base64Unescape(b64, &decoded)) {
        std::cerr << "Failed to decode base64 image" << std::endl;
        return;
    }

    std::ofstream out(outputPath, std::ios::binary);
    out.write(decoded.data(), decoded.size());
    std::cout << "Saved generated image (" << decoded.size() << " bytes) to: " << outputPath << std::endl;
}

TEST(HfPullImageGenWithLora, PullServeAndGenerateWithLoras) {
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();

    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;
    std::string downloadPath = SDXL_DOWNLOAD_PATH;
    std::string modelName = SDXL_MODEL_NAME;
    std::string task = "image_generation";

    // Prepare output directory for generated images
    std::filesystem::create_directories(SDXL_OUTPUT_PATH);

    // ==================== PART 1: Pull model + LoRAs ====================
    std::string sourceLoras =
        "xray=" + LORA_XRAY_URL + ","
                                  "chalkboard=" +
        LORA_CHALKBOARD_URL + ","
                              "combo=@xray:0.8+@chalkboard:0.45";
    // Alternative LoRAs (swap in as needed):
    // "point=" + LORA_POINT_URL + ","
    // "ukiyoe=" + LORA_UKIYOE_URL + ","
    // "vector=" + LORA_VECTOR_URL + ","

    ::SetUpServerForDownloadWithLoras(t, server, modelName, downloadPath, task, sourceLoras,
        EXIT_SUCCESS, 8 * SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS);

    // Server exits after pull — join and reset
    server.setShutdownRequest(1);
    t->join();
    t.reset();
    server.setShutdownRequest(0);

    // Verify model was downloaded
    std::string modelBasePath = ovms::FileSystem::joinPath({downloadPath, "OpenVINO", "stable-diffusion-xl-base-1.0-int8-ov"});
    ASSERT_TRUE(std::filesystem::exists(modelBasePath)) << "Model not downloaded to: " << modelBasePath;

    std::string graphPath = ovms::FileSystem::appendSlash(modelBasePath) + "graph.pbtxt";
    ASSERT_TRUE(std::filesystem::exists(graphPath)) << "graph.pbtxt not found: " << graphPath;

    // Verify graph.pbtxt references all LoRA aliases
    std::string graphContents = GetFileContents(graphPath);
    EXPECT_NE(graphContents.find("lora_adapters"), std::string::npos) << "graph.pbtxt should contain lora_adapters";
    EXPECT_NE(graphContents.find("xray"), std::string::npos) << "graph.pbtxt should reference xray alias";
    EXPECT_NE(graphContents.find("chalkboard"), std::string::npos) << "graph.pbtxt should reference chalkboard alias";
    EXPECT_NE(graphContents.find("combo"), std::string::npos) << "graph.pbtxt should reference combo composite alias";

    // Verify LoRA files were downloaded
    std::string lorasDir = ovms::FileSystem::joinPath({modelBasePath, "loras"});
    std::string xrayLoraPath = ovms::FileSystem::joinPath({lorasDir, "xray", "DD-xray-v1.safetensors"});
    std::string chalkboardLoraPath = ovms::FileSystem::joinPath({lorasDir, "chalkboard", "SDXL_ChalkBoardDrawing_LoRA_r8.safetensors"});
    ASSERT_TRUE(std::filesystem::exists(xrayLoraPath)) << "X-ray LoRA not found at: " << xrayLoraPath;
    ASSERT_TRUE(std::filesystem::exists(chalkboardLoraPath)) << "Chalkboard LoRA not found at: " << chalkboardLoraPath;

    std::cout << "=== PULL COMPLETE ===" << std::endl;
    std::cout << "Model path: " << modelBasePath << std::endl;
    std::cout << "Graph: " << graphPath << std::endl;
    std::cout << "X-ray LoRA: " << xrayLoraPath << std::endl;
    std::cout << "Chalkboard LoRA: " << chalkboardLoraPath << std::endl;

    // ==================== PART 2: Serve from pulled directory + generate ====================
    // Re-configure with local file paths for the second server launch
    std::string sourceLorasLocal =
        "xray=" + xrayLoraPath + ","
                                 "chalkboard=" +
        chalkboardLoraPath + ","
                             "combo=@xray:0.8+@chalkboard:0.45";

    std::string restPort = "9233";
    ::SetUpServerForDownloadAndStartWithLoras(t, server,
        modelName, downloadPath, task, sourceLorasLocal, restPort, 8 * SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS);

    std::cout << "=== SERVER STARTED === REST port: " << restPort << std::endl;

    auto cli = std::make_unique<httplib::Client>(std::string("http://localhost:") + restPort);
    cli->set_read_timeout(600);  // SDXL image generation is slow on CPU

    auto healthRes = cli->Get("/v2/health/live");
    ASSERT_TRUE(healthRes) << "Failed to reach server health endpoint";
    ASSERT_EQ(healthRes->status, 200) << "Server not healthy";

    // --- Generate: base model ---
    std::string baseRequestBody = R"({
        "model": ")" + SDXL_MODEL_NAME +
                                  R"(",
        "prompt": "a simple red circle on white background",
        "size": "256x256",
        "num_inference_steps": 4
    })";
    auto baseRes = cli->Post("/v3/images/generations", baseRequestBody, "application/json");
    ASSERT_TRUE(baseRes) << "Base model request failed";
    ASSERT_EQ(baseRes->status, 200) << "Base model failed: " << baseRes->status << " body: " << baseRes->body.substr(0, 500);
    EXPECT_NE(baseRes->body.find("\"b64_json\""), std::string::npos);
    saveGeneratedImage(baseRes->body, SDXL_OUTPUT_PATH + "/base_model.png");

    // --- Generate: X-ray LoRA ---
    std::string xrayRequestBody = R"({
        "model": "xray",
        "prompt": "xray a castle on a hill, detailed architecture",
        "size": "256x256",
        "num_inference_steps": 4
    })";
    auto xrayRes = cli->Post("/v3/images/generations", xrayRequestBody, "application/json");
    ASSERT_TRUE(xrayRes) << "X-ray LoRA request failed";
    ASSERT_EQ(xrayRes->status, 200) << "X-ray failed: " << xrayRes->status << " body: " << xrayRes->body.substr(0, 500);
    EXPECT_NE(xrayRes->body.find("\"b64_json\""), std::string::npos);
    saveGeneratedImage(xrayRes->body, SDXL_OUTPUT_PATH + "/xray_lora.png");

    // --- Generate: Chalkboard LoRA ---
    std::string chalkboardRequestBody = R"({
        "model": "chalkboard",
        "prompt": "A colorful chalkboard drawing of a castle on a hill",
        "size": "256x256",
        "num_inference_steps": 4
    })";
    auto chalkboardRes = cli->Post("/v3/images/generations", chalkboardRequestBody, "application/json");
    ASSERT_TRUE(chalkboardRes) << "Chalkboard LoRA request failed";
    ASSERT_EQ(chalkboardRes->status, 200) << "Chalkboard failed: " << chalkboardRes->status << " body: " << chalkboardRes->body.substr(0, 500);
    EXPECT_NE(chalkboardRes->body.find("\"b64_json\""), std::string::npos);
    saveGeneratedImage(chalkboardRes->body, SDXL_OUTPUT_PATH + "/chalkboard_lora.png");

    // --- Generate: Composite LoRA (combo = xray:0.8 + chalkboard:0.45) ---
    std::string comboRequestBody = R"({
        "model": "combo",
        "prompt": "xray A colorful chalkboard drawing of a castle on a hill, detailed architecture",
        "size": "256x256",
        "num_inference_steps": 4
    })";
    auto comboRes = cli->Post("/v3/images/generations", comboRequestBody, "application/json");
    ASSERT_TRUE(comboRes) << "Composite LoRA request failed";
    ASSERT_EQ(comboRes->status, 200) << "Composite failed: " << comboRes->status << " body: " << comboRes->body.substr(0, 500);
    EXPECT_NE(comboRes->body.find("\"b64_json\""), std::string::npos);
    saveGeneratedImage(comboRes->body, SDXL_OUTPUT_PATH + "/combo_lora.png");

    std::cout << "=== ALL IMAGES GENERATED ===" << std::endl;
    std::cout << "Output directory: " << SDXL_OUTPUT_PATH << std::endl;

    // Shutdown
    server.setShutdownRequest(1);
    t->join();
    t.reset();
    server.setShutdownRequest(0);
}
#endif  // !_WIN32
