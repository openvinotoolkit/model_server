//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <chrono>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../capi_frontend/server_settings.hpp"
#include "../utils/env_guard.hpp"
#include "src/filesystem/filesystem.hpp"
#include "src/pull_module/cmd_exec.hpp"
#include "src/pull_module/model_downloader.hpp"
#include "src/pull_module/oci_downloader.hpp"
#include "src/pull_module/optimum_export.hpp"
#include "platform_utils.hpp"
#include "test_utils.hpp"
#include "test_with_temp_dir.hpp"
#include "environment.hpp"

#include "../status.hpp"

using ovms::OciDownloader;
using ovms::StatusCode;
using testing::EndsWith;
using testing::HasSubstr;

// OptimumDownloader driven by the mock optimum-cli that records the export
// command it was about to run.
class MockOptimumConverter : public ovms::OptimumDownloader {
public:
    MockOptimumConverter(const ovms::ExportSettings& exportSettings, const ovms::GraphExportType& task,
        const std::string& sourceModel, const std::string& downloadPath, bool overwrite,
        const std::string& cliMock, std::string& recordedCmd) :
        OptimumDownloader(exportSettings, task, sourceModel, downloadPath, overwrite,
            cliMock + " export ", cliMock + " -h", cliMock + " export ", cliMock + " -h"),
        recordedCmd(recordedCmd) {}

    ovms::Status downloadModel() override {
        this->recordedCmd = this->getExportCmd();
        return OptimumDownloader::downloadModel();
    }

private:
    std::string& recordedCmd;
};

// Exposes the protected surface of OciDownloader so the individual steps can
// be asserted without running the whole download.
class TestOciDownloader : public OciDownloader {
public:
    explicit TestOciDownloader(const ovms::HFSettingsImpl& inHfSettings) :
        OciDownloader(inHfSettings.exportSettings, inHfSettings.task, inHfSettings.sourceModel,
            ovms::IModelDownloader::getGraphDirectory(inHfSettings.downloadPath, inHfSettings.sourceModel),
            inHfSettings.overwriteModels) {}

    std::string getVersionCmd() const { return OciDownloader::getVersionCmd(); }
    std::string getResolveCmd() const { return OciDownloader::getResolveCmd(); }
    ovms::Status checkLlmmanIsPresent() { return OciDownloader::checkLlmmanIsPresent(); }
    ovms::Status validateGraphDirectory() const { return OciDownloader::validateGraphDirectory(); }
    std::string getGraphDirectory() { return OciDownloader::getGraphDirectory(); }
    static ovms::Status parseResolveOutput(const std::string& output, std::string& outPath, std::string& outFormat) {
        return OciDownloader::parseResolveOutput(output, outPath, outFormat);
    }
    static bool containsOpenVinoIr(const std::string& directory) {
        return OciDownloader::containsOpenVinoIr(directory);
    }

    // When set, the safetensors conversion runs against this mock optimum-cli.
    std::string optimumMockPath;
    mutable std::string recordedExportCmd;

protected:
    std::unique_ptr<ovms::IModelDownloader> createConverter(const std::string& resolvedPath) const override {
        if (this->optimumMockPath.empty()) {
            return OciDownloader::createConverter(resolvedPath);
        }
        return std::make_unique<MockOptimumConverter>(this->exportSettings, this->task, resolvedPath,
            this->downloadPath, this->overwriteModels, this->optimumMockPath, this->recordedExportCmd);
    }
};

// ----------------------------------------------------------------------------
// oci:// scheme handling
// ----------------------------------------------------------------------------

TEST(OciSchemeTest, IsOciDownloadRequiresExplicitScheme) {
    EXPECT_TRUE(ovms::isOciDownload("oci://ghcr.io/org/model:tag"));
    EXPECT_TRUE(ovms::isOciDownload("OCI://ghcr.io/org/model:tag"));
    // A bare registry reference is indistinguishable from a HuggingFace repo
    // id, so it must keep going down the HuggingFace path.
    EXPECT_FALSE(ovms::isOciDownload("ghcr.io/org/model:tag"));
    EXPECT_FALSE(ovms::isOciDownload("OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov"));
    EXPECT_FALSE(ovms::isOciDownload("meta-llama/Llama-3-8B"));
    EXPECT_FALSE(ovms::isOciDownload(""));
}

TEST(OciSchemeTest, StripOciScheme) {
    EXPECT_EQ(ovms::stripOciScheme("oci://ghcr.io/org/model:tag"), "ghcr.io/org/model:tag");
    EXPECT_EQ(ovms::stripOciScheme("OCI://ghcr.io/org/model:tag"), "ghcr.io/org/model:tag");
    EXPECT_EQ(ovms::stripOciScheme("OpenVINO/Phi-3"), "OpenVINO/Phi-3");
}

TEST(OciSchemeTest, LocalModelDirectoryNameIsIdentityForHuggingFace) {
    EXPECT_EQ(ovms::localModelDirectoryName("OpenVINO/Phi-3"), "OpenVINO/Phi-3");
    EXPECT_EQ(ovms::localModelDirectoryName(""), "");
}

TEST(OciSchemeTest, LocalModelDirectoryNameDropsSchemeAndTagSeparator) {
    EXPECT_EQ(ovms::localModelDirectoryName("oci://ghcr.io/org/model:tag"), "ghcr.io/org/model+tag");
    EXPECT_EQ(ovms::localModelDirectoryName("oci://registry:5000/org/model:tag"), "registry+5000/org/model+tag");
    EXPECT_EQ(ovms::localModelDirectoryName("oci://ghcr.io/org/model"), "ghcr.io/org/model");
}

TEST(OciSchemeTest, GraphDirectoryUsesSanitizedName) {
    const std::string expected = ovms::FileSystem::joinPath({"/models", "ghcr.io/org/model+tag"});
    EXPECT_EQ(ovms::IModelDownloader::getGraphDirectory("/models", "oci://ghcr.io/org/model:tag"), expected);
}

TEST(OciSchemeTest, OciReferencesAreNotOptimumCliDownloads) {
    // Otherwise --weight-format would silently reroute an oci:// reference
    // into the optimum-cli-from-HuggingFace path.
    EXPECT_FALSE(ovms::isOptimumCliDownload("oci://ghcr.io/org/model:tag", std::nullopt));
    EXPECT_TRUE(ovms::isOptimumCliDownload("meta-llama/Llama-3-8B", std::nullopt));
}

// ----------------------------------------------------------------------------
// llmman command construction
// ----------------------------------------------------------------------------

class OciDownloaderCommands : public TestWithTempDir {
public:
    ovms::HFSettingsImpl hfSettings;
    EnvGuard pathGuard;
    void SetUp() override {
        TestWithTempDir::SetUp();
        pathGuard.set("PATH", this->directoryPath);
        hfSettings.sourceModel = "oci://ghcr.io/org/model:tag";
        hfSettings.downloadPath = "/models";
        hfSettings.task = ovms::TEXT_GENERATION_GRAPH;
        hfSettings.downloadType = ovms::OCI_DOWNLOAD;
    }
};

TEST_F(OciDownloaderCommands, ResolveCommandDropsTheScheme) {
    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.getResolveCmd(), "llmman resolve ghcr.io/org/model:tag");
    EXPECT_EQ(downloader.getVersionCmd(), "llmman --version");
}

TEST_F(OciDownloaderCommands, ReferenceWithSpacesIsQuoted) {
    hfSettings.sourceModel = "oci://ghcr.io/org/my model:tag";
    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.getVersionCmd(), "llmman --version");
    EXPECT_EQ(downloader.getResolveCmd(), "llmman resolve \"ghcr.io/org/my model:tag\"");
}

TEST_F(OciDownloaderCommands, GraphDirectoryIsSanitized) {
    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.getGraphDirectory(), ovms::FileSystem::joinPath({"/models", "ghcr.io/org/model+tag"}));
}

TEST_F(OciDownloaderCommands, MissingBinaryIsReported) {
    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.checkLlmmanIsPresent(), StatusCode::OCI_LLMMAN_NOT_FOUND);
}

TEST(CmdExecQuoteTest, QuotesOnlyWhenNeeded) {
    EXPECT_EQ(ovms::quote_cmd_arg("model/name"), "model/name");
    EXPECT_EQ(ovms::quote_cmd_arg(""), "");
    EXPECT_EQ(ovms::quote_cmd_arg("a b"), "\"a b\"");
    EXPECT_EQ(ovms::quote_cmd_arg("a\"b"), "\"a\\\"b\"");
}

// ----------------------------------------------------------------------------
// llmman resolve output parsing
// ----------------------------------------------------------------------------

TEST(OciResolveOutputTest, ParsesTheJsonLine) {
    std::string path;
    std::string format;
    ASSERT_EQ(TestOciDownloader::parseResolveOutput(
                  R"({"reference":"ghcr.io/org/model:tag","path":"/store/cache/abc","format":"safetensors"})",
                  path, format),
        StatusCode::OK);
    EXPECT_EQ(path, "/store/cache/abc");
    EXPECT_EQ(format, "safetensors");
}

TEST(OciResolveOutputTest, IgnoresDiagnosticsPrintedBeforeTheJson) {
    // exec_cmd() merges the child's stderr into the same buffer, so llmman's
    // progress output shows up interleaved with the machine-readable line.
    const std::string output =
        "[llmman] pulling ghcr.io/org/model:tag\n"
        "[llmman] using blob directly: sha256:deadbeef\n"
        R"({"path":"/store/cache/abc/model.gguf","format":"gguf"})"
        "\n";
    std::string path;
    std::string format;
    ASSERT_EQ(TestOciDownloader::parseResolveOutput(output, path, format), StatusCode::OK);
    EXPECT_EQ(path, "/store/cache/abc/model.gguf");
    EXPECT_EQ(format, "gguf");
}

TEST(OciResolveOutputTest, RejectsOutputWithoutJson) {
    std::string path;
    std::string format;
    EXPECT_EQ(TestOciDownloader::parseResolveOutput("", path, format), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
    EXPECT_EQ(TestOciDownloader::parseResolveOutput("not json at all", path, format), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
    EXPECT_EQ(TestOciDownloader::parseResolveOutput("[1, 2, 3]", path, format), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

TEST(OciResolveOutputTest, RejectsJsonWithoutRequiredMembers) {
    std::string path;
    std::string format;
    EXPECT_EQ(TestOciDownloader::parseResolveOutput(R"({"path":"/store/cache/abc"})", path, format), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
    EXPECT_EQ(TestOciDownloader::parseResolveOutput(R"({"format":"gguf"})", path, format), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
    EXPECT_EQ(TestOciDownloader::parseResolveOutput(R"({"path":42,"format":"gguf"})", path, format), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

// ----------------------------------------------------------------------------
// Payload classification
// ----------------------------------------------------------------------------

class OciDownloaderPayload : public TestWithTempDir {
public:
    std::string llmmanMockPath;
    std::string optimumMockPath;
    std::string resolvedPath;
    ovms::HFSettingsImpl hfSettings;
    EnvGuard pathGuard;

    void SetUp() override {
        TestWithTempDir::SetUp();
#ifdef _WIN32
        llmmanMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/llmman.exe");
        optimumMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/optimum-cli.exe");
#else
        llmmanMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/llmman");
        optimumMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/optimum-cli");
#endif
        const std::string pathSeparator =
#ifdef _WIN32
            ";";
#else
            ":";
#endif
        pathGuard.set("PATH", std::filesystem::path(llmmanMockPath).parent_path().string() + pathSeparator + GetEnvVar("PATH"));
        resolvedPath = std::filesystem::path(this->directoryPath).append("llmman-store").generic_string();
        std::filesystem::create_directories(resolvedPath);

        hfSettings.sourceModel = "oci://ghcr.io/org/model:tag";
        hfSettings.downloadPath = std::filesystem::path(this->directoryPath).append("repository").generic_string();
        hfSettings.task = ovms::TEXT_GENERATION_GRAPH;
        hfSettings.downloadType = ovms::OCI_DOWNLOAD;
    }

    void createFile(const std::string& directory, const std::string& name, const std::string& contents = "x") {
        std::ofstream stream(std::filesystem::path(directory).append(name));
        stream << contents;
    }
};

TEST_F(OciDownloaderPayload, ContainsOpenVinoIrNeedsBothXmlAndBin) {
    EXPECT_FALSE(TestOciDownloader::containsOpenVinoIr(resolvedPath));
    createFile(resolvedPath, "openvino_model.xml");
    EXPECT_FALSE(TestOciDownloader::containsOpenVinoIr(resolvedPath));
    createFile(resolvedPath, "openvino_model.bin");
    EXPECT_TRUE(TestOciDownloader::containsOpenVinoIr(resolvedPath));
}

TEST_F(OciDownloaderPayload, ContainsOpenVinoIrIgnoresAuxiliaryTokenizerPair) {
    createFile(resolvedPath, "openvino_tokenizer.xml");
    createFile(resolvedPath, "openvino_tokenizer.bin");
    EXPECT_FALSE(TestOciDownloader::containsOpenVinoIr(resolvedPath));
}

TEST_F(OciDownloaderPayload, ContainsOpenVinoIrIsFalseForMissingDirectory) {
    EXPECT_FALSE(TestOciDownloader::containsOpenVinoIr(std::filesystem::path(this->directoryPath).append("nope").generic_string()));
}

TEST_F(OciDownloaderPayload, OpenVinoIrModelIsServedFromTheLlmmanStore) {
    createFile(resolvedPath, "openvino_model.xml");
    createFile(resolvedPath, "openvino_model.bin");
    createFile(resolvedPath, "config.json", "{}");

    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", resolvedPath);
    guard.set("LLMMAN_MOCK_FORMAT", "safetensors");
    guard.set("LLMMAN_MOCK_NOISE", "1");

    TestOciDownloader downloader(hfSettings);
    ASSERT_EQ(downloader.downloadModel(), StatusCode::OK);
    // No second copy of the weights: graph.pbtxt just points at llmman's store.
    EXPECT_EQ(downloader.getModelPath(), resolvedPath);
    EXPECT_FALSE(downloader.getGgufFilename().has_value());
    // The graph directory still has to exist, that is where graph.pbtxt goes.
    EXPECT_TRUE(std::filesystem::is_directory(downloader.getGraphDirectory()));
}

TEST_F(OciDownloaderPayload, QuotedArgumentReachesTheProcessAsOneArgument) {
    // The mock optimum-cli echoes every argv entry it receives.
    const std::string tricky = "a  b \"c\" 'd' \\e";
    int retCode = -1;
    const std::string output = ovms::exec_cmd(optimumMockPath + " " + ovms::quote_cmd_arg(tricky), retCode);
    EXPECT_EQ(retCode, 0);
    EXPECT_THAT(output, HasSubstr("Number of arguments: 2"));
    EXPECT_THAT(output, HasSubstr("Argument 1: " + tricky));
}

TEST_F(OciDownloaderPayload, SafetensorsCheckoutIsConvertedIntoTheGraphDirectory) {
    createFile(resolvedPath, "config.json", "{}");
    createFile(resolvedPath, "model.safetensors");

    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", resolvedPath);
    guard.set("LLMMAN_MOCK_FORMAT", "safetensors");

    TestOciDownloader downloader(hfSettings);
    downloader.optimumMockPath = optimumMockPath;
    ASSERT_EQ(downloader.downloadModel(), StatusCode::OK);
    // The checkout is the export source and the graph directory the target,
    // so graph.pbtxt keeps the default models_path.
    EXPECT_THAT(downloader.recordedExportCmd, HasSubstr("--model " + resolvedPath + " "));
    EXPECT_THAT(downloader.recordedExportCmd, EndsWith(downloader.getGraphDirectory()));
    EXPECT_EQ(downloader.getModelPath(), "./");
    EXPECT_FALSE(downloader.getGgufFilename().has_value());
    EXPECT_TRUE(std::filesystem::is_directory(downloader.getGraphDirectory()));
}

TEST_F(OciDownloaderPayload, SafetensorsConversionFailureIsPropagated) {
    createFile(resolvedPath, "config.json", "{}");
    createFile(resolvedPath, "model.safetensors");

    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", resolvedPath);
    guard.set("LLMMAN_MOCK_FORMAT", "safetensors");

    TestOciDownloader downloader(hfSettings);
    downloader.optimumMockPath = "NonExistingCommand33";
    EXPECT_EQ(downloader.downloadModel(), StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI);
}

TEST_F(OciDownloaderPayload, GgufModelIsSplitIntoDirectoryAndFilename) {
    createFile(resolvedPath, "model-Q4_K_M.gguf");
    const std::string ggufPath = std::filesystem::path(resolvedPath).append("model-Q4_K_M.gguf").generic_string();

    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", ggufPath);
    guard.set("LLMMAN_MOCK_FORMAT", "gguf");

    TestOciDownloader downloader(hfSettings);
    ASSERT_EQ(downloader.downloadModel(), StatusCode::OK);
    // The graph exporter joins these two back together into models_path.
    EXPECT_EQ(std::filesystem::path(downloader.getModelPath()).generic_string(), resolvedPath);
    ASSERT_TRUE(downloader.getGgufFilename().has_value());
    EXPECT_EQ(downloader.getGgufFilename().value(), "model-Q4_K_M.gguf");

    downloader.onDownloadComplete(hfSettings);
    EXPECT_EQ(hfSettings.exportSettings.modelPath, downloader.getModelPath());
    EXPECT_EQ(hfSettings.ggufFilename, downloader.getGgufFilename());
}

TEST_F(OciDownloaderPayload, ContentAddressedGgufBlobGetsUsableExtensionWithoutCopying) {
    const std::string blobPath = std::filesystem::path(resolvedPath).append("sha256-digest").generic_string();
    createFile(resolvedPath, "sha256-digest", "raw GGUF bytes");

    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", blobPath);
    guard.set("LLMMAN_MOCK_FORMAT", "gguf");

    TestOciDownloader downloader(hfSettings);
    ASSERT_EQ(downloader.downloadModel(), StatusCode::OK);
    ASSERT_EQ(downloader.getModelPath(), downloader.getGraphDirectory());
    ASSERT_TRUE(downloader.getGgufFilename().has_value());
    EXPECT_EQ(downloader.getGgufFilename().value(), "sha256-digest.gguf");

    const std::filesystem::path modelFile = std::filesystem::path(downloader.getGraphDirectory()) / downloader.getGgufFilename().value();
    ASSERT_TRUE(std::filesystem::is_regular_file(modelFile));
    EXPECT_EQ(std::filesystem::file_size(modelFile), std::filesystem::file_size(blobPath));
}

TEST_F(OciDownloaderPayload, GgufResolvedToDirectoryIsRejected) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", resolvedPath);
    guard.set("LLMMAN_MOCK_FORMAT", "gguf");

    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

TEST_F(OciDownloaderPayload, GgufPayloadRejectsNonTextGenerationTasks) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", std::filesystem::path(resolvedPath).append("model.gguf").generic_string());
    guard.set("LLMMAN_MOCK_FORMAT", "gguf");
    createFile(resolvedPath, "model.gguf");

    hfSettings.task = ovms::EMBEDDINGS_GRAPH;
    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_UNSUPPORTED_MODEL_FORMAT);
}

TEST_F(OciDownloaderPayload, SafetensorsResolvedToFileIsRejected) {
    createFile(resolvedPath, "model.safetensors");
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", std::filesystem::path(resolvedPath).append("model.safetensors").generic_string());
    guard.set("LLMMAN_MOCK_FORMAT", "safetensors");

    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

TEST_F(OciDownloaderPayload, UnsupportedFormatIsRejected) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", resolvedPath);
    guard.set("LLMMAN_MOCK_FORMAT", "onnx");

    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_UNSUPPORTED_MODEL_FORMAT);
}

TEST_F(OciDownloaderPayload, UnparseableResolveOutputIsRejected) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_OUTPUT", "this is not the JSON you are looking for");

    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

TEST_F(OciDownloaderPayload, NonExistentResolvedPathIsRejected) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", std::filesystem::path(this->directoryPath).append("gone").generic_string());
    guard.set("LLMMAN_MOCK_FORMAT", "safetensors");

    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

TEST_F(OciDownloaderPayload, EscapedDownloadPathIsRejected) {
    hfSettings.downloadPath = "../some/path";
    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::PATH_INVALID);
}

TEST_F(OciDownloaderPayload, SymlinkedRepositoryPathIsRejectedBeforeRunningLlmman) {
    const std::string externalDirectory = std::filesystem::path(this->directoryPath).append("external-repository").string();
    ASSERT_TRUE(std::filesystem::create_directories(externalDirectory));
    const std::string symlinkPath = std::filesystem::path(this->directoryPath).append("repository-link").string();
    std::error_code ec;
    std::filesystem::create_directory_symlink(externalDirectory, symlinkPath, ec);
    if (ec) {
        GTEST_SKIP() << "Could not create directory symlink on this platform: " << ec.message();
    }
    hfSettings.downloadPath = symlinkPath;

    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::PATH_INVALID);
    EXPECT_TRUE(std::filesystem::is_empty(externalDirectory));
}

TEST_F(OciDownloaderPayload, GraphDirectoryIsRevalidatedAfterItBecomesASymlink) {
    const std::string externalDirectory = std::filesystem::path(this->directoryPath).append("external-repository").string();
    ASSERT_TRUE(std::filesystem::create_directories(externalDirectory));
    hfSettings.downloadPath = std::filesystem::path(this->directoryPath).append("repository").string();
    TestOciDownloader downloader(hfSettings);

    ASSERT_EQ(downloader.validateGraphDirectory(), StatusCode::OK);
    std::filesystem::remove(hfSettings.downloadPath);
    std::error_code ec;
    std::filesystem::create_directory_symlink(externalDirectory, hfSettings.downloadPath, ec);
    if (ec) {
        GTEST_SKIP() << "Could not create directory symlink on this platform: " << ec.message();
    }

    EXPECT_EQ(downloader.validateGraphDirectory(), StatusCode::PATH_INVALID);
}

class OciModelPackServerProcess {
public:
    ~OciModelPackServerProcess() {
        this->stop();
    }

    bool start(const std::string& executable, const std::vector<std::string>& arguments) {
#ifdef _WIN32
        std::string commandLine = ovms::quote_cmd_arg(executable);
        for (const auto& argument : arguments) {
            commandLine += " " + ovms::quote_cmd_arg(argument);
        }
        STARTUPINFOA startupInfo{};
        startupInfo.cb = sizeof(startupInfo);
        if (!CreateProcessA(nullptr, commandLine.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                nullptr, nullptr, &startupInfo, &processInfo)) {
            this->lastError = "CreateProcessA failed with error " + std::to_string(GetLastError());
            return false;
        }
        return true;
#else
        processId = fork();
        if (processId < 0) {
            this->lastError = "fork failed";
            return false;
        }
        if (processId == 0) {
            std::vector<std::string> commandArguments;
            commandArguments.reserve(arguments.size() + 1);
            commandArguments.push_back(executable);
            commandArguments.insert(commandArguments.end(), arguments.begin(), arguments.end());
            std::vector<char*> argv;
            argv.reserve(commandArguments.size() + 1);
            for (auto& argument : commandArguments) {
                argv.push_back(argument.data());
            }
            argv.push_back(nullptr);
            execv(executable.c_str(), argv.data());
            _exit(127);
        }
        return true;
#endif
    }

    const std::string& getLastError() const { return this->lastError; }

    void stop() {
#ifdef _WIN32
        if (processInfo.hProcess != nullptr) {
            DWORD exitCode = 0;
            if (GetExitCodeProcess(processInfo.hProcess, &exitCode) && exitCode == STILL_ACTIVE) {
                TerminateProcess(processInfo.hProcess, 0);
                WaitForSingleObject(processInfo.hProcess, 30000);
            }
            CloseHandle(processInfo.hThread);
            CloseHandle(processInfo.hProcess);
            processInfo = {};
        }
#else
        if (processId > 0) {
            kill(processId, SIGTERM);
            int status = 0;
            while (waitpid(processId, &status, 0) < 0 && errno == EINTR) {
            }
            processId = -1;
        }
#endif
    }

private:
    std::string lastError;
#ifdef _WIN32
    PROCESS_INFORMATION processInfo{};
#else
    pid_t processId = -1;
#endif
};

class OciModelPackInferenceTest : public TestWithTempDir {
public:
    // Immutable Docker Hub manifest digest: updates to the mutable 0.6b tag
    // cannot silently change the code or weights exercised by CI.
    const std::string modelReference = "docker.io/ai/qwen3@sha256:34d2ca5e0ab03487bd1883013f2aca671fc5440dfa6dbacf9e439ef677ca0626";
    OciModelPackServerProcess ovmsProcess;
    EnvGuard llmmanStoreGuard;
    std::string llmmanStorePath;
    bool modelPullAttempted = false;

    void TearDown() override {
        this->ovmsProcess.stop();
        if (modelPullAttempted) {
            int retCode = -1;
            const std::string output = ovms::exec_cmd("llmman rm " + ovms::quote_cmd_arg(modelReference), retCode);
            EXPECT_EQ(retCode, 0) << "Failed to remove test OCI model from llmman's store: " << output;
        }
        TestWithTempDir::TearDown();
    }
};

TEST_F(OciModelPackInferenceTest, PullSmallPublicModelAndRunInference) {
    std::string sourceModel = "oci://" + modelReference;
    std::string repositoryPath = std::filesystem::path(this->directoryPath).append("repository").string();
    llmmanStorePath = std::filesystem::path(this->directoryPath).append("llmman-store").string();
    llmmanStoreGuard.set("LLMMAN_MODELS", llmmanStorePath);
    std::string task = "text_generation";
    std::string restPort = "9233";
    std::string serverPort = "9133";
    const std::string modelPath = ovms::IModelDownloader::getGraphDirectory(repositoryPath, sourceModel);
    randomizeAndEnsureFrees(serverPort, restPort);
    const std::string ovmsExecutable = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/ovms");
#ifdef _WIN32
    const std::string ovmsExecutableWithExtension = ovmsExecutable + ".exe";
#else
    const std::string& ovmsExecutableWithExtension = ovmsExecutable;
#endif
    // ovms_test sets these to require in-process MediaPipe/Python symbols.
    // The standalone ovms binary must load its runtime shared libraries like
    // a production process instead.
    EnvGuard testRuntimeGuard;
    testRuntimeGuard.unset("OVMS_TEST_PYTHON_CALCULATORS_INPROCESS");
    testRuntimeGuard.unset("OVMS_TEST_MEDIAPIPE_RUNTIME_INPROCESS");
#ifdef _WIN32
    const std::string llmmanInstallDir = "C:\\opt";
    const std::string pathSeparator = ";";
#else
    const std::string llmmanInstallDir = "/usr/local/bin";
    const std::string pathSeparator = ":";
#endif
    EnvGuard llmmanPathGuard;
    llmmanPathGuard.set("PATH", llmmanInstallDir + pathSeparator + GetEnvVar("PATH"));
    int llmmanVersionResult = -1;
    const std::string llmmanVersion = ovms::exec_cmd("llmman --version", llmmanVersionResult);
    ASSERT_EQ(llmmanVersionResult, 0) << "llmman is not available to the standalone OVMS process: " << llmmanVersion;

    modelPullAttempted = true;
    const std::string pullCommand = ovms::quote_cmd_arg(ovmsExecutableWithExtension) +
                                    " --pull --source_model " + ovms::quote_cmd_arg(sourceModel) +
                                    " --model_repository_path " + ovms::quote_cmd_arg(repositoryPath) +
                                    " --task " + ovms::quote_cmd_arg(task);
    int pullResult = -1;
    const std::string pullOutput = ovms::exec_cmd(pullCommand, pullResult);
    ASSERT_EQ(pullResult, 0) << "OVMS failed to pull the OCI ModelPack: " << pullOutput;

    ASSERT_TRUE(this->ovmsProcess.start(ovmsExecutableWithExtension,
        {"--port", serverPort, "--rest_port", restPort, "--model_name", "qwen", "--model_path", modelPath}))
        << this->ovmsProcess.getLastError();

    EnvGuard proxyGuard;
    proxyGuard.unset("http_proxy");
    proxyGuard.unset("https_proxy");
    proxyGuard.unset("HTTP_PROXY");
    proxyGuard.unset("HTTPS_PROXY");
    proxyGuard.unset("ALL_PROXY");
    proxyGuard.unset("all_proxy");

    bool httpReady = false;
    std::string lastHealthOutput;
    for (int attempt = 0; attempt < 30; ++attempt) {
        int healthCode = -1;
        lastHealthOutput = ovms::exec_cmd("curl --noproxy " + ovms::quote_cmd_arg("*") + " --silent --show-error --fail http://127.0.0.1:" + restPort + "/v2/models/qwen/ready", healthCode);
        if (healthCode == 0) {
            httpReady = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    ASSERT_TRUE(httpReady) << "OVMS model qwen did not become ready on port " << restPort << ": " << lastHealthOutput;

    const std::string requestBody = R"({
        "model": "qwen",
        "stream": false,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Reply with one word: hello"}]
    })";

    int inferenceCode = -1;
    const std::string response = ovms::exec_cmd("curl --noproxy " + ovms::quote_cmd_arg("*") + " --silent --show-error --write-out " +
                                                    ovms::quote_cmd_arg("\\nHTTP_STATUS:%{http_code}") + " --request POST http://127.0.0.1:" + restPort +
                                                    "/v3/chat/completions --header " + ovms::quote_cmd_arg("Content-Type: application/json") + " --data " +
                                                    ovms::quote_cmd_arg(requestBody),
        inferenceCode);
    ASSERT_EQ(inferenceCode, 0) << "curl failed during OCI ModelPack inference: " << response;
    EXPECT_THAT(response, EndsWith("HTTP_STATUS:200"));
    EXPECT_THAT(response, HasSubstr("\"choices\""));
    EXPECT_THAT(response, HasSubstr("\"content\""));
}
