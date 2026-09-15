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
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../capi_frontend/server_settings.hpp"
#include "../utils/env_guard.hpp"
#include "src/filesystem/filesystem.hpp"
#include "src/pull_module/model_downloader.hpp"
#include "src/pull_module/oci_downloader.hpp"
#include "platform_utils.hpp"
#include "test_utils.hpp"
#include "test_with_temp_dir.hpp"

#include "../status.hpp"

using ovms::OciDownloader;
using ovms::StatusCode;

// Exposes the protected surface of OciDownloader so the individual steps can
// be asserted without running the whole download.
class TestOciDownloader : public OciDownloader {
public:
    TestOciDownloader(const ovms::HFSettingsImpl& inHfSettings, const std::string& llmmanBinary = "") :
        OciDownloader(inHfSettings.exportSettings, inHfSettings.task, inHfSettings.sourceModel,
            ovms::IModelDownloader::getGraphDirectory(inHfSettings.downloadPath, inHfSettings.sourceModel),
            inHfSettings.overwriteModels, llmmanBinary) {}

    std::string getVersionCmd() const { return OciDownloader::getVersionCmd(); }
    std::string getResolveCmd() const { return OciDownloader::getResolveCmd(); }
    ovms::Status checkLlmmanIsPresent() { return OciDownloader::checkLlmmanIsPresent(); }
    std::string getGraphDirectory() { return OciDownloader::getGraphDirectory(); }
    static ovms::Status parseResolveOutput(const std::string& output, std::string& outPath, std::string& outFormat) {
        return OciDownloader::parseResolveOutput(output, outPath, outFormat);
    }
    static bool containsOpenVinoIr(const std::string& directory) {
        return OciDownloader::containsOpenVinoIr(directory);
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
    EXPECT_EQ(ovms::localModelDirectoryName("oci://ghcr.io/org/model:tag"), "ghcr.io/org/model_tag");
    EXPECT_EQ(ovms::localModelDirectoryName("oci://registry:5000/org/model:tag"), "registry_5000/org/model_tag");
    EXPECT_EQ(ovms::localModelDirectoryName("oci://ghcr.io/org/model"), "ghcr.io/org/model");
}

TEST(OciSchemeTest, GraphDirectoryUsesSanitizedName) {
    const std::string expected = ovms::FileSystem::joinPath({"/models", "ghcr.io/org/model_tag"});
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

class OciDownloaderCommands : public ::testing::Test {
public:
    ovms::HFSettingsImpl hfSettings;
    void SetUp() override {
        hfSettings.sourceModel = "oci://ghcr.io/org/model:tag";
        hfSettings.downloadPath = "/models";
        hfSettings.task = ovms::TEXT_GENERATION_GRAPH;
        hfSettings.downloadType = ovms::OCI_DOWNLOAD;
    }
};

TEST_F(OciDownloaderCommands, ResolveCommandDropsTheScheme) {
    TestOciDownloader downloader(hfSettings, "llmman");
    EXPECT_EQ(downloader.getResolveCmd(), "llmman resolve \"ghcr.io/org/model:tag\"");
    EXPECT_EQ(downloader.getVersionCmd(), "llmman --version");
}

TEST_F(OciDownloaderCommands, GraphDirectoryIsSanitized) {
    TestOciDownloader downloader(hfSettings, "llmman");
    EXPECT_EQ(downloader.getGraphDirectory(), ovms::FileSystem::joinPath({"/models", "ghcr.io/org/model_tag"}));
}

TEST_F(OciDownloaderCommands, BinaryIsTakenFromEnvironment) {
    EnvGuard guard;
    guard.set("LLMMAN_BIN", "/opt/bin/llmman");
    EXPECT_EQ(OciDownloader::resolveLlmmanBinary(), "/opt/bin/llmman");
    TestOciDownloader downloader(hfSettings);
    EXPECT_EQ(downloader.getVersionCmd(), "/opt/bin/llmman --version");
}

TEST_F(OciDownloaderCommands, BinaryDefaultsToPathLookup) {
    EnvGuard guard;
    guard.unset("LLMMAN_BIN");
    EXPECT_EQ(OciDownloader::resolveLlmmanBinary(), "llmman");
}

TEST_F(OciDownloaderCommands, MissingBinaryIsReported) {
    TestOciDownloader downloader(hfSettings, "llmman-that-does-not-exist");
    EXPECT_EQ(downloader.checkLlmmanIsPresent(), StatusCode::OCI_LLMMAN_NOT_FOUND);
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
    std::string resolvedPath;
    ovms::HFSettingsImpl hfSettings;

    void SetUp() override {
        TestWithTempDir::SetUp();
#ifdef _WIN32
        llmmanMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/llmman.exe");
#else
        llmmanMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/llmman");
#endif
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

    TestOciDownloader downloader(hfSettings, llmmanMockPath);
    ASSERT_EQ(downloader.downloadModel(), StatusCode::OK);
    // No second copy of the weights: graph.pbtxt just points at llmman's store.
    EXPECT_EQ(downloader.getModelPath(), resolvedPath);
    EXPECT_FALSE(downloader.getGgufFilename().has_value());
    // The graph directory still has to exist, that is where graph.pbtxt goes.
    EXPECT_TRUE(std::filesystem::is_directory(downloader.getGraphDirectory()));
}

TEST_F(OciDownloaderPayload, GgufModelIsSplitIntoDirectoryAndFilename) {
    createFile(resolvedPath, "model-Q4_K_M.gguf");
    const std::string ggufPath = std::filesystem::path(resolvedPath).append("model-Q4_K_M.gguf").generic_string();

    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", ggufPath);
    guard.set("LLMMAN_MOCK_FORMAT", "gguf");

    TestOciDownloader downloader(hfSettings, llmmanMockPath);
    ASSERT_EQ(downloader.downloadModel(), StatusCode::OK);
    // The graph exporter joins these two back together into models_path.
    EXPECT_EQ(std::filesystem::path(downloader.getModelPath()).generic_string(), resolvedPath);
    ASSERT_TRUE(downloader.getGgufFilename().has_value());
    EXPECT_EQ(downloader.getGgufFilename().value(), "model-Q4_K_M.gguf");
}

TEST_F(OciDownloaderPayload, UnsupportedFormatIsRejected) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", resolvedPath);
    guard.set("LLMMAN_MOCK_FORMAT", "onnx");

    TestOciDownloader downloader(hfSettings, llmmanMockPath);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_UNSUPPORTED_MODEL_FORMAT);
}

TEST_F(OciDownloaderPayload, ResolveFailureIsPropagated) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_FAIL", "1");

    TestOciDownloader downloader(hfSettings, llmmanMockPath);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_LLMMAN_RESOLVE_FAILED);
}

TEST_F(OciDownloaderPayload, UnparseableResolveOutputIsRejected) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_OUTPUT", "this is not the JSON you are looking for");

    TestOciDownloader downloader(hfSettings, llmmanMockPath);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

TEST_F(OciDownloaderPayload, NonExistentResolvedPathIsRejected) {
    EnvGuard guard;
    guard.set("LLMMAN_MOCK_PATH", std::filesystem::path(this->directoryPath).append("gone").generic_string());
    guard.set("LLMMAN_MOCK_FORMAT", "safetensors");

    TestOciDownloader downloader(hfSettings, llmmanMockPath);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID);
}

TEST_F(OciDownloaderPayload, EscapedDownloadPathIsRejected) {
    hfSettings.downloadPath = "../some/path";
    TestOciDownloader downloader(hfSettings, llmmanMockPath);
    EXPECT_EQ(downloader.downloadModel(), StatusCode::PATH_INVALID);
}
