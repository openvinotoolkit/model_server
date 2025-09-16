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
#include <memory>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/test/env_guard.hpp"
#include "src/test/light_test_utils.hpp"
#include "src/test/test_utils.hpp"
#include "src/test/test_with_temp_dir.hpp"
#include "src/filesystem.hpp"
#include "src/pull_module/hf_pull_model_module.hpp"
#include "src/pull_module/libgit2.hpp"
#include "src/pull_module/optimum_export.hpp"
#include "src/servables_config_manager_module/listmodels.hpp"
#include "src/modelextensions.hpp"

#include "../module.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "src/stringutils.hpp"
#include "../timer.hpp"

class HfDownloaderPullHfModel : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;

    void ServerPullHfModel(std::string& sourceModel, std::string& downloadPath, std::string& task, int expected_code = 0, int timeoutSeconds = 15) {
        ::SetUpServerForDownload(this->t, this->server, sourceModel, downloadPath, task, expected_code, timeoutSeconds);
    }

    void ServerPullHfModelWithDraft(std::string& draftModel, std::string& sourceModel, std::string& downloadPath, std::string& task, int expected_code = 0, int timeoutSeconds = 15) {
        ::SetUpServerForDownloadWithDraft(this->t, this->server, draftModel, sourceModel, downloadPath, task, expected_code, timeoutSeconds);
    }

    void SetUpServerForDownloadAndStart(std::string& sourceModel, std::string& downloadPath, std::string& task, int timeoutSeconds = 15) {
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

    // Removes # OpenVINO Model Server REPLACE_PROJECT_VERSION comment added for debug purpose in graph export at the begging of graph.pbtxt
    // This string differs per build and setup
    std::string removeVersionString(std::string input) {
        return input.erase(0, input.find("\n") + 1);
    }
};

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
            plugin_config: '{ }',
            enable_prefix_caching: true,
            cache_size: 10,
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
            plugin_config: '{ }',
            enable_prefix_caching: true,
            cache_size: 10,
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

TEST_F(HfDownloaderPullHfModel, PositiveDownload) {
    GTEST_SKIP() << "Skipping test in CI - PositiveDownloadAndStart has full scope testing.";
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = ovms::FileSystem::joinPath({this->directoryPath, "repository"});
    std::string task = "text_generation";
    this->ServerPullHfModel(modelName, downloadPath, task);

    std::string basePath = ovms::FileSystem::joinPath({this->directoryPath, "repository", "OpenVINO", "Phi-3-mini-FastDraft-50M-int8-ov"});
    std::string modelPath = ovms::FileSystem::appendSlash(basePath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), 52417240);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(HfDownloaderPullHfModel, PositiveDownloadAndStart) {
    // EnvGuard guard;
    // guard.set("HF_ENDPOINT", "https://modelscope.cn");
    // guard.set("HF_ENDPOINT", "https://hf-mirror.com");
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    this->filesToPrintInCaseOfFailure.emplace_back("config.json");
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = ovms::FileSystem::joinPath({this->directoryPath, "repository"});
    std::string task = "text_generation";
    this->SetUpServerForDownloadAndStart(modelName, downloadPath, task);

    std::string basePath = ovms::FileSystem::joinPath({this->directoryPath, "repository", "OpenVINO", "Phi-3-mini-FastDraft-50M-int8-ov"});
    std::string modelPath = ovms::FileSystem::appendSlash(basePath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), 52417240);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContents, removeVersionString(graphContents)) << graphContents;
}

TEST_F(HfDownloaderPullHfModel, PositiveDownloadWithDraftModel) {
    // EnvGuard guard;
    // guard.set("HF_ENDPOINT", "https://modelscope.cn");
    // guard.set("HF_ENDPOINT", "https://hf-mirror.com");
    this->filesToPrintInCaseOfFailure.emplace_back("graph.pbtxt");
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string draftModel = "OpenVINO/distil-small.en-int4-ov";
    std::string downloadPath = ovms::FileSystem::joinPath({this->directoryPath, "repository"});
    std::string task = "text_generation";
    this->ServerPullHfModelWithDraft(draftModel, modelName, downloadPath, task);

    std::string basePath = ovms::FileSystem::joinPath({this->directoryPath, "repository", "OpenVINO", "Phi-3-mini-FastDraft-50M-int8-ov"});
    std::string modelPath = ovms::FileSystem::appendSlash(basePath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(basePath) + "graph.pbtxt";

    ASSERT_EQ(std::filesystem::exists(modelPath), true) << modelPath;
    ASSERT_EQ(std::filesystem::exists(graphPath), true) << graphPath;
    ASSERT_EQ(std::filesystem::file_size(modelPath), 52417240);
    std::string graphContents = GetFileContents(graphPath);

    ASSERT_EQ(expectedGraphContentsDraft, removeVersionString(graphContents)) << graphContents;

    std::string basePath2 = ovms::FileSystem::joinPath({basePath, "OpenVINO-distil-small.en-int4-ov"});
    std::string modelPath2 = ovms::FileSystem::appendSlash(basePath2) + "openvino_tokenizer.bin";

    ASSERT_EQ(std::filesystem::exists(modelPath2), true) << modelPath2;
    ASSERT_EQ(std::filesystem::file_size(modelPath2), 2022483);
}

class TestOptimumDownloader : public ovms::OptimumDownloader {
public:
    TestOptimumDownloader(const ovms::HFSettingsImpl& inHfSettings) :
        ovms::OptimumDownloader(inHfSettings, inHfSettings.sourceModel, inHfSettings.downloadPath, inHfSettings.overwriteModels) {}
    std::string getExportCmd() { return ovms::OptimumDownloader::getExportCmd(); }
    std::string getGraphDirectory() { return ovms::OptimumDownloader::getGraphDirectory(); }
    void setExportCliCheckCommand(const std::string& input) { this->OPTIMUM_CLI_CHECK_COMMAND = input; }
    void setExportCliExportCommand(const std::string& input) { this->OPTIMUM_CLI_EXPORT_COMMAND = input; }
    ovms::Status checkRequiredToolsArePresent() { return ovms::OptimumDownloader::checkRequiredToolsArePresent(); }
};

class TestHfDownloader : public ovms::HfDownloader {
public:
    TestHfDownloader(const std::string& sourceModel, const std::string& downloadPath, const std::string& hfEndpoint, const std::string& hfToken, const std::string& httpProxy, bool overwrite) :
        HfDownloader(sourceModel, downloadPath, hfEndpoint, hfToken, httpProxy, overwrite) {}
    std::string GetRepoUrl() { return HfDownloader::GetRepoUrl(); }
    std::string GetRepositoryUrlWithPassword() { return HfDownloader::GetRepositoryUrlWithPassword(); }
    bool CheckIfProxySet() { return HfDownloader::CheckIfProxySet(); }
    void setProxy(const std::string& proxy) { this->httpProxy = proxy; }
    void setEndpoint(const std::string& endpoint) { this->hfEndpoint = endpoint; }
    const std::string& getEndpoint() { return this->hfEndpoint; }
    const std::string& getProxy() { return this->httpProxy; }
    std::string getGraphDirectory(const std::string& downloadPath, const std::string& sourceModel) { return HfDownloader::getGraphDirectory(downloadPath, sourceModel); }
    std::string getGraphDirectory() { return HfDownloader::getGraphDirectory(); }
};

TEST(HfDownloaderClassTest, Methods) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::string hfEndpoint = "www.new_hf.com/";
    std::string hfToken = "123$$o_O123!AAbb";
    std::string httpProxy = "https://proxy_test1:123";
    std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(modelName, downloadPath, hfEndpoint, hfToken, httpProxy, false);
    ASSERT_EQ(hfDownloader->getProxy(), httpProxy);
    ASSERT_EQ(hfDownloader->CheckIfProxySet(), true);

    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).CheckIfProxySet(), false);
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

class TestOptimumDownloaderSetup : public ::testing::Test {
public:
    ovms::HFSettingsImpl inHfSettings;
    std::string cliMockPath;
    void SetUp() override {
        inHfSettings.sourceModel = "model/name";
        inHfSettings.downloadPath = "/path/to/Download";
        inHfSettings.precision = "fp64";
        inHfSettings.extraQuantizationParams = "--param --param value";
        inHfSettings.task = ovms::TEXT_GENERATION_GRAPH;
        inHfSettings.downloadType = ovms::OPTIMUM_CLI_DOWNLOAD;
#ifdef _WIN32
        cliMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/optimum-cli.exe");
#else
        cliMockPath = getGenericFullPathForBazelOut("/ovms/bazel-bin/src/optimum-cli");
#endif
    }
};

TEST_F(TestOptimumDownloaderSetup, Methods) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedPath = inHfSettings.downloadPath + "/" + inHfSettings.sourceModel;
    std::string expectedCmd = "optimum-cli export openvino --model model/name --trust-remote-code  --weight-format fp64 --param --param value \\path\\to\\Download\\model\\name";
#ifdef _WIN32
    std::replace(expectedPath.begin(), expectedPath.end(), '/', '\\');
#endif
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getGraphDirectory(), expectedPath);
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
}

TEST_F(TestOptimumDownloaderSetup, RerankExportCmd) {
    inHfSettings.task = ovms::RERANK_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --disable-convert-tokenizer --model model/name --trust-remote-code  --weight-format fp64 --task text-classification  \\path\\to\\Download\\model\\name";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
}

TEST_F(TestOptimumDownloaderSetup, ImageGenExportCmd) {
    inHfSettings.task = ovms::IMAGE_GENERATION_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --model model/name --weight-format fp64 \\path\\to\\Download\\model\\name";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
}

TEST_F(TestOptimumDownloaderSetup, EmbeddingsExportCmd) {
    inHfSettings.task = ovms::EMBEDDINGS_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string expectedCmd = "optimum-cli export openvino --disable-convert-tokenizer --task feature-extraction --library sentence_transformers --model model/name --trust-remote-code  --weight-format fp64 \\path\\to\\Download\\model\\name";
#ifdef __linux__
    std::replace(expectedCmd.begin(), expectedCmd.end(), '\\', '/');
#endif
    ASSERT_EQ(optimumDownloader->getExportCmd(), expectedCmd);
}

TEST_F(TestOptimumDownloaderSetup, UnknownExportCmd) {
    inHfSettings.task = ovms::UNKNOWN_GRAPH;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->getExportCmd(), "");
}

TEST_F(TestOptimumDownloaderSetup, NegativeWrongPath) {
    inHfSettings.downloadPath = "../path/to/Download";
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->cloneRepository(), ovms::StatusCode::PATH_INVALID);
}

TEST_F(TestOptimumDownloaderSetup, NegativeWrongDownloadType) {
    inHfSettings.downloadType = ovms::GIT_CLONE_DOWNLOAD;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->cloneRepository(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST_F(TestOptimumDownloaderSetup, NegativeUnknownDownloadType) {
    inHfSettings.downloadType = ovms::UNKNOWN_DOWNLOAD;
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    ASSERT_EQ(optimumDownloader->cloneRepository(), ovms::StatusCode::INTERNAL_ERROR);
}

TEST_F(TestOptimumDownloaderSetup, NegativeExportCommandFailed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    optimumDownloader->setExportCliCheckCommand("ls");
#ifdef _WIN32
    optimumDownloader->setExportCliCheckCommand("dir");
#endif
    optimumDownloader->setExportCliExportCommand("NonExistingCommand22");
    ASSERT_EQ(optimumDownloader->cloneRepository(), ovms::StatusCode::HF_RUN_OPTIMUM_CLI_EXPORT_FAILED);
}

TEST_F(TestOptimumDownloaderSetup, NegativeCheckOptimumExistsCommandFailed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    optimumDownloader->setExportCliCheckCommand("NonExistingCommand33");
    ASSERT_EQ(optimumDownloader->checkRequiredToolsArePresent(), ovms::StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI);
}

TEST_F(TestOptimumDownloaderSetup, PositiveOptimumExistsCommandPassed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    cliMockPath += " -h";
    optimumDownloader->setExportCliCheckCommand(cliMockPath);
    ASSERT_EQ(optimumDownloader->checkRequiredToolsArePresent(), ovms::StatusCode::OK);
}

TEST_F(TestOptimumDownloaderSetup, PositiveOptimumExportCommandPassed) {
    std::unique_ptr<TestOptimumDownloader> optimumDownloader = std::make_unique<TestOptimumDownloader>(inHfSettings);
    std::string cliCheckCommand = cliMockPath += " -h";
    optimumDownloader->setExportCliCheckCommand(cliCheckCommand);
    cliMockPath += " export";
    optimumDownloader->setExportCliExportCommand(cliMockPath);
    ASSERT_EQ(optimumDownloader->cloneRepository(), ovms::StatusCode::OK);
}

TEST(HfDownloaderClassTest, ProtocollsWithPassword) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::string hfEndpoint = "www.new_hf.com/";
    std::string hfToken = "";
    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "www.new_hf.com/model/name");
    hfEndpoint = "https://www.new_hf.com/";
    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "https://www.new_hf.com/model/name");
    hfEndpoint = "www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "http://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "http://123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "git://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "git://123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "ssh://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "ssh://123!$token:123!$token@www.new_hf.com/model/name");
    hfEndpoint = "what_ever_is_here://www.new_hf.com/";
    hfToken = "123!$token";
    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "", false).GetRepositoryUrlWithPassword(), "what_ever_is_here://123!$token:123!$token@www.new_hf.com/model/name");
}

TEST_F(HfDownloaderPullHfModel, MethodsNegative) {
    EXPECT_EQ(TestHfDownloader("name/test", "../some/path", "", "", "", false).cloneRepository(), ovms::StatusCode::PATH_INVALID);
    // Library not initialized
    EXPECT_EQ(TestHfDownloader("name/test", this->directoryPath, "", "", "", false).cloneRepository(), ovms::StatusCode::HF_GIT_CLONE_FAILED);
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
        eGuard.set("https_proxy", "");
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

TEST(ServerModulesBehaviorTests, PullAndStartModeErrorAndExpectFailAndNoOtherModulesStarted) {
    std::unique_ptr<ServerShutdownGuard> serverGuard;
    ovms::Server& server = ovms::Server::instance();
    DefaultEmptyValuesConfig config;
    config.getServerSettings().serverMode = ovms::HF_PULL_AND_START_MODE;
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
