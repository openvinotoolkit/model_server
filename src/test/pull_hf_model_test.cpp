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

#include "test_utils.hpp"
#include "../filesystem.hpp"
#include "src/pull_module/hf_pull_model_module.hpp"
#include "src/pull_module/libgit2.hpp"
#include "src/servables_config_manager_module/listmodels.hpp"
#include "src/modelextensions.hpp"

#include "../server.hpp"
#include "src/stringutils.hpp"
#include "../timer.hpp"

struct EnvGuard {
    EnvGuard() {
        SPDLOG_TRACE("EnvGuardConstructor");
    }
    void set(const std::string& name, const std::string& value) {
        std::optional<std::string> originalValue = std::nullopt;
        const char* currentVal = std::getenv(name.c_str());
        if (currentVal) {
            SPDLOG_TRACE("Var:{} is set to value:{}", name, currentVal);
            originalValue = std::string(currentVal);
        } else {
            SPDLOG_TRACE("Var:{} was not set");
        }
        if (originalValues.find(name) == originalValues.end()) {
            SPDLOG_TRACE("Var:{} value was not stored yet", name);
            originalValues[name] = originalValue;
        }
        SetEnvironmentVar(name, value);
    }
    void unset(const std::string& name) {
        std::optional<std::string> originalValue = std::nullopt;
        const char* currentVal = std::getenv(name.c_str());
        if (currentVal) {
            SPDLOG_TRACE("Var:{} is set to value:{}", name, currentVal);
            originalValue = std::string(currentVal);
        } else {
            SPDLOG_TRACE("Var:{} was not set");
        }
        if (originalValues.find(name) == originalValues.end()) {
            SPDLOG_TRACE("Var:{} value was not stored yet", name);
            originalValues[name] = originalValue;
        }
        UnSetEnvironmentVar(name);
    }
    ~EnvGuard() {
        SPDLOG_TRACE("EnvGuardDestructor");
        for (auto& [k, v] : originalValues) {
            if (v.has_value()) {
                SPDLOG_TRACE("Var:{} was set to value:{}", k, v.value());
                SetEnvironmentVar(k, v.value());
            } else {
                SPDLOG_TRACE("Var:{} was empty", k);
                UnSetEnvironmentVar(k);
            }
        }
    }

private:
    std::unordered_map<std::string, std::optional<std::string>> originalValues;
};

class HfDownloaderPullHfModel : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;

    void ServerPullHfModel(std::string& sourceModel, std::string& downloadPath) {
        ::SetUpServerForDownload(this->t, this->server, sourceModel, downloadPath);
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        // Clone sets readonly - need to remove it before we can delete on windows
        RemoveReadonlyFileAttributeFromDir(this->directoryPath);
        TestWithTempDir::TearDown();
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
            device: CPU,
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

TEST_F(HfDownloaderPullHfModel, PositiveDownload) {
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = ovms::FileSystem::appendSlash(this->directoryPath) + "repository";
    this->ServerPullHfModel(modelName, downloadPath);
    std::string modelPath = ovms::FileSystem::appendSlash(downloadPath) + "openvino_model.bin";
    std::string graphPath = ovms::FileSystem::appendSlash(downloadPath) + "graph.pbtxt";
    ASSERT_EQ(std::filesystem::exists(modelPath), true);
    ASSERT_EQ(std::filesystem::exists(graphPath), true);
    ASSERT_EQ(std::filesystem::file_size(modelPath), 52417240);
    std::string graphContents = GetFileContents(graphPath);
    std::cout << graphContents << std::endl;

    ASSERT_EQ(expectedGraphContents, graphContents);
}

class TestHfDownloader : public ovms::HfDownloader {
public:
    TestHfDownloader(const std::string& sourceModel, const std::string& downloadPath, const std::string& hfEndpoint, const std::string& hfToken, const std::string& httpProxy) :
        HfDownloader(sourceModel, downloadPath, hfEndpoint, hfToken, httpProxy) {}
    std::string GetRepoUrl() { return HfDownloader::GetRepoUrl(); }
    std::string GetRepositoryUrlWithPassword() { return HfDownloader::GetRepositoryUrlWithPassword(); }
    bool CheckIfProxySet() { return HfDownloader::CheckIfProxySet(); }
    void setProxy(const std::string& proxy) { this->httpProxy = proxy; }
    void setEndpoint(const std::string& endpoint) { this->hfEndpoint = endpoint; }
    const std::string& getEndpoint() { return this->hfEndpoint; }
    const std::string& getProxy() { return this->httpProxy; }
};

TEST(HfDownloaderClassTest, Methods) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::string hfEndpoint = "www.new_hf.com/";
    std::string hfToken = "123$$o_O123!AAbb";
    std::string httpProxy = "https://proxy_test1:123";
    std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(modelName, downloadPath, hfEndpoint, hfToken, httpProxy);
    ASSERT_EQ(hfDownloader->getProxy(), httpProxy);
    ASSERT_EQ(hfDownloader->CheckIfProxySet(), true);

    EXPECT_EQ(TestHfDownloader(modelName, downloadPath, hfEndpoint, hfToken, "").CheckIfProxySet(), false);
    ASSERT_EQ(hfDownloader->getEndpoint(), "www.new_hf.com/");
    ASSERT_EQ(hfDownloader->GetRepoUrl(), "https://www.new_hf.com/model/name");
    ASSERT_EQ(hfDownloader->GetRepositoryUrlWithPassword(), "https://123$$o_O123!AAbb:123$$o_O123!AAbb@www.new_hf.com/model/name");
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
    ASSERT_EQ(testHfPullModelModule->GetHfEndpoint(), "huggingface.co/");
    this->guard.set(endpoint_env, endpoint);

    std::string hfEndpoint = testHfPullModelModule->GetHfEndpoint();
    ASSERT_EQ(hfEndpoint, "www.new_hf.com/");
}

TEST(OvmsGraphConfigTest, positiveAllChanged) {
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = "test/repository";
    char* n_argv[] = {
        (char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--pipeline_type",
        (char*)"VLM",
        (char*)"--max_num_seqs",
        (char*)"128",
        (char*)"--graph_target_device",
        (char*)"GPU",
        (char*)"--enable_prefix_caching",
        (char*)"false",
        (char*)"--cache_size",
        (char*)"20",
        (char*)"--max_num_batched_tokens",
        (char*)"16",
        (char*)"--dynamic_split_fuse",
        (char*)"true",
        (char*)"--draft_source_model",
        (char*)"/draft/model/source",
    };

    int arg_count = 22;
    ConstructorEnabledConfig config;
    config.parse(arg_count, n_argv);

    ASSERT_EQ(config.getServerSettings().hfSettings.sourceModel, modelName);
    ASSERT_EQ(config.getServerSettings().hfSettings.downloadPath, downloadPath);
    ASSERT_EQ(config.getServerSettings().hfSettings.pullHfModelMode, true);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.pipelineType.value(), "VLM");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.modelPath, "./");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.maxNumSeqs, 128);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.targetDevice, "GPU");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.pluginConfig.kvCachePrecision.has_value(), false);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.enablePrefixCaching, "false");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.cacheSize, 20);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.maxNumBatchedTokens.value(), 16);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.dynamicSplitFuse, "true");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.draftModelDirName.value(), "/draft/model/source");
}

TEST(OvmsGraphConfigTest, positiveSomeChanged) {
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = "test/repository";
    char* n_argv[] = {
        (char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--pipeline_type",
        (char*)"VLM",
        (char*)"--max_num_seqs",
        (char*)"128",
        (char*)"--graph_target_device",
        (char*)"NPU",
    };

    int arg_count = 12;
    ConstructorEnabledConfig config;
    config.parse(arg_count, n_argv);

    ASSERT_EQ(config.getServerSettings().hfSettings.sourceModel, modelName);
    ASSERT_EQ(config.getServerSettings().hfSettings.downloadPath, downloadPath);
    ASSERT_EQ(config.getServerSettings().hfSettings.pullHfModelMode, true);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.pipelineType.value(), "VLM");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.modelPath, "./");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.maxNumSeqs, 128);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.targetDevice, "NPU");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.pluginConfig.kvCachePrecision.has_value(), false);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.enablePrefixCaching, "true");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.cacheSize, 10);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.maxNumBatchedTokens.has_value(), false);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.dynamicSplitFuse, "true");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.draftModelDirName.has_value(), false);
}

TEST(OvmsGraphConfigTest, positiveDefault) {
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = "test/repository";
    char* n_argv[] = {
        (char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)modelName.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
    };

    int arg_count = 6;
    ConstructorEnabledConfig config;
    config.parse(arg_count, n_argv);
    ASSERT_EQ(config.getServerSettings().hfSettings.sourceModel, modelName);
    ASSERT_EQ(config.getServerSettings().hfSettings.downloadPath, downloadPath);
    ASSERT_EQ(config.getServerSettings().hfSettings.pullHfModelMode, true);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.pipelineType.has_value(), false);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.modelPath, "./");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.maxNumSeqs, 256);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.targetDevice, "CPU");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.pluginConfig.kvCachePrecision.has_value(), false);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.enablePrefixCaching, "true");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.cacheSize, 10);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.maxNumBatchedTokens.has_value(), false);
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.dynamicSplitFuse, "true");
    ASSERT_EQ(config.getServerSettings().hfSettings.graphSettings.draftModelDirName.has_value(), false);
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
        nullptr};

    int arg_count = 6;
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
