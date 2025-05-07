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
#include "../libgt2/hf_pull_model_module.hpp"
#include "src/libgt2/libgt2.hpp"
#include "../server.hpp"

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
    std::string oldHfEndpoint = "";
    std::string oldHfToken = "";
    std::string oldHttpProxy = "";
    std::string proxy_env = "https_proxy";
    std::string token_env = "HF_TOKEN";
    std::string endpoint_env = "HF_ENDPOINT";
    void SetUp() {
        this->oldHttpProxy = GetEnvVar(proxy_env);
        this->oldHfToken = GetEnvVar(token_env);
        this->oldHfEndpoint = GetEnvVar(endpoint_env);
    }

    void TearDown() {
        SetEnvironmentVar(proxy_env, oldHttpProxy);
        SetEnvironmentVar(token_env, oldHfToken);
        SetEnvironmentVar(endpoint_env, oldHfEndpoint);
    }
};

TEST_F(HfDownloaderHfEnvTest, Methods) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::unique_ptr<TestHfPullModelModule> testHfPullModelModule = std::make_unique<TestHfPullModelModule>();

    std::string proxy = "https://proxy_test1:123";
    UnSetEnvironmentVar(proxy_env);
    ASSERT_EQ(testHfPullModelModule->GetProxy(), "");
    SetEnvironmentVar(proxy_env, proxy);
    ASSERT_EQ(testHfPullModelModule->GetProxy(), proxy);

    std::string token = "123$$o_O123!AAbb";
    UnSetEnvironmentVar(token_env);
    ASSERT_EQ(testHfPullModelModule->GetHfToken(), "");
    SetEnvironmentVar(token_env, token);
    ASSERT_EQ(testHfPullModelModule->GetHfToken(), token);

    std::string endpoint = "www.new_hf.com";
    UnSetEnvironmentVar(endpoint_env);
    ASSERT_EQ(testHfPullModelModule->GetHfEndpoint(), "huggingface.co/");
    SetEnvironmentVar(endpoint_env, endpoint);

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
