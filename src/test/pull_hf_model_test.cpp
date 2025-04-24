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

TEST_F(HfDownloaderPullHfModel, PositiveDownload) {
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = ovms::FileSystem::appendSlash(this->directoryPath) + "repository";
    this->ServerPullHfModel(modelName, downloadPath);
    std::string modelPath = ovms::FileSystem::appendSlash(downloadPath) + "openvino_model.bin";
    ASSERT_EQ(std::filesystem::exists(modelPath), true);
    ASSERT_EQ(std::filesystem::file_size(modelPath), 52417240);
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
