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

#include "test_utils.hpp"
#include "../filesystem.hpp"
#include "src/libgt2/libgt2.hpp"
#include "../server.hpp"

class PullHfModel : public TestWithTempDir {
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

TEST_F(PullHfModel, PositiveDownload) {
    std::string modelName = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov";
    std::string downloadPath = ovms::FileSystem::appendSlash(this->directoryPath) + "repository";
    this->ServerPullHfModel(modelName, downloadPath);
    std::string modelPath = ovms::FileSystem::appendSlash(downloadPath) + "openvino_model.bin";
    ASSERT_EQ(std::filesystem::exists(modelPath), true);
    ASSERT_EQ(std::filesystem::file_size(modelPath), 52417240);
}

TEST(HfDownloaderClassTest, Constructor) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::unique_ptr<ovms::HfDownloader> hfDownloader = std::make_unique<ovms::HfDownloader>(modelName, downloadPath, true);
    ASSERT_EQ(hfDownloader->isPullHfModelModeOn(), true);
}

class TestHfDownloader : public ovms::HfDownloader {
public:
    TestHfDownloader(const std::string& sourceModel, const std::string& downloadPath, bool pullHfModelMode) {
        this->sourceModel = sourceModel;
        this->downloadPath = downloadPath;
        this->pullHfModelMode = pullHfModelMode;
    }
    std::string GetRepoUrl(std::string& hfEndpoint) { return HfDownloader::GetRepoUrl(hfEndpoint); }
    std::string GetHfEndpoint() { return HfDownloader::GetHfEndpoint(); }
    std::string GetRepositoryUrlWithPassword(std::string& hfEndpoint) { return HfDownloader::GetRepositoryUrlWithPassword(hfEndpoint); }
    bool CheckIfProxySet() { return HfDownloader::CheckIfProxySet(); }
    bool CheckIfTokenSet() { return HfDownloader::CheckIfTokenSet(); }
};

TEST(HfDownloaderClassTest, Methods) {
    std::string modelName = "model/name";
    std::string downloadPath = "/path/to/Download";
    std::unique_ptr<TestHfDownloader> hfDownloader = std::make_unique<TestHfDownloader>(modelName, downloadPath, false);
    ASSERT_EQ(hfDownloader->isPullHfModelModeOn(), false);

    std::string proxy_env = "https_proxy";
    std::string proxy = "https://proxy_test1:123";
    UnSetEnvironmentVar(proxy_env);
    ASSERT_EQ(hfDownloader->CheckIfProxySet(), false);
    SetEnvironmentVar(proxy_env, proxy);
    ASSERT_EQ(hfDownloader->CheckIfProxySet(), true);

    std::string token_env = "HF_TOKEN";
    std::string token = "123$$o_O123!AAbb";
    UnSetEnvironmentVar(token_env);
    ASSERT_EQ(hfDownloader->CheckIfTokenSet(), false);
    SetEnvironmentVar(token_env, token);
    ASSERT_EQ(hfDownloader->CheckIfTokenSet(), true);

    std::string endpoint_env = "HF_ENDPOINT";
    std::string endpoint = "www.new_hf.com";
    UnSetEnvironmentVar(endpoint_env);
    ASSERT_EQ(hfDownloader->GetHfEndpoint(), "huggingface.co/");
    SetEnvironmentVar(endpoint_env, endpoint);

    std::string hfEndpoint = hfDownloader->GetHfEndpoint();
    ASSERT_EQ(hfEndpoint, "www.new_hf.com/");

    ASSERT_EQ(hfDownloader->GetRepoUrl(hfEndpoint), "https://www.new_hf.com/model/name");

    ASSERT_EQ(hfDownloader->GetRepositoryUrlWithPassword(hfEndpoint), "https://123$$o_O123!AAbb:123$$o_O123!AAbb@www.new_hf.com/model/name");
}
