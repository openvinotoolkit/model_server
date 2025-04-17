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
#include <string>
#include <thread>

#include "test_utils.hpp"
#include "../filesystem.hpp"
#include "../server.hpp"

class PullHfModel : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;

    void ServerPullHfModel(std::string& sourceModel, std::string& repoPath) {
        ::SetUpServerForDownload(this->t, this->server, sourceModel, repoPath);
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
    std::string repoPath = ovms::FileSystem::appendSlash(this->directoryPath) + "repository";
    this->ServerPullHfModel(modelName, repoPath);
    
    std::string modelPath = ovms::FileSystem::appendSlash(repoPath) + "openvino_model.bin";
    ASSERT_EQ(std::filesystem::exists(modelPath), true);
    ASSERT_EQ(std::filesystem::file_size(modelPath), 52417240);
}
