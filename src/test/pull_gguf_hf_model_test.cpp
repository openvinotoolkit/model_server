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

#include <git2.h>
#include "test_utils.hpp"
#include "gguf_environment.hpp"
#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "src/pull_module/gguf_export.hpp"
#include "src/modelextensions.hpp"

#include "../module.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "src/stringutils.hpp"
#include "../timer.hpp"
namespace {
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
}  // namespace

class GGUFDownloaderPullHfModel : public TestWithTempDir {
protected:
    void TearDown() {
        RemoveReadonlyFileAttributeFromDir(this->directoryPath);
        // TestWithTempDir::TearDown();
    }
};

// Make parametrized test for GGUFDownloaderPullHfModel
// where i pass endpoint, sourceModel, ggufFilename

class GGUFDownloaderPullHfModelParameterized : public GGUFDownloaderPullHfModel, public ::testing::WithParamInterface<std::tuple<std::string, std::string, std::string, std::string>> {
};

TEST_P(GGUFDownloaderPullHfModelParameterized, PositiveDownload) {
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string hfEndpoint = std::get<0>(GetParam());
    const std::string sourceModel = std::get<1>(GetParam());
    const std::string filenamePrefix = std::get<2>(GetParam());
    const std::string ggufFileName = std::get<3>(GetParam());
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    auto status = ovms::GGUFDownloader::downloadWithCurl(hfEndpoint, sourceModel, filenamePrefix, ggufFileName, downloadPath);
    ASSERT_TRUE(status.ok()) << status.string();
    std::string fullPath = ovms::FileSystem::joinPath({downloadPath, ggufFileName});
    bool exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(exist) << "File " << fullPath << " does not exist after download";
}

std::vector<std::tuple<std::string, std::string, std::string, std::string>> ggufParams = {
    std::make_tuple("https://huggingface.co/", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "/resolve/main/", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
    std::make_tuple("https://www.modelscope.cn/", "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", "/resolve/master/", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
    // std::make_tuple("https://www.modelscope.cn/", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "/resolve/main/", "DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf"),
    std::make_tuple("https://hf-mirror.com/", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "/resolve/main/", "DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf")};
// FIXME make handling independent of "/" in the endpoint

INSTANTIATE_TEST_SUITE_P(
    GGUFDownloaderPullHfModelTests,
    GGUFDownloaderPullHfModelParameterized,
    ::testing::ValuesIn(ggufParams),
    [](const ::testing::TestParamInfo<GGUFDownloaderPullHfModelParameterized::ParamType>& info) {
        auto paramTuple = info.param;
        std::string paramName = ovms::joins({
                                                std::get<0>(paramTuple),  // hfEndpoint
                                                std::get<1>(paramTuple),  // sourceModel
                                                std::get<2>(paramTuple),  // filenamePrefix
                                                std::get<3>(paramTuple)   // ggufFileName
                                            },
            "_");
        std::replace(paramName.begin(), paramName.end(), '-', '_');
        std::replace(paramName.begin(), paramName.end(), '/', '_');
        std::replace(paramName.begin(), paramName.end(), ':', '_');
        std::replace(paramName.begin(), paramName.end(), '.', '_');
        return paramName;
    });

TEST_F(GGUFDownloaderPullHfModel, PositiveDownload) {
    // curl -L   -H "Authorization: Bearer $HF_TOKEN"   -o DeepSeek‑R1‑Distill‑Qwen‑7B‑Q4_K_M.gguf   https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    // https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string sourceModel = "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF";
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    const std::string filenamePrefix = "/resolve/main/";
    const std::string ggufFileName = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf";
    const std::string hfEndpoint = "https://huggingface.co/";
    // const std::string hfEndpoint = "https://www.modelscope.cn";
    SPDLOG_ERROR("ER");
    auto status = ovms::GGUFDownloader::downloadWithCurl(hfEndpoint, sourceModel, filenamePrefix, ggufFileName, downloadPath);
    SPDLOG_ERROR("ER");
    ASSERT_TRUE(status.ok()) << status.string();
    // TODO check if file exists, its size
    std::string fullPath = ovms::FileSystem::joinPath({downloadPath, ggufFileName});
    bool exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok());
    // check size of the file with std::filesystem
    size_t fileSize = 0;
    std::filesystem::path filePath(fullPath);
    fileSize = std::filesystem::file_size(filePath);
    EXPECT_EQ(13, fileSize);
}

void find_file_in_tree(git_repository* repo, git_tree* tree,
    const std::regex& pattern,
    const std::string& current_path,
    std::vector<std::string>& matches) {
    size_t count = git_tree_entrycount(tree);
    for (size_t i = 0; i < count; ++i) {
        const git_tree_entry* entry = git_tree_entry_byindex(tree, i);
        std::string name = git_tree_entry_name(entry);
        std::string full_path = current_path.empty() ? name : current_path + "/" + name;

        if (git_tree_entry_type(entry) == GIT_OBJECT_BLOB) {
            if (std::regex_match(name, pattern)) {
                matches.push_back(full_path);
            }
        } else if (git_tree_entry_type(entry) == GIT_OBJECT_TREE) {
            git_object* obj = nullptr;
            if (git_tree_entry_to_object(&obj, repo, entry) == 0) {
                git_tree* subtree = (git_tree*)obj;
                find_file_in_tree(repo, subtree, pattern, full_path, matches);
                git_tree_free(subtree);
            }
        }
    }
}

TEST(libgit2, manuallyClonedRepoMetadataGetGGUFFiles) {
    GTEST_SKIP() << "Assuming we could run git in shell we could retrievegguf files like this";
    // git clone --filter=blob:none --no-checkout https://huggingface.co/unsloth/gpt-oss-120b-GGUF ./curl_test_repo
    git_libgit2_init();
    git_repository* repo = nullptr;
    git_repository_open(&repo, "/ovms/curl_test_repo");

    git_object* commit_obj = nullptr;
    git_revparse_single(&commit_obj, repo, "HEAD");

    git_commit* commit = (git_commit*)commit_obj;
    git_tree* tree = nullptr;
    git_commit_tree(&tree, commit);

    std::vector<std::string> matches;
    std::string partial_filename = "gguf";
    // construct regex that catches all file with gguf in name
    std::regex pattern(".*" + partial_filename + ".*");

    find_file_in_tree(repo, tree, pattern, "/ovms_curl_test_repo", matches);

    for (auto& m : matches) {
        std::cout << "Found: " << m << "\n";
    }

    git_tree_free(tree);
    git_commit_free(commit);
    git_repository_free(repo);
    git_libgit2_shutdown();
}
