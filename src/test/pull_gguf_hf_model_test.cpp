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
#include "env_guard.hpp"
#include "test_utils.hpp"
#include "gguf_environment.hpp"
#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "src/pull_module/gguf_downloader.hpp"
#include "src/modelextensions.hpp"

#include "../module.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "src/stringutils.hpp"
#include "../timer.hpp"

class GGUFDownloaderPullHfModel : public TestWithTempDir {
protected:
    void TearDown() {
        RemoveReadonlyFileAttributeFromDir(this->directoryPath);
        TestWithTempDir::TearDown();
    }
};

class GGUFDownloaderPullHfModelWithServer : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;

    void SetUpServerForDownloadAndStartGGUF(std::string& ggufFile, std::string& sourceModel, std::string& downloadPath, std::string& task, int timeoutSeconds = 60) {
        ::SetUpServerForDownloadAndStartGGUF(this->t, this->server, ggufFile, sourceModel, downloadPath, task, timeoutSeconds);
    }
    void TearDown() {
        server.setShutdownRequest(1);
        if (t)
            t->join();
        server.setShutdownRequest(0);
        RemoveReadonlyFileAttributeFromDir(this->directoryPath);
        TestWithTempDir::TearDown();
    }
};

class GGUFDownloaderPullHfModelParameterized : public GGUFDownloaderPullHfModel, public ::testing::WithParamInterface<std::tuple<std::string, std::string, std::string, std::string, size_t>> {
};

class GGUFDownloaderPullHfModelParameterizedWithServer : public GGUFDownloaderPullHfModelWithServer, public ::testing::WithParamInterface<std::tuple<std::string, std::string, size_t>> {
};

TEST_P(GGUFDownloaderPullHfModelParameterized, PositiveDownload) {
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string hfEndpoint = std::get<0>(GetParam());
    const std::string sourceModel = std::get<1>(GetParam());
    const std::string filenamePrefix = std::get<2>(GetParam());
    const std::string ggufFilename = std::get<3>(GetParam());
    const size_t expectedSize = std::get<4>(GetParam());
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    auto status = ovms::GGUFDownloader::downloadWithCurl(hfEndpoint, sourceModel, filenamePrefix, ggufFilename, downloadPath);
    ASSERT_TRUE(status.ok()) << status.string();
    std::string fullPath = ovms::FileSystem::joinPath({downloadPath, ggufFilename});
    bool exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(exist) << "File " << fullPath << " does not exist after download";
    // check size of the file with std::filesystem
    std::filesystem::path filePath(fullPath);
    size_t fileSize = std::filesystem::file_size(filePath);
    EXPECT_EQ(expectedSize, fileSize);
}

TEST_P(GGUFDownloaderPullHfModelParameterizedWithServer, PositiveDownloadAndStart) {
    SKIP_AND_EXIT_IF_NO_GGUF();
    std::string sourceModel = std::get<0>(GetParam());
    std::string ggufFilename = std::get<1>(GetParam());
    const size_t expectedSize = std::get<2>(GetParam());
    std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    std::string task = "text_generation";
    this->SetUpServerForDownloadAndStartGGUF(ggufFilename, sourceModel, downloadPath, task);
    std::string fullPath = ovms::FileSystem::joinPath({downloadPath, sourceModel, ggufFilename});
    bool exist = false;
    auto status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(exist) << "File " << fullPath << " does not exist after download";
    std::filesystem::path filePath(fullPath);
    size_t fileSize = std::filesystem::file_size(filePath);
    EXPECT_EQ(expectedSize, fileSize);
}

std::vector<std::tuple<std::string, std::string, std::string, std::string, size_t>> ggufParams = {
    std::make_tuple("https://huggingface.co/", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "/resolve/main/", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", size_t(1117321312)),
    std::make_tuple("https://www.modelscope.cn/", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "/resolve/main/", "DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf", size_t(752880736)),
    std::make_tuple("https://hf-mirror.com/", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "/resolve/main/", "DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf", size_t(752880736))};

std::vector<std::tuple<std::string, std::string, size_t>> ggufServerStartParams = {
    std::make_tuple("unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", size_t(1117321312))};

#ifndef _WIN32
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
                                                std::get<3>(paramTuple)   // ggufFilename
                                            },
            "_");
        std::replace(paramName.begin(), paramName.end(), '-', '_');
        std::replace(paramName.begin(), paramName.end(), '/', '_');
        std::replace(paramName.begin(), paramName.end(), ':', '_');
        std::replace(paramName.begin(), paramName.end(), '.', '_');
        return paramName;
    });
#else
INSTANTIATE_TEST_SUITE_P(
    GGUFDownloaderPullHfModelTests,
    GGUFDownloaderPullHfModelParameterized,
    ::testing::ValuesIn(ggufParams));
#endif

#ifndef _WIN32
INSTANTIATE_TEST_SUITE_P(
    GGUFDownloaderPullHfModelWithServerTests,
    GGUFDownloaderPullHfModelParameterizedWithServer,
    ::testing::ValuesIn(ggufServerStartParams),
    [](const ::testing::TestParamInfo<GGUFDownloaderPullHfModelParameterizedWithServer::ParamType>& info) {
        auto paramTuple = info.param;
        std::string paramName = ovms::joins({
                                                std::get<0>(paramTuple),  // sourceModel
                                                std::get<1>(paramTuple)   // ggufFilename
                                            },
            "_");
        std::replace(paramName.begin(), paramName.end(), '-', '_');
        std::replace(paramName.begin(), paramName.end(), '/', '_');
        std::replace(paramName.begin(), paramName.end(), ':', '_');
        std::replace(paramName.begin(), paramName.end(), '.', '_');
        return paramName;
    });
#else
INSTANTIATE_TEST_SUITE_P(
    GGUFDownloaderPullHfModelWithServerTests,
    GGUFDownloaderPullHfModelParameterizedWithServer,
    ::testing::ValuesIn(ggufServerStartParams));
#endif

TEST_F(GGUFDownloaderPullHfModel, PositiveDownload) {
    // curl -L   -H "Authorization: Bearer $HF_TOKEN"   -o DeepSeek‑R1‑Distill‑Qwen‑7B‑Q4_K_M.gguf   https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    // https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string sourceModel = "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF";
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    const std::string filenamePrefix = "/resolve/main/";
    const std::string ggufFilename = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf";
    const std::string hfEndpoint = "https://huggingface.co/";
    // const std::string hfEndpoint = "https://www.modelscope.cn";
    auto status = ovms::GGUFDownloader::downloadWithCurl(hfEndpoint, sourceModel, filenamePrefix, ggufFilename, downloadPath);
    ASSERT_TRUE(status.ok()) << status.string();
    std::string fullPath = ovms::FileSystem::joinPath({downloadPath, ggufFilename});
    bool exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok());
    ASSERT_TRUE(exist);
    // check size of the file with std::filesystem
    size_t fileSize = 0;
    std::filesystem::path filePath(fullPath);
    fileSize = std::filesystem::file_size(filePath);
    EXPECT_EQ(4683073248, fileSize);
}

TEST_F(GGUFDownloaderPullHfModel, PositiveDownloadMultipleQuantizationsWithNoOverride) {
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    ovms::HFSettingsImpl hfSettings;
    hfSettings.overwriteModels = false;
    hfSettings.sourceModel = "unsloth/Llama-3.2-1B-Instruct-GGUF";
    hfSettings.downloadPath = downloadPath;
    hfSettings.downloadType = ovms::ModelDownlaodType::GGUF_DOWNLOAD;
    const std::string hfEndpoint = "https://huggingface.co/";
    std::unordered_map<std::string, size_t> quantizations{
        {"Llama-3.2-1B-Instruct-Q8_0.gguf",1321082528}};
    for (const auto& [ggufFilename, expectedSize] : quantizations) {
        hfSettings.ggufFilename = ggufFilename;
        ovms::GGUFDownloader downloader(hfEndpoint, hfSettings);
        auto status = downloader.downloadModel();
        ASSERT_TRUE(status.ok()) << status.string();
        std::string fullPath = ovms::FileSystem::joinPath({downloadPath, hfSettings.sourceModel, ggufFilename});
        bool exist = false;
        status = ovms::LocalFileSystem::exists(fullPath, &exist);
        EXPECT_TRUE(status.ok()) << status.string();
        EXPECT_TRUE(exist) << "File " << fullPath << " does not exist after download";
        if (!exist) continue;
        // check size of the file with std::filesystem
        size_t fileSize = 0;
        std::filesystem::path filePath(fullPath);
        fileSize = std::filesystem::file_size(filePath);
        EXPECT_EQ(expectedSize, fileSize);
    }
}

TEST_F(GGUFDownloaderPullHfModel, ShouldSkipDownloadWithNoOverrideWhenSomePartsExist) {
// Plan:
// 1. Create (touch file) first part of the model. With no override it should fail to download
// 2. With override it should download both parts
// 3. Remove first part
// 4. Try to download with no override - it should fail 
// 5. With override it should download both parts
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string sourceModel = "Qwen/Qwen2.5-7B-Instruct-GGUF";
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    const std::string filenamePrefix = "/resolve/main/";
    const std::string ggufFilename = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";
    const std::string ggufFilenamePart2 = "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf";
    const std::string hfEndpoint = "https://huggingface.co/";
    auto filePathPart1 = ovms::FileSystem::joinPath({downloadPath, sourceModel, ggufFilename});
    ovms::HFSettingsImpl hfSettings;
    hfSettings.overwriteModels = false;
    hfSettings.sourceModel = sourceModel;
    hfSettings.downloadPath = downloadPath;
    hfSettings.downloadType = ovms::ModelDownlaodType::GGUF_DOWNLOAD;
    hfSettings.ggufFilename = ggufFilename;
    ovms::GGUFDownloader downloader(hfEndpoint, hfSettings);
    auto dirPath = ovms::FileSystem::joinPath({downloadPath, sourceModel});
    // create dir
    std::filesystem::create_directories(dirPath);
    // create file
    auto filePath = ovms::FileSystem::joinPath({dirPath, ggufFilename});
    auto status = downloader.downloadModel();
    ASSERT_TRUE(status.ok()) << status.string();
    // check that part2 does not exist
    bool exist = false;
    status = ovms::LocalFileSystem::exists(ovms::FileSystem::joinPath({dirPath, ggufFilenamePart2}), &exist);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_FALSE(exist) << "File " << ggufFilenamePart2 << " should not exist";
    // Part 2 with override should succeed
    hfSettings.overwriteModels = true;
    status = downloader.downloadModel();
    ASSERT_TRUE(status.ok()) << status.string();
    std::string fullPathPart1 = ovms::FileSystem::joinPath({downloadPath, sourceModel, ggufFilename});
    std::string fullPathPart2 = ovms::FileSystem::joinPath({downloadPath, sourceModel, ggufFilenamePart2});
    exist = false;
    status = ovms::LocalFileSystem::exists(fullPathPart1, &exist);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_TRUE(exist) << "File " << fullPathPart1 << " does not exist after download";
    exist = false;
    status = ovms::LocalFileSystem::exists(fullPathPart2, &exist);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_TRUE(exist) << "File " << fullPathPart2 << " does not exist after download";
    // Part 3
    // remove fullPathPart1
    std::filesystem::remove(fullPathPart1);
    hfSettings.overwriteModels = false;
    status = downloader.downloadModel();
    ASSERT_EQ(status.getCode(), ovms::StatusCode::INTERNAL_ERROR) << status.string();



////////////////////////////////
///
////////////
////////////
/// Plan:
/// 1. Create (touch file) first part of the model. With no override it shoudl fail to download
/// 2. With override it should download both parts
/// 3. Remove first part
/// 4. Try to download with no override - it should fail 
/// 5. With override it should download both parts
}


TEST_F(GGUFDownloaderPullHfModel, PositiveDownloadMultipleQuantizationsWithOverride) {
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    ovms::HFSettingsImpl hfSettings;
    hfSettings.overwriteModels = true;
    hfSettings.sourceModel = "unsloth/Llama-3.2-1B-Instruct-GGUF";
    hfSettings.downloadPath = downloadPath;
    hfSettings.downloadType = ovms::ModelDownlaodType::GGUF_DOWNLOAD;
    const std::string hfEndpoint = "https://huggingface.co/";
    std::vector<std::pair<std::string, size_t>> quantizations{
        {"Llama-3.2-1B-Instruct-Q4_K_M.gguf",807694368},
        {"Llama-3.2-1B-Instruct-Q2_K.gguf",580874272},
        {"Llama-3.2-1B-Instruct-Q8_0.gguf",1321082528}};
    size_t fileSize = 0;
    auto graphPbtxtPath = ovms::FileSystem::joinPath({downloadPath, hfSettings.sourceModel, "graph.pbtxt"});
    ////////
    // first we pull first with override true to see if it worksa even if no files exist
    ////////
    hfSettings.ggufFilename = quantizations[0].first;
    auto downloader = std::make_unique<ovms::GGUFDownloader>(hfEndpoint, hfSettings);
    auto status = downloader->downloadModel();
    ASSERT_TRUE(status.ok()) << status.string();
    std::string fullPath = ovms::FileSystem::joinPath({downloadPath, hfSettings.sourceModel, hfSettings.ggufFilename.value()});
    bool exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_TRUE(exist) << "File " << fullPath << " does not exist after download";
    if (exist) {
        std::filesystem::path filePath(fullPath);
        fileSize = std::filesystem::file_size(filePath);
    }
    EXPECT_EQ(quantizations[0].second, fileSize);
    
    ///////
    // now we pull with no override and it should skip download
    ///////
    hfSettings.overwriteModels = false;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    status = downloader->downloadModel();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_LE(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count(), 500) << "Download took too long, probably it was not skipped";

    ///////
    // now we pull another quantization with override true and expect previous file to exist
    ///////

    hfSettings.overwriteModels = true;
    hfSettings.ggufFilename = quantizations[1].first;
    status = downloader->downloadModel();
    ASSERT_TRUE(status.ok()) << status.string();
    fullPath = ovms::FileSystem::joinPath({downloadPath, hfSettings.sourceModel, hfSettings.ggufFilename.value()});
    exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_TRUE(exist) << "File " << fullPath << " does not exist after download";
    if (exist) {
        std::filesystem::path filePath(fullPath);
        fileSize = std::filesystem::file_size(filePath);
    }
    EXPECT_EQ(quantizations[1].second, fileSize);
    // now we check previous quantization
    fullPath = ovms::FileSystem::joinPath({downloadPath, hfSettings.sourceModel, quantizations[0].first});
    exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_TRUE(exist) << "File " << fullPath << " does not exist after download";
}


// TODO:
// -> no overwrite only some of the files exist
// -> no overwrite all files exist
// -> overwrite dir exist not files
// -> overwrite some files exist

TEST_F(GGUFDownloaderPullHfModel, PositiveMultipartModel) {
    SKIP_AND_EXIT_IF_NO_GGUF();
    const std::string sourceModel = "Qwen/Qwen2.5-7B-Instruct-GGUF";
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    const std::string filenamePrefix = "/resolve/main/";
    const std::string ggufFilename = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";
    const std::string hfEndpoint = "https://huggingface.co/";
    auto status = ovms::GGUFDownloader::downloadWithCurl(hfEndpoint, sourceModel, filenamePrefix, ggufFilename, downloadPath);
    ASSERT_TRUE(status.ok()) << status.string();

    std::string fullPathPart1 = ovms::FileSystem::joinPath({downloadPath, ggufFilename});
    bool existPart1 = false;
    status = ovms::LocalFileSystem::exists(fullPathPart1, &existPart1);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_TRUE(existPart1);
    size_t fileSize = 0;
    std::filesystem::path filePathPart1(fullPathPart1);
    fileSize = std::filesystem::file_size(filePathPart1);
    EXPECT_EQ(3993201344, fileSize);

    const std::string ggufFileNamePart2 = "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf";
    std::string fullPathPart2 = ovms::FileSystem::joinPath({downloadPath, ggufFileNamePart2});
    bool existPart2 = false;
    status = ovms::LocalFileSystem::exists(fullPathPart2, &existPart2);
    ASSERT_TRUE(status.ok()) << status.string();
    ASSERT_TRUE(existPart2);
    std::filesystem::path filePathPart2(fullPathPart2);
    fileSize = std::filesystem::file_size(filePathPart2);
    EXPECT_EQ(689872288, fileSize);
}

class GGUFDownloaderPullHfModelGGUFFilenameParameterizedNegative : public GGUFDownloaderPullHfModel, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};

TEST_P(GGUFDownloaderPullHfModelGGUFFilenameParameterizedNegative, NonMatchingParts) {
    const std::string hfEndpoint = std::get<1>(GetParam());
    const std::string ggufFilename = std::get<0>(GetParam());
    // to many zeros
    const std::string sourceModel = "Qwen/Qwen2.5-7B-Instruct-GGUF";
    const std::string downloadPath = ovms::FileSystem::appendSlash(directoryPath);
    const std::string filenamePrefix = "/resolve/main/";
    //ggufFilename = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";
    auto status = ovms::GGUFDownloader::downloadWithCurl(hfEndpoint, sourceModel, filenamePrefix, ggufFilename, downloadPath);
    EXPECT_EQ(status.getCode(), ovms::StatusCode::PATH_INVALID) << status.string();
    std::string fullPath = ovms::FileSystem::joinPath({downloadPath, ggufFilename});
    bool exist = false;
    status = ovms::LocalFileSystem::exists(fullPath, &exist);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_FALSE(exist);
    // too long second part
}

std::vector<std::tuple<std::string, std::string>> ggufPartsParams = {
    std::make_tuple("qwen2.5-7b-instruct-q4_k_m-000001-of-00002.gguf", "https://modelscope.cn/"),
    std::make_tuple("qwen2.5-7b-instruct-q4_k_m-000001-of-00002.gguf", "https://huggingface.co/"),
    std::make_tuple("qwen2.5-7b-instruct-q4_k_m-00001-of-000002.gguf", "https://huggingface.co/"),
    std::make_tuple("qwen2.5-7b-instruct-q4_k_m-0001-of-00002.gguf", "https://huggingface.co/"),
    std::make_tuple("qwen2.5-7b-instruct-q4_k_m-00001-of-0002.gguf", "https://huggingface.co/"),
    std::make_tuple("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf", "https://huggingface.co/")};

#ifndef _WIN32
INSTANTIATE_TEST_SUITE_P(
    GGUF,
    GGUFDownloaderPullHfModelGGUFFilenameParameterizedNegative,
    ::testing::ValuesIn(ggufPartsParams),
    [](const ::testing::TestParamInfo<GGUFDownloaderPullHfModelGGUFFilenameParameterizedNegative::ParamType>& info) {
        auto paramTuple = info.param;
        std::string paramName = ovms::joins({
                                                std::get<0>(paramTuple),  // ggufFilename
                                                std::get<1>(paramTuple)   // hfEndpoint
                                            },
            "_");
        std::replace(paramName.begin(), paramName.end(), '-', '_');
        std::replace(paramName.begin(), paramName.end(), '/', '_');
        std::replace(paramName.begin(), paramName.end(), ':', '_');
        std::replace(paramName.begin(), paramName.end(), '.', '_');
        return paramName;
    });
#else
INSTANTIATE_TEST_SUITE_P(
    GGUF,
    GGUFDownloaderPullHfModelGGUFFilenameParameterizedNegative,
    ::testing::ValuesIn(ggufPartsParams));
#endif

class GGUFDownloaderMultipartUtils : public ::testing::Test {
};

TEST_F(GGUFDownloaderMultipartUtils, PreparePartFilenamePositive) {
    std::string ggufFilename = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";
    EXPECT_EQ("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", ovms::GGUFDownloader::preparePartFilename(ggufFilename, 1, 2));
    EXPECT_EQ("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", ovms::GGUFDownloader::preparePartFilename(ggufFilename, 1, 2));
    ggufFilename = "Mixtral-8x22B-v0.1-Q3_K_M-00001-of-00005.gguf";
    EXPECT_EQ("Mixtral-8x22B-v0.1-Q3_K_M-00001-of-00005.gguf", ovms::GGUFDownloader::preparePartFilename(ggufFilename, 1, 5));
    EXPECT_EQ("Mixtral-8x22B-v0.1-Q3_K_M-00003-of-00005.gguf", ovms::GGUFDownloader::preparePartFilename(ggufFilename, 3, 5));
    EXPECT_EQ("Mixtral-8x22B-v0.1-Q3_K_M-00005-of-00005.gguf", ovms::GGUFDownloader::preparePartFilename(ggufFilename, 5, 5));
}

TEST_F(GGUFDownloaderMultipartUtils, PreparePartFilenameNegative) {
    std::string ggufFilename = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, 0, 2), std::invalid_argument);
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, -1, 2), std::invalid_argument);
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, 3, 2), std::invalid_argument);
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, 1, 1), std::invalid_argument);
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, 1, 0), std::invalid_argument);
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, 1, -1), std::invalid_argument);
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, 1, 100000), std::invalid_argument);
    EXPECT_THROW(ovms::GGUFDownloader::preparePartFilename(ggufFilename, 100000, 99999), std::invalid_argument);
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
    GTEST_SKIP() << "Assuming we could run git in shell we could retrieve gguf files like this";
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
