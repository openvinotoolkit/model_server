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

class ListModelsTest : public TestWithTempDir {
};

using ovms::listServables;
using ovms::ServableType_t;

std::string dirTree(const std::string& path, const std::string& indent = "") {
    if (!std::filesystem::exists(path)) {
        SPDLOG_ERROR("Path does not exist: {}", path);
        return "NON_EXISTENT_PATH";
    }
    std::stringstream tree;
    // if is directory, add to stream its name followed by "/"
    // if is file, add to stream its name

    tree << indent;
    if (!indent.empty()) {
        tree << "|-- ";
    }

    tree << std::filesystem::path(path).filename().string();
    if (std::filesystem::is_directory(path)) {
        tree << "/";
    }
    tree << std::endl;
    if (!std::filesystem::is_directory(path)) {
        return tree.str();
    }
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::string passDownIndent = indent.empty() ? "|   " : (indent + "    ");
        tree << dirTree(entry.path().string(), passDownIndent);
    }
    return tree.str();
}

void logDirTree(const std::string& path) {
    SPDLOG_DEBUG("Directory tree:\n{}", dirTree(path));
}

void logListModels(const std::unordered_map<std::string, ServableType_t>& servablesList) {
    std::stringstream ss;
    ss << "List of servables:\n";
    ss << "Path\t\tType\n";
    for (const auto& [name, type] : servablesList) {
        ss << name << "\t\t" << (type == ServableType_t::SERVABLE_TYPE_MODEL ? "Model" : "MediapipeGraph") << "\n";
    }
    ss << "End of list\n";
    SPDLOG_DEBUG(ss.str());
}

class IsVersionDirTest : public TestWithTempDir {};
TEST_F(IsVersionDirTest, IsVersionDir) {
    using ovms::isMediapipeGraphDir;
    using ovms::isVersionDir;
    std::string modelName = "model";
    std::string versionDir = ovms::FileSystem::appendSlash(this->directoryPath) + "1";
    std::filesystem::create_directory(versionDir);

    // check if versionDir is a version dir
    logDirTree(this->directoryPath);
    EXPECT_EQ(isVersionDir(versionDir), true) << versionDir;
    // now create directory with non-numeric name and check if it is not a version dir
    std::string nonVersionDir = ovms::FileSystem::appendSlash(this->directoryPath) + "nonVersionDir";
    std::filesystem::create_directory(nonVersionDir);
    logDirTree(this->directoryPath);
    EXPECT_EQ(isVersionDir(nonVersionDir), false);
    // now create empty file with numeric name and check if it is not a version dir
    std::string emptyFile = ovms::FileSystem::appendSlash(this->directoryPath) + "2";
    std::ofstream file(emptyFile);
    file.close();
    logDirTree(this->directoryPath);
    EXPECT_EQ(isVersionDir(emptyFile), false);
}

class IsMediapipeGraphTest : public TestWithTempDir {};
TEST_F(IsMediapipeGraphTest, IsMediapipeGraph) {
    // create directory with OpenVINO IR model inside this->DirectoryPath
    using ovms::isMediapipeGraphDir;
    std::string modelName = "model";
    std::string graphPbtxtDirectory = ovms::FileSystem::appendSlash(this->directoryPath) + "graph.pbtxt";
    std::filesystem::create_directory(graphPbtxtDirectory);
    logDirTree(this->directoryPath);
    EXPECT_EQ(isMediapipeGraphDir(this->directoryPath), false) << "should fail because it only contains graph.pbtxt directory";

    // now create empty file with name model.pbtxt and check if it is a mediapipe graph
    std::string graphFile = ovms::FileSystem::appendSlash(this->directoryPath) + "model.pbtxt";
    std::ofstream graph(graphFile);
    graph.close();
    logDirTree(this->directoryPath);
    EXPECT_EQ(isMediapipeGraphDir(this->directoryPath), false) << "should fail because it only contains model.pbtxt file not graph.pbtxt";

    // create empty graph.pbtxt and verify that it is a mediapipe graph
    std::string subdirectoryPath = ovms::FileSystem::appendSlash(this->directoryPath) + "subdirectory";
    std::string graphFile2 = ovms::FileSystem::appendSlash(subdirectoryPath) + "graph.pbtxt";
    std::filesystem::create_directory(subdirectoryPath);
    std::ofstream graph2(graphFile2);
    graph2.close();
    logDirTree(subdirectoryPath);
    EXPECT_EQ(isMediapipeGraphDir(subdirectoryPath), true);
}

class HasRequiredExtensionsTest : public TestWithTempDir {};

TEST_F(HasRequiredExtensionsTest, HasRequiredExtensions) {
    using ovms::hasRequiredExtensions;
    // create directory with OpenVINO IR model inside this->DirectoryPath
    std::string modelName = "model";
    std::string versionDir = ovms::FileSystem::appendSlash(this->directoryPath) + "1";
    std::filesystem::create_directory(versionDir);
    EXPECT_EQ(hasRequiredExtensions(versionDir, ovms::OV_MODEL_FILES_EXTENSIONS), false) << "test on empty failed";

    std::string binFile = ovms::FileSystem::appendSlash(versionDir) + modelName + ".bin";
    std::ofstream bin(binFile);
    bin.close();
    logDirTree(this->directoryPath);
    EXPECT_EQ(hasRequiredExtensions(versionDir, ovms::OV_MODEL_FILES_EXTENSIONS), false) << "should fails since incomplete";
    std::string xmlFile = ovms::FileSystem::appendSlash(versionDir) + modelName + ".xml";
    std::ofstream xml(xmlFile);
    xml.close();
    logDirTree(this->directoryPath);
    EXPECT_EQ(hasRequiredExtensions(versionDir, ovms::OV_MODEL_FILES_EXTENSIONS), true);
    EXPECT_EQ(hasRequiredExtensions(versionDir, std::array<const char*, 1>{".intel"}), false);
}

class GetPartialPathTest : public TestWithTempDir {};
TEST_F(GetPartialPathTest, GetPartialPath) {
    using ovms::getPartialPath;
    std::string modelFileName = "model.bin";
    int i = 4;
    std::string versionDir = ovms::FileSystem::appendSlash(this->directoryPath);
    while (i > 0) {
        versionDir = ovms::FileSystem::appendSlash(versionDir) + std::to_string(i--);
        std::filesystem::create_directory(versionDir);
    }
    std::string binFile = ovms::FileSystem::appendSlash(versionDir) + modelFileName;
    std::ofstream bin(binFile);
    bin.close();
    logDirTree(this->directoryPath);
    auto sep = ovms::FileSystem::getOsSeparator();
    EXPECT_EQ(getPartialPath(binFile, 0), modelFileName);
    EXPECT_EQ(getPartialPath(binFile, 1), std::to_string(1) + sep + modelFileName);
    EXPECT_EQ(getPartialPath(binFile, 4), std::to_string(4) + sep +
                                              std::to_string(3) + sep +
                                              std::to_string(2) + sep +
                                              std::to_string(1) + sep + modelFileName);
    // expect to get exception
    EXPECT_THROW(getPartialPath(versionDir, 70), std::runtime_error);
}

TEST_F(ListModelsTest, EmptyDir) {
    std::string emptyDir = ovms::FileSystem::appendSlash(this->directoryPath) + "emptyDir";
    std::filesystem::create_directory(emptyDir);
    std::unordered_map<std::string, ServableType_t> servablesList = listServables(emptyDir);
    EXPECT_EQ(servablesList.size(), 0);
    logDirTree(this->directoryPath);
    logListModels(servablesList);
}

TEST_F(ListModelsTest, OpenVINOIRCompleteModel) {
    std::string modelName = "model";
    std::string modelDir = ovms::FileSystem::appendSlash(this->directoryPath) + modelName;
    std::filesystem::create_directory(modelDir);
    std::string versionDir = ovms::FileSystem::appendSlash(modelDir) + "3";
    std::filesystem::create_directory(versionDir);

    std::string binFile = ovms::FileSystem::appendSlash(versionDir) + modelName + ".bin";
    std::string xmlFile = ovms::FileSystem::appendSlash(versionDir) + modelName + ".xml";
    std::ofstream bin(binFile);
    bin.close();
    std::ofstream xml(xmlFile);
    xml.close();
    logDirTree(this->directoryPath);
    std::unordered_map<std::string, ServableType_t> servablesList = listServables(this->directoryPath);
    EXPECT_EQ(servablesList.size(), 1);
    EXPECT_THAT(servablesList, ::testing::UnorderedElementsAre(::testing::Pair(modelName, ServableType_t::SERVABLE_TYPE_MODEL)));
    logListModels(servablesList);
}

const std::string GRAPH_FILE_NAME = "graph.pbtxt";
TEST_F(ListModelsTest, MediapipeGraph) {
    std::string dirName = "graphDirectory";
    std::string graphDir = ovms::FileSystem::appendSlash(this->directoryPath) + dirName;
    std::filesystem::create_directory(graphDir);
    std::string graphFile = ovms::FileSystem::appendSlash(graphDir) + GRAPH_FILE_NAME;
    std::ofstream graph(graphFile);
    graph.close();
    logDirTree(this->directoryPath);
    auto servablesList = listServables(this->directoryPath);
    EXPECT_EQ(servablesList.size(), 1);
    EXPECT_THAT(servablesList, ::testing::UnorderedElementsAre(::testing::Pair(dirName, ServableType_t::SERVABLE_TYPE_MEDIAPIPEGRAPH)));
    logListModels(servablesList);
}

TEST_F(ListModelsTest, BothMediapipeGraphAndModelPresent) {
    // first create graphDir in which we have graph.pbtxt file and inside graphDir we also have
    // version 1 directory which contains model.onnx file
    std::string dirName = "graphDirectory";
    std::string graphDir = ovms::FileSystem::appendSlash(this->directoryPath) + dirName;
    std::filesystem::create_directory(graphDir);
    std::string graphFile = ovms::FileSystem::appendSlash(graphDir) + GRAPH_FILE_NAME;
    std::ofstream graph(graphFile);
    graph.close();
    std::string versionDir = ovms::FileSystem::appendSlash(graphDir) + "1";
    std::filesystem::create_directory(versionDir);
    std::string modelFile = ovms::FileSystem::appendSlash(versionDir) + "model.onnx";
    std::ofstream model(modelFile);
    model.close();
    logDirTree(this->directoryPath);
    auto servablesList = listServables(this->directoryPath);
    EXPECT_EQ(servablesList.size(), 1);
    EXPECT_THAT(servablesList, ::testing::UnorderedElementsAre(::testing::Pair(dirName, ServableType_t::SERVABLE_TYPE_MEDIAPIPEGRAPH)));
    logListModels(servablesList);
}

TEST_F(ListModelsTest, GraphPbtxtPresentInsideVersionDirExpectModel) {
    // first create graphDir in which we have graph.pbtxt file and inside graphDir we also have
    // version 1 directory which contains model.onnx file
    std::string dirName = "modelDirectory";
    std::string dirPath = ovms::FileSystem::appendSlash(this->directoryPath) + dirName;
    std::filesystem::create_directory(dirPath);
    std::string versionDir = ovms::FileSystem::appendSlash(dirPath) + "1";
    std::filesystem::create_directory(versionDir);
    std::string graphFile = ovms::FileSystem::appendSlash(versionDir) + GRAPH_FILE_NAME;
    std::ofstream graph(graphFile);
    graph.close();
    std::string modelFile = ovms::FileSystem::appendSlash(versionDir) + "model.onnx";
    std::ofstream model(modelFile);
    model.close();
    logDirTree(this->directoryPath);
    auto servablesList = listServables(this->directoryPath);
    EXPECT_EQ(servablesList.size(), 1);
    EXPECT_THAT(servablesList, ::testing::UnorderedElementsAre(::testing::Pair(dirName, ServableType_t::SERVABLE_TYPE_MODEL)));
    logListModels(servablesList);
}
TEST_F(ListModelsTest, NestedDirShouldShowPath) {
    // we will have
    // directory structure like this:
    // directoryPath
    //   |-- resnet
    //   |   |-- rn50
    //   |   |   |-- 1
    //   |   |       |-- model.onnx
    // we should have renset/rn50 Model
    std::string modelName = "resnet";
    std::string modelDir = ovms::FileSystem::appendSlash(this->directoryPath) + modelName;
    std::filesystem::create_directory(modelDir);
    std::string rn50DirName = "rn50";
    std::string rn50Dir = ovms::FileSystem::appendSlash(modelDir) + rn50DirName;
    std::filesystem::create_directory(rn50Dir);
    std::string versionDir = ovms::FileSystem::appendSlash(rn50Dir) + "1";
    std::filesystem::create_directory(versionDir);
    std::string modelFile = ovms::FileSystem::appendSlash(versionDir) + "model.onnx";
    std::ofstream model(modelFile);
    model.close();
    logDirTree(this->directoryPath);
    auto servablesList = listServables(this->directoryPath);
    EXPECT_EQ(servablesList.size(), 1);
    EXPECT_THAT(servablesList, ::testing::UnorderedElementsAre(::testing::Pair(ovms::FileSystem::appendSlash(modelName) + rn50DirName, ServableType_t::SERVABLE_TYPE_MODEL)));
    logListModels(servablesList);
}

TEST_F(ListModelsTest, NestedDirectoryStructure) {
    // we will have
    // directory structure like this:
    // directoryPath
    //   |-- model
    //   |   |-- 1
    //   |       |-- model.onnx
    //   |-- graphDirectory
    //   |   |-- graph.pbtxt
    //   |-- meta
    //   |   |-- llama3
    //   |   |   |-- graph.pbtxt
    //   |   |--llama2
    //   |       |-- graph.pbtxt
    //   |-- resnet
    //   |   |-- rn50
    //   |   |   |-- 1
    //   |   |       |-- model.onnx
    //   |   |-- rn101
    //   |       |-- 5
    //   |           |-- model.tflite
    std::string modelDirName = "model";
    std::string modelDir = ovms::FileSystem::appendSlash(this->directoryPath) + modelDirName;
    std::filesystem::create_directory(modelDir);
    std::string versionDir = ovms::FileSystem::appendSlash(modelDir) + "1";
    std::filesystem::create_directory(versionDir);
    std::string modelFile = ovms::FileSystem::appendSlash(versionDir) + "model.onnx";
    std::ofstream model(modelFile);
    model.close();
    std::string graphDirName = "graphDirectory";
    std::string graphDir = ovms::FileSystem::appendSlash(this->directoryPath) + graphDirName;
    std::filesystem::create_directory(graphDir);
    std::string graphFile = ovms::FileSystem::appendSlash(graphDir) + GRAPH_FILE_NAME;
    std::ofstream graph(graphFile);
    graph.close();
    std::string metaDirName = "meta";
    std::string metaDir = ovms::FileSystem::appendSlash(this->directoryPath) + metaDirName;
    std::filesystem::create_directory(metaDir);
    std::string llama3DirName = "llama3";
    std::string llama3Dir = ovms::FileSystem::appendSlash(metaDir) + llama3DirName;
    std::filesystem::create_directory(llama3Dir);
    std::string graphFile2 = ovms::FileSystem::appendSlash(llama3Dir) + GRAPH_FILE_NAME;
    std::ofstream graph2(graphFile2);
    graph2.close();
    std::string llama2DirName = "llama2";
    std::string llama2Dir = ovms::FileSystem::appendSlash(metaDir) + llama2DirName;
    std::filesystem::create_directory(llama2Dir);
    std::string graphFile3 = ovms::FileSystem::appendSlash(llama2Dir) + GRAPH_FILE_NAME;
    std::ofstream graph3(graphFile3);
    graph3.close();
    std::string resnetDirName = "resnet";
    std::string resnetDir = ovms::FileSystem::appendSlash(this->directoryPath) + resnetDirName;
    std::filesystem::create_directory(resnetDir);
    std::string rn50DirName = "rn50";
    std::string rn50Dir = ovms::FileSystem::appendSlash(resnetDir) + rn50DirName;
    std::filesystem::create_directory(rn50Dir);
    std::string rn50VersionDir = ovms::FileSystem::appendSlash(rn50Dir) + "1";
    std::filesystem::create_directory(rn50VersionDir);
    std::string modelFile2 = ovms::FileSystem::appendSlash(rn50VersionDir) + "model.onnx";
    std::ofstream model2(modelFile2);
    model2.close();
    std::string rn101DirName = "rn101";
    std::string rn101Dir = ovms::FileSystem::appendSlash(resnetDir) + rn101DirName;
    std::filesystem::create_directory(rn101Dir);
    std::string rn101VersionDir = ovms::FileSystem::appendSlash(rn101Dir) + "5";
    std::filesystem::create_directory(rn101VersionDir);
    std::string modelFile3 = ovms::FileSystem::appendSlash(rn101VersionDir) + "model.tflite";
    std::ofstream model3(modelFile3);
    model3.close();
    logDirTree(this->directoryPath);
    auto servablesList = listServables(this->directoryPath);
    EXPECT_EQ(servablesList.size(), 6);
    EXPECT_THAT(servablesList, ::testing::UnorderedElementsAre(
                                   ::testing::Pair(modelDirName, ServableType_t::SERVABLE_TYPE_MODEL),
                                   ::testing::Pair(graphDirName, ServableType_t::SERVABLE_TYPE_MEDIAPIPEGRAPH),
                                   ::testing::Pair(ovms::FileSystem::appendSlash(metaDirName) + llama2DirName, ServableType_t::SERVABLE_TYPE_MEDIAPIPEGRAPH),
                                   ::testing::Pair(ovms::FileSystem::appendSlash(metaDirName) + llama3DirName, ServableType_t::SERVABLE_TYPE_MEDIAPIPEGRAPH),
                                   ::testing::Pair(ovms::FileSystem::appendSlash(resnetDirName) + rn50DirName, ServableType_t::SERVABLE_TYPE_MODEL),
                                   ::testing::Pair(ovms::FileSystem::appendSlash(resnetDirName) + rn101DirName, ServableType_t::SERVABLE_TYPE_MODEL)));
    logListModels(servablesList);
}
