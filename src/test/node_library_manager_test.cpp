//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../custom_node_library_manager.hpp"
#include "test_utils.hpp"

using namespace ovms;

TEST(NodeLibraryManagerTest, NewManagerExpectMissingLibrary) {
    CustomNodeLibraryManager manager;
    NodeLibrary library;
    auto status = manager.getLibrary("random_name", library);
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_MISSING);
}

TEST(NodeLibraryManagerTest, SuccessfullLibraryLoadingAndExecution) {
    CustomNodeLibraryManager manager;
    NodeLibrary library;
    auto status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_mock.so");
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.getLibrary("random_name", library);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_NE(library.execute, nullptr);
    ASSERT_NE(library.getInputsInfo, nullptr);
    ASSERT_NE(library.getOutputsInfo, nullptr);
    ASSERT_NE(library.release, nullptr);
    EXPECT_EQ(library.execute(nullptr, 0, nullptr, nullptr, nullptr, 0), 1);
    EXPECT_EQ(library.getInputsInfo(nullptr, nullptr, nullptr, 0), 2);
    EXPECT_EQ(library.getOutputsInfo(nullptr, nullptr, nullptr, 0), 3);
    EXPECT_EQ(library.release(nullptr), 4);
}

TEST(NodeLibraryManagerTest, LibraryLoadingDuplicateName) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_mock.so");
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_mock.so");
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_ALREADY_LOADED);
}

TEST(NodeLibraryManagerTest, LibraryLoadingDuplicatePath) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("library_A", "/ovms/bazel-bin/src/lib_node_mock.so");
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.loadLibrary("library_B", "/ovms/bazel-bin/src/lib_node_mock.so");
    EXPECT_EQ(status, StatusCode::OK);
}

TEST(NodeLibraryManagerTest, LibraryLoadingMissingImplementation) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_missing_implementation.so");
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM);
}

TEST(NodeLibraryManagerTest, TryLoadingCorruptedLibraryNextLoadCorrectLibrary) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_missing_implementation.so");
    ASSERT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM);
    status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_mock.so");
    EXPECT_EQ(status, StatusCode::OK);
}

TEST(NodeLibraryManagerTest, LibraryLoadingMissingFile) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/tmp/non_existing_library_file");
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED_OPEN);
}

TEST(NodeLibraryManagerTest, ErrorWhenLibraryPathNotEscaped) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/tmp/../my_dir/non_existing_library_file");
    EXPECT_EQ(status, StatusCode::PATH_INVALID);
}

class ModelManagerNodeLibraryTest : public TestWithTempDir {};

TEST_F(ModelManagerNodeLibraryTest, LoadCustomNodeLibrary) {
    const char* config = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"}
        ]})";
    std::string fileToReload = directoryPath + "/ovms_config_file1.json";
    createConfigFileWithContent(config, fileToReload);
    ConstructorEnabledModelManager manager;
    NodeLibrary library;
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.getCustomNodeLibraryManager().getLibrary("lib1", library);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_NE(library.execute, nullptr);
    ASSERT_NE(library.getInputsInfo, nullptr);
    ASSERT_NE(library.getOutputsInfo, nullptr);
    ASSERT_NE(library.release, nullptr);
    EXPECT_EQ(library.execute(nullptr, 0, nullptr, nullptr, nullptr, 0), 1);
    EXPECT_EQ(library.getInputsInfo(nullptr, nullptr, nullptr, 0), 2);
    EXPECT_EQ(library.getOutputsInfo(nullptr, nullptr, nullptr, 0), 3);
    EXPECT_EQ(library.release(nullptr), 4);
}

TEST_F(ModelManagerNodeLibraryTest, FailLoadingCorruptedCustomNodeLibrary) {
    const char* config = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_missing_implementation.so"}
        ]})";
    std::string fileToReload = directoryPath + "/ovms_config_file1.json";
    createConfigFileWithContent(config, fileToReload);
    ConstructorEnabledModelManager manager;
    NodeLibrary library;
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.getCustomNodeLibraryManager().getLibrary("lib1", library);
    ASSERT_EQ(status, StatusCode::NODE_LIBRARY_MISSING);
    EXPECT_EQ(library.execute, nullptr);
    EXPECT_EQ(library.getInputsInfo, nullptr);
    EXPECT_EQ(library.getOutputsInfo, nullptr);
    EXPECT_EQ(library.release, nullptr);
}

TEST_F(ModelManagerNodeLibraryTest, AddAndRemoveLibrariesInConfigReload) {
    const char* configBefore = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"}
        ]})";
    const char* configAfter = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"},
            {"name": "lib2", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"}
        ]})";
    std::string fileToReload = directoryPath + "/ovms_config_file1.json";

    // Start with configBefore
    createConfigFileWithContent(configBefore, fileToReload);
    ConstructorEnabledModelManager manager;
    NodeLibrary lib1Before, lib2Before;
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1Before), StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib2", lib2Before), StatusCode::NODE_LIBRARY_MISSING);

    // Expect lib1 to be loaded but lib2 not
    EXPECT_NE(lib1Before.execute, nullptr);
    EXPECT_NE(lib1Before.getInputsInfo, nullptr);
    EXPECT_NE(lib1Before.getOutputsInfo, nullptr);
    EXPECT_NE(lib1Before.release, nullptr);
    EXPECT_EQ(lib2Before.execute, nullptr);
    EXPECT_EQ(lib2Before.getInputsInfo, nullptr);
    EXPECT_EQ(lib2Before.getOutputsInfo, nullptr);
    EXPECT_EQ(lib2Before.release, nullptr);

    // Reload with configAfter
    NodeLibrary lib1After, lib2After;
    createConfigFileWithContent(configAfter, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1After), StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib2", lib2After), StatusCode::OK);

    // Expect lib1 not to change and lib2 to be created after reload.
    EXPECT_EQ(lib1Before.execute, lib1After.execute);
    EXPECT_EQ(lib1Before.getInputsInfo, lib1After.getInputsInfo);
    EXPECT_EQ(lib1Before.getOutputsInfo, lib1After.getOutputsInfo);
    EXPECT_EQ(lib1Before.release, lib1After.release);
    EXPECT_NE(lib2After.execute, nullptr);
    EXPECT_NE(lib2After.getInputsInfo, nullptr);
    EXPECT_NE(lib2After.getOutputsInfo, nullptr);
    EXPECT_NE(lib2After.release, nullptr);

    // Reload with initial config (remove lib2 entry)
    NodeLibrary lib1Entry, lib2Entry;
    createConfigFileWithContent(configBefore, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1Entry), StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib2", lib2Entry), StatusCode::OK);

    // Expect lib1 not to change and lib2 to still be loaded
    EXPECT_EQ(lib1After.execute, lib1Entry.execute);
    EXPECT_EQ(lib1After.getInputsInfo, lib1Entry.getInputsInfo);
    EXPECT_EQ(lib1After.getOutputsInfo, lib1Entry.getOutputsInfo);
    EXPECT_EQ(lib1After.release, lib1Entry.release);
    EXPECT_EQ(lib2After.execute, lib2Entry.execute);
    EXPECT_EQ(lib2After.getInputsInfo, lib2Entry.getInputsInfo);
    EXPECT_EQ(lib2After.getOutputsInfo, lib2Entry.getOutputsInfo);
    EXPECT_EQ(lib2After.release, lib2Entry.release);
}
