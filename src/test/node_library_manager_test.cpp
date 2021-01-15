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
    auto library = manager.getLibrary("random_name");
    EXPECT_EQ(library.execute, nullptr);
    EXPECT_EQ(library.releaseBuffer, nullptr);
    EXPECT_EQ(library.releaseTensors, nullptr);
}

TEST(NodeLibraryManagerTest, SuccessfullLibraryLoadingAndExecution) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_mock.so");
    ASSERT_EQ(status, StatusCode::OK);
    auto library = manager.getLibrary("random_name");
    ASSERT_NE(library.execute, nullptr);
    ASSERT_NE(library.releaseBuffer, nullptr);
    ASSERT_NE(library.releaseTensors, nullptr);
    EXPECT_EQ(library.execute(nullptr, 0, nullptr, nullptr, nullptr, 0), 1);
    EXPECT_EQ(library.releaseBuffer(nullptr), 2);
    EXPECT_EQ(library.releaseTensors(nullptr), 3);
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
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    auto library = manager.getCustomNodeLibraryManager().getLibrary("lib1");
    ASSERT_NE(library.execute, nullptr);
    ASSERT_NE(library.releaseBuffer, nullptr);
    ASSERT_NE(library.releaseTensors, nullptr);
    EXPECT_EQ(library.execute(nullptr, 0, nullptr, nullptr, nullptr, 0), 1);
    EXPECT_EQ(library.releaseBuffer(nullptr), 2);
    EXPECT_EQ(library.releaseTensors(nullptr), 3);
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
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    auto library = manager.getCustomNodeLibraryManager().getLibrary("lib1");
    EXPECT_EQ(library.execute, nullptr);
    EXPECT_EQ(library.releaseBuffer, nullptr);
    EXPECT_EQ(library.releaseTensors, nullptr);
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
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    auto lib1Before = manager.getCustomNodeLibraryManager().getLibrary("lib1");
    auto lib2Before = manager.getCustomNodeLibraryManager().getLibrary("lib2");

    // Reload with configAfter
    createConfigFileWithContent(configAfter, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    auto lib1After = manager.getCustomNodeLibraryManager().getLibrary("lib1");
    auto lib2After = manager.getCustomNodeLibraryManager().getLibrary("lib2");

    // Expect lib1 not to change and lib2 to be created after reload.
    EXPECT_EQ(lib1Before.execute, lib1After.execute);
    EXPECT_EQ(lib1Before.releaseBuffer, lib1After.releaseBuffer);
    EXPECT_EQ(lib1Before.releaseTensors, lib1After.releaseTensors);
    EXPECT_EQ(lib2Before.execute, nullptr);
    EXPECT_EQ(lib2Before.releaseBuffer, nullptr);
    EXPECT_EQ(lib2Before.releaseTensors, nullptr);
    EXPECT_NE(lib2After.execute, nullptr);
    EXPECT_NE(lib2After.releaseBuffer, nullptr);
    EXPECT_NE(lib2After.releaseTensors, nullptr);

    // Reload with initial config (remove lib2 entry)
    createConfigFileWithContent(configBefore, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    auto lib1Entry = manager.getCustomNodeLibraryManager().getLibrary("lib1");
    auto lib2Entry = manager.getCustomNodeLibraryManager().getLibrary("lib2");

    // Expect lib1 not to change and lib2 to still be loaded
    EXPECT_EQ(lib1After.execute, lib1Entry.execute);
    EXPECT_EQ(lib1After.releaseBuffer, lib1Entry.releaseBuffer);
    EXPECT_EQ(lib1After.releaseTensors, lib1Entry.releaseTensors);
    EXPECT_EQ(lib2After.execute, lib2Entry.execute);
    EXPECT_EQ(lib2After.releaseBuffer, lib2Entry.releaseBuffer);
    EXPECT_EQ(lib2After.releaseTensors, lib2Entry.releaseTensors);
}
