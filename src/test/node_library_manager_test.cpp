//*****************************************************************************
// Copyright 2020 Intel Corporation
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
    EXPECT_NE(library.execute, nullptr);
    EXPECT_NE(library.releaseBuffer, nullptr);
    EXPECT_NE(library.releaseTensors, nullptr);
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
    auto status = manager.loadLibrary("random_name", "/tmp/lib_node_missing_implementation.so");
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED);
}

TEST(NodeLibraryManagerTest, TryLoadingCorruptedLibraryNextLoadCorrectLibrary) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/tmp/lib_node_missing_implementation.so");
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED);
    status = manager.loadLibrary("random_name", "/ovms/bazel-bin/src/lib_node_mock.so");
    EXPECT_EQ(status, StatusCode::OK);
}
